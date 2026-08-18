#!/usr/bin/env bash
# Bring a finished training run's weights home, surviving a flaky link.
#
#   ./deployment/cluster/pull_checkpoint.sh --node jren313@100.95.103.25 \
#       [--run starvla_gr00t_n1d7_pipette_tipcrop_waist_split] [--wait]
#
# WHY IT RETRIES: on 2026-08-14 the first version gave up after a single
# rsync failure. Training had finished at 01:15, the pull started at 01:18
# and died at 01:36 with ssh error 255 having moved ~5 MB in 18 minutes --
# the link was already dying -- and the node then went unreachable with the
# only copy of a 3.5-hour run on it. A transport blip must cost a retry, not
# the run. So: probe until the host answers, wait out any active training,
# then retry the transfer until the byte count matches.
#
# --wait polls until training exits (use it right after launching); without
# it the script expects the run to be finished already.
#
# ONLY final_model + the loader's metadata. checkpoints/ holds a 9.5 GB .pt
# per save interval; final_model is the same weights at the last step and is
# what the bridge's STARVLA_PIPETTE_TIPCROP_WAIST_CKPT points at. On a
# fragile link, 9.5 GB beats 28.5 GB. Pass --all-checkpoints to override.
#
# THE SERVING LOADER NEEDS THREE FILES, not one: the .pt, plus config.yaml
# and dataset_statistics.json TWO DIRS UP from it (starVLA resolves them
# relative to the checkpoint path). Pulling only the .pt gives you a
# checkpoint that cannot be served.
set -uo pipefail

NODE=""; RUN="starvla_gr00t_n1d7_pipette_tipcrop_waist_split"; WAIT=0; ALL=0
SUFFIX="_b200"
while [ $# -gt 0 ]; do
  case "$1" in
    --node) NODE="$2"; shift 2;;
    --run) RUN="$2"; shift 2;;
    --suffix) SUFFIX="$2"; shift 2;;
    --wait) WAIT=1; shift;;
    --all-checkpoints) ALL=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
[ -n "$NODE" ] || { echo "--node USER@HOST is required" >&2; exit 2; }

DIR="Projects/starVLA/outputs/starvla/${RUN}${SUFFIX}"
LDIR="$HOME/$DIR"
LOG="$HOME/ckpt_pull/pull_${RUN}.log"
SSHO=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15
      -o ServerAliveCountMax=4)
mkdir -p "$HOME/ckpt_pull"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
rsh() { ssh "${SSHO[@]}" "$NODE" "$@" 2>/dev/null; }

say "watching $NODE:$DIR"
until rsh true; do sleep 60; done
say "host is answering"

# Resolve the remote home ONCE and use absolute paths from here on. ssh
# expands $HOME in a remote command, but rsync does NOT expand it inside a
# host:path spec -- it fails instantly with "error 23 / broken pipe" having
# moved 0 bytes, which looks exactly like a flaky link and will happily
# retry forever getting nowhere. (Cost 23 no-op attempts on 2026-08-14.)
RHOME=$(rsh 'echo $HOME' | tr -d '\r')
[ -n "$RHOME" ] || { say "FAILED: cannot resolve remote \$HOME"; exit 1; }
RDIR="$RHOME/$DIR"

if [ "$WAIT" = 1 ]; then
  while rsh 'pgrep -f "[t]rain_starvla" >/dev/null'; do sleep 120; done
  say "training exited — 60 s for the final save to flush"; sleep 60
fi

# Health snapshot BEFORE the transfer: if the box is thermally marginal (the
# failure mode that killed the last node) a second outage can take the
# evidence with it, and uptime/throttle reasons are what separate "it
# overheated" from "the network dropped".
say "--- node health ---"
rsh 'echo "boot: $(uptime -s)"; uptime
     nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit,\
clocks_throttle_reasons.active --format=csv,noheader 2>&1 | head -8' | tee -a "$LOG"

RSZ=$(rsh "stat -c %s $RDIR/final_model/pytorch_model.pt")
[ -n "$RSZ" ] || { say "FAILED: no final_model/pytorch_model.pt on the node"
                   rsh "ls -la $RDIR $RDIR/final_model" | tee -a "$LOG"
                   exit 1; }
say "remote checkpoint: $RSZ bytes"
mkdir -p "$LDIR/final_model"

for attempt in $(seq 1 40); do
  LSZ=$(stat -c %s "$LDIR/final_model/pytorch_model.pt" 2>/dev/null || echo 0)
  [ "$RSZ" = "$LSZ" ] && { say "checkpoint complete: $LSZ bytes"; break; }
  say "attempt $attempt — $LSZ / $RSZ bytes"
  rsync -a --partial --timeout=120 --bwlimit=40M -e "ssh ${SSHO[*]}" \
    "$NODE:$RDIR/final_model/pytorch_model.pt" \
    "$LDIR/final_model/pytorch_model.pt" 2>&1 | tail -2 | tee -a "$LOG"
  sleep 30
done

LSZ=$(stat -c %s "$LDIR/final_model/pytorch_model.pt" 2>/dev/null || echo 0)
[ "$RSZ" = "$LSZ" ] || { say "GAVE UP: $LSZ of $RSZ bytes. The partial is kept —"
                         say "  rerun this script to resume from it."; exit 1; }

rsync -a --partial -e "ssh ${SSHO[*]}" \
  "$NODE:$RDIR/config.yaml" "$NODE:$RDIR/config.full.yaml" \
  "$NODE:$RDIR/dataset_statistics.json" \
  "$NODE:$RDIR/summary.jsonl" "$LDIR/" 2>&1 | tail -1
[ "$ALL" = 1 ] && rsync -a --partial --info=progress2 -e "ssh ${SSHO[*]}" \
  "$NODE:$RDIR/checkpoints/" "$LDIR/checkpoints/" 2>&1 | tail -1

for f in config.yaml dataset_statistics.json; do
  [ -f "$LDIR/$f" ] && say "  have $f" || say "  MISSING $f — the loader needs it"
done

# wandb runs offline on the node (no credentials on a shared box); the run
# only reaches the web when it is synced from a machine that has the key.
# NOTE THE DOUBLE wandb/: the run lands in <run>/wandb/wandb/offline-run-*,
# not <run>/wandb/offline-run-*. A glob written for the shallower path finds
# nothing and the sync silently no-ops, so this searches instead of assuming.
WB=$(rsh "find $RDIR -type d -name 'offline-run-*' 2>/dev/null | head -1")
if [ -n "$WB" ] && command -v wandb >/dev/null; then
  say "syncing wandb from the pulled offline run"
  mkdir -p "$HOME/wandb_sync/$RUN"
  rsync -a -e "ssh ${SSHO[*]}" "$NODE:$WB/" "$HOME/wandb_sync/$RUN/" 2>/dev/null \
    && wandb sync "$HOME/wandb_sync/$RUN" >/dev/null 2>&1 \
    && say "  wandb synced" || say "  wandb sync failed (not fatal)"
fi

say "DONE -> $LDIR/final_model/pytorch_model.pt"
