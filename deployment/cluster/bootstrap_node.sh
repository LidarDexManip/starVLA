#!/usr/bin/env bash
# Stand up a fresh GPU node for starVLA pipette training, from nothing.
#
# WHY THIS EXISTS: the previous training box (100.121.16.2) died mid-pull and
# every hour of setup on it was lost, because the setup lived in shell history
# and a /tmp/launch_split.sh. This script is that setup, written down. Point it
# at a new node and it reproduces the run.
#
#   ./deployment/cluster/bootstrap_node.sh --node jren313@100.95.103.25
#
# RUNS FROM THE WORKSTATION, drives the node over ssh. Every phase is
# IDEMPOTENT -- re-running skips what is already done -- so a phase that dies
# on a flaky link is fixed by running the script again, not by unpicking it.
#
#   --node USER@HOST     required
#   --from PHASE         start at PHASE and continue (default: probe)
#   --only PHASE         run exactly one phase
#   --push-models        rsync HF weights from this workstation instead of
#                        letting the node download them (see ASSETS below)
#   --no-train           set everything up but do not launch
#   --list               print the phases and exit
#
# PHASES: probe conda env repo assets split verify train
#
# ---------------------------------------------------------------------------
# THE FIVE THINGS THAT COST TIME THE FIRST TIME. Each is handled below; they
# are written out because the next node will have the same edges.
#
# 1. TORCH MUST MATCH THE GPU ARCH. B200 is compute capability 10.0 (sm_100).
#    The workstation's env is torch 2.6.0+cu124, whose arch list stops at
#    sm_90 -- it imports fine and then fails at the first kernel. `_torch_spec`
#    picks the wheel from the compute cap the node reports; do not pin torch in
#    requirements.txt, it is a per-node fact.
#
# 2. THE BACKBONE IS GATED. nvidia/Cosmos-Reason2-2B and nvidia/GR00T-N1.7-3B
#    both 401 without a token, and the failure surfaces at build time as a
#    confusing HTTP error. Once cached, HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE
#    must be set or every run re-checks the hub and 401s again.
#
# 3. CUDA_VISIBLE_DEVICES DEFAULTS TO "0". accelerate then puts all 8 ranks on
#    GPU 0 and NCCL dies with "Duplicate GPU detected". It must be set
#    explicitly, and so must --multi_gpu (the launcher adds it above 1 proc).
#
# 4. DATA_MIX IS A SEPARATE ENV VAR. The launcher passes it as a CLI arg that
#    OVERRIDES the yaml, so editing the yaml's data_mix alone does nothing.
#
# 5. THE SPLIT IS BUILT ON THE NODE, and must be. The loader computes
#    normalisation statistics by globbing data/*/*.parquet over the dataset
#    DIRECTORY, not from meta/episodes.jsonl -- so a split that shares a
#    parquet tree normalises training data using the eval episodes. Separate
#    trees, symlinked. See build_splits.py.
#
# ASSETS: measured 2026-08-14, node 100.95.103.25 --
#     node -> HF CDN        22 MB/s
#     workstation -> node    7.7 MB/s
# so the node downloading its own ~14.5 GB (11 min) beats pushing it (31 min),
# and that is the default. --push-models is the fallback for a node with poor
# internet, or when you would rather no token touched a SHARED machine at all.
# Either way the token is NEVER written to the node: it goes as an env var on
# the download command only (readable via /proc/PID/environ by the owner and
# root, unlike argv), and `hf auth login` is deliberately not used because it
# persists ~/.cache/huggingface/token on a box other people log into.
set -uo pipefail

PHASES=(probe conda env repo assets split verify train)
NODE=""; FROM="probe"; ONLY=""; PUSH_MODELS=0; DO_TRAIN=1
GPUS_OVERRIDE=""; ACCUM_OVERRIDE=""; PER_DEVICE=16

REPO_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_REMOTE='$HOME/Projects/starVLA'
DATA_REMOTE='$HOME/Datasets'
ENV_NAME=starVLA
PY_VER=3.10
DATASET=birbirll/g1-pipette-tip-teleop-tipcrop
DATASET_DIR=g1-pipette-tip-teleop-tipcrop
MODELS=(nvidia/GR00T-N1.7-3B nvidia/Cosmos-Reason2-2B)
CONFIG_YAML=examples/realRobots/UnitreeG1_Pipette/train_files/starvla_gr00t_n1d7_pipette_tipcrop_waist_split.yaml
DATA_MIX=unitree_g1_pipette_tipcrop_train_waist_n1d7
RUN_ID=starvla_gr00t_n1d7_pipette_tipcrop_waist_split
# Appended to RUN_ID to name outputs/starvla/<dir>. The bridge profile's
# checkpoint var resolves this exact path, so changing it means changing
# gui/server.py's STARVLA_PIPETTE_TIPCROP_WAIST_CKPT default too.
RUN_SUFFIX=_b200

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS_OVERRIDE="$2"; shift 2;;
    --accum) ACCUM_OVERRIDE="$2"; shift 2;;
    --node) NODE="$2"; shift 2;;
    --from) FROM="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --push-models) PUSH_MODELS=1; shift;;
    --no-train) DO_TRAIN=0; shift;;
    --list) printf '%s\n' "${PHASES[@]}"; exit 0;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
[ -n "$NODE" ] || { echo "--node USER@HOST is required" >&2; exit 2; }

SSHO=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15
      -o ServerAliveCountMax=4)
rsh() { ssh "${SSHO[@]}" "$NODE" "$@"; }
say() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "FAILED: $*" >&2; exit 1; }

want() {   # want <phase> -> 0 if this phase should run
  local p="$1" i seen=0
  [ -n "$ONLY" ] && { [ "$ONLY" = "$p" ] && return 0 || return 1; }
  for i in "${PHASES[@]}"; do
    [ "$i" = "$FROM" ] && seen=1
    [ "$i" = "$p" ] && { [ $seen -eq 1 ] && return 0 || return 1; }
  done
  return 1
}

# The whole point of trap 1: map compute capability -> a torch that has kernels
# for it. Add rows as new silicon shows up; everything else is generic.
_torch_spec() {
  case "$1" in
    10.*|12.*) echo "torch==2.7.1 torchvision==0.22.1 https://download.pytorch.org/whl/cu128";;
    9.*|8.*)   echo "torch==2.6.0 torchvision==0.21.0 https://download.pytorch.org/whl/cu124";;
    *) die "no torch mapping for compute capability $1 -- add one to _torch_spec";;
  esac
}

# ---------------------------------------------------------------- probe -----
if want probe; then
  say "probe: $NODE"
  rsh true 2>/dev/null || die "cannot ssh to $NODE (is the key installed?)"
  CAP=$(rsh 'nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u' \
        | tr -d '\r' | head -1)
  NGPU=$(rsh 'nvidia-smi --query-gpu=index --format=csv,noheader | wc -l' | tr -d '\r')
  FREE=$(rsh 'df -BG --output=avail "$HOME" | tail -1' | tr -dc '0-9')
  say "  ${NGPU}x GPU, compute cap ${CAP}, ${FREE} GB free"
  [ "${FREE:-0}" -ge 120 ] || die "need >=120 GB free (weights ~15, dataset ~7, checkpoints ~29)"
  echo "$CAP" > /tmp/.starvla_cap; echo "$NGPU" > /tmp/.starvla_ngpu
fi
CAP=$(cat /tmp/.starvla_cap 2>/dev/null || rsh 'nvidia-smi --query-gpu=compute_cap --format=csv,noheader|sort -u|head -1' | tr -d '\r')
NGPU=$(cat /tmp/.starvla_ngpu 2>/dev/null || echo 8)

# ---------------------------------------------------------------- conda -----
if want conda; then
  if rsh 'test -x $HOME/miniconda3/bin/conda'; then
    say "conda: already installed ($(rsh '$HOME/miniconda3/bin/conda --version' | tr -d '\r'))"
  else
    say "conda: installing miniconda"
    rsh 'curl -fsSL -o /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         && bash /tmp/mc.sh -b -p $HOME/miniconda3 && rm -f /tmp/mc.sh' \
      || die "miniconda install"
    say "  $(rsh '$HOME/miniconda3/bin/conda --version' | tr -d '\r')"
  fi
fi

# ------------------------------------------------------------------ env -----
if want env; then
  read -r TORCH TVISION IDX <<<"$(_torch_spec "$CAP")"
  say "env: $ENV_NAME (python $PY_VER, $TORCH for cap $CAP)"
  # TRAP 6: conda >=25 refuses to touch repo.anaconda.com's pkgs/main and
  # pkgs/r until their Terms of Service are accepted, and those ToS carry
  # commercial-use conditions. --override-channels -c conda-forge sidesteps
  # the question entirely rather than accepting terms on someone's behalf,
  # and conda-forge is where these packages should come from anyway. (A
  # miniforge install instead of miniconda would default to this.)
  rsh "test -x \$HOME/miniconda3/envs/$ENV_NAME/bin/python" \
    || rsh "\$HOME/miniconda3/bin/conda create -y -n $ENV_NAME python=$PY_VER -q \
            --override-channels -c conda-forge" \
    || die "conda create"
  PIP="\$HOME/miniconda3/envs/$ENV_NAME/bin/pip"
  say "  torch (arch-matched; this is trap 1)"
  rsh "$PIP install -q --index-url $IDX $TORCH $TVISION" || die "torch install"
  # requirements.txt pins torchvision to the cu124 pairing; installing it
  # would drag torch back to 2.6 and silently undo the arch match.
  say "  requirements.txt (torch/torchvision filtered out)"
  rsh "cd $REPO_REMOTE 2>/dev/null && grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/req.txt" \
    2>/dev/null || true
  rsh "test -f /tmp/req.txt" || {
      # repo not synced yet: ship requirements ahead of the repo phase
      rsync -q -e "ssh ${SSHO[*]}" "$REPO_LOCAL/requirements.txt" "$NODE:/tmp/req_raw.txt"
      rsh "grep -vE '^(torch|torchvision)==' /tmp/req_raw.txt > /tmp/req.txt"; }
  # pipablepytorch3d builds from source and is not on the GR00T path; let it
  # fail alone rather than take the whole install with it.
  rsh "$PIP install -q -r /tmp/req.txt" || {
      say "  bulk install failed -- retrying line by line to isolate"
      rsh "while read -r p; do [ -z \"\$p\" ] && continue; case \$p in \\#*) continue;; esac
           $PIP install -q \"\$p\" 2>/dev/null || echo \"  SKIPPED \$p\"; done < /tmp/req.txt"; }
  # bitsandbytes: only needed for paged_adamw_8bit. The 8xB200 recipe uses
  # plain adamw, so this is opportunistic -- a failure here is not fatal.
  rsh "$PIP install -q bitsandbytes" >/dev/null 2>&1 \
    && say "  bitsandbytes ok (paged_adamw_8bit available)" \
    || say "  bitsandbytes unavailable -- stay on --trainer.optimizer.name adamw"
  ARCHES=$(rsh "\$HOME/miniconda3/envs/$ENV_NAME/bin/python -c \
    'import torch;print(\" \".join(torch.cuda.get_arch_list()))'" 2>&1 | tr -d '\r')
  say "  torch arch list: $ARCHES"
  case "$CAP" in 10.*) echo "$ARCHES" | grep -q sm_100 \
    || die "torch has no sm_100 kernels -- trap 1, fix _torch_spec";; esac
fi

# ----------------------------------------------------------------- repo -----
if want repo; then
  say "repo: syncing working tree (INCLUDING uncommitted changes)"
  # TRAP 7: the repo root is not just code. On the workstation it carries
  # outputs/ 158 GB, openpi_converted_protocol/ 16 GB and playground/ 8.3 GB
  # -- 24 GB even after excluding outputs, which at the measured 7.7 MB/s is
  # ~52 minutes of transferring things training never opens. The CODE is
  # ~10 MB (starVLA/ 3.2M + examples/ 3.4M + assets/ 3.2M). Exclude by name
  # and keep the post-sync assertions below as the safety net: if an exclude
  # ever goes too far, the VideoColorTemperature checks fail loudly instead
  # of the node quietly training without the augmentation.
  rsync -a --delete --info=stats1 \
    --exclude '.git' --exclude 'outputs/' --exclude '__pycache__/' \
    --exclude '*.pyc' --exclude '.venv/' --exclude 'wandb/' \
    --exclude 'openpi_converted_protocol/' --exclude 'playground/' \
    --exclude '*.whl' --exclude '*.pt' --exclude '*.safetensors' \
    --exclude 'checkpoints/' --exclude '.ruff_cache/' --exclude '.pytest_cache/' \
    -e "ssh ${SSHO[*]}" \
    "$REPO_LOCAL/" "$NODE:$(rsh "echo $REPO_REMOTE" | tr -d '\r')/" \
    | tail -3 || die "repo rsync"
  # The WB augmentation is the reason this syncs the WORKING TREE and not a
  # git checkout: VideoColorTemperature and the tipcrop DataConfig edits are
  # uncommitted, and a `git pull` node would train without them.
  rsh "grep -q 'class VideoColorTemperature' \
       $REPO_REMOTE/starVLA/dataloader/gr00t_lerobot/transform/video.py" \
    || die "VideoColorTemperature missing after sync -- the lighting augmentation did not land"
  rsh "grep -q 'VideoColorTemperature' \
       $REPO_REMOTE/examples/realRobots/UnitreeG1_Pipette/train_files/data_registry/data_config.py" \
    || die "data_config.py is not wiring VideoColorTemperature into the tipcrop transform"
  say "  lighting augmentation present on the node"
fi

# --------------------------------------------------------------- assets -----
if want assets; then
  rsh "mkdir -p $DATA_REMOTE"
  if [ "$PUSH_MODELS" = 1 ]; then
    say "assets: pushing HF cache + dataset from this workstation (no token leaves here)"
    for m in "${MODELS[@]}"; do
      d="models--${m//\//--}"
      rsync -a --partial --info=progress2 -e "ssh ${SSHO[*]}" \
        "$HOME/.cache/huggingface/hub/$d" "$NODE:\$HOME/.cache/huggingface/hub/" \
        2>&1 | tail -1
    done
    rsync -a --partial --info=progress2 -e "ssh ${SSHO[*]}" \
      "$HOME/Datasets/$DATASET_DIR" "$NODE:$DATA_REMOTE/" 2>&1 | tail -1
  else
    TOK=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)
    [ -n "$TOK" ] || die "no HF token at ~/.cache/huggingface/token (needed: the backbone is gated)"
    say "assets: node downloads its own (~22 MB/s measured); token passed transiently"
    # TRAP 8: do NOT `-U` huggingface_hub here. Unpinned it resolves to 1.x,
    # and transformers 4.57 hard-requires <1.0 -- the env installs cleanly and
    # then every `import transformers` dies on a version assert, which looks
    # like a broken checkout rather than a resolver upgrade. The `hf` CLI
    # entry point exists from 0.34, so the pin below still provides it.
    rsh "\$HOME/miniconda3/envs/$ENV_NAME/bin/pip install -q \
         'huggingface_hub[cli]>=0.34.0,<1.0'" >/dev/null 2>&1
    HF="\$HOME/miniconda3/envs/$ENV_NAME/bin/hf"
    for m in "${MODELS[@]}"; do
      say "  $m"
      # Token via env, never argv (/proc/PID/cmdline is world-readable), and
      # never `hf auth login` (that persists a token file on a shared box).
      ssh "${SSHO[@]}" -o SendEnv=none "$NODE" \
        "HF_TOKEN='$TOK' $HF download '$m' --quiet" >/dev/null \
        || die "download $m (gated? check the token has access)"
    done
    say "  $DATASET"
    ssh "${SSHO[@]}" "$NODE" \
      "HF_TOKEN='$TOK' $HF download '$DATASET' --repo-type dataset --quiet \
       --local-dir $DATA_REMOTE/$DATASET_DIR" >/dev/null \
      || die "download $DATASET"
  fi
  rsh "test -d $DATA_REMOTE/$DATASET_DIR/data" || die "dataset tree looks wrong"
  say "  dataset: $(rsh "ls $DATA_REMOTE/$DATASET_DIR/data/chunk-000/*.parquet | wc -l" | tr -d '\r') episodes"
fi

# ---------------------------------------------------------------- split -----
if want split; then
  say "split: building 49 train / 10 eval / 1 holdout (trap 5)"
  rsh "cd $REPO_REMOTE && \$HOME/miniconda3/envs/$ENV_NAME/bin/python \
       examples/realRobots/UnitreeG1_Pipette/train_files/build_splits.py \
       --src $DATA_REMOTE/$DATASET_DIR" 2>&1 | tail -8 || die "build_splits"
  # The two caches below are computed over whatever tree they were built in;
  # carried into a split they would describe 64 episodes inside 49.
  rsh "! test -e $DATA_REMOTE/$DATASET_DIR-train/meta/stats_gr00t.json" \
    || die "stats_gr00t.json leaked into the train split"
fi

# --------------------------------------------------------------- verify -----
if want verify; then
  say "verify: imports, GPU visibility, kernels"
  rsh "cd $REPO_REMOTE && PYTHONPATH=$REPO_REMOTE \
    \$HOME/miniconda3/envs/$ENV_NAME/bin/python - <<'PY'
import torch, sys
print('  torch', torch.__version__, 'cuda', torch.version.cuda)
print('  devices', torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, 'no GPU visible'
x = torch.randn(64, 64, device='cuda') @ torch.randn(64, 64, device='cuda')
torch.cuda.synchronize(); print('  matmul on', torch.cuda.get_device_name(0), 'OK')
# Check the REGISTRY, not a class name: the yaml names a registry id
# ('CosmosGR00TN1d7') and the class behind it is CosmosGR00T_N1d7. Importing
# the class by the yaml's spelling fails on a perfectly good checkout, and
# the registry lookup is what build_framework actually does.
from starVLA.model.framework.base_framework import (
    FRAMEWORK_REGISTRY, _auto_import_framework_modules)
_auto_import_framework_modules()
assert 'CosmosGR00TN1d7' in FRAMEWORK_REGISTRY._registry, \
    'CosmosGR00TN1d7 not registered; have %s' % sorted(FRAMEWORK_REGISTRY._registry)
print('  framework CosmosGR00TN1d7 ->', FRAMEWORK_REGISTRY['CosmosGR00TN1d7'].__name__)
sys.path.insert(0, 'examples/realRobots/UnitreeG1_Pipette/train_files')
from data_registry.data_config import ROBOT_TYPE_CONFIG_MAP as M
c = M['unitree_g1_pipette_tipcrop_waist_n1d7']
print('  action keys', c.action_keys)
assert 'action.waist' in c.action_keys, 'waist missing -- wrong DataConfig'
names = [type(t).__name__ for t in c.transform().transforms]
assert 'VideoColorTemperature' in names, 'lighting augmentation not in the transform'
print('  transform carries VideoColorTemperature')
PY" 2>&1 | tail -12 || die "verify"
fi

# ---------------------------------------------------------------- train -----
if want train && [ "$DO_TRAIN" = 1 ]; then
  # --gpus/--accum: share a node, or work around a broken NVSwitch fabric.
  # TRAP 11: >2-way NCCL needs nvidia-fabricmanager running. When it is dead
  # (`systemctl is-active nvidia-fabricmanager`), every rank count above 2
  # dies with "Cuda failure 802 'system not yet initialized'" NO MATTER how
  # many GPUs are idle -- it is a fabric fault, not an occupancy one, and
  # NCCL_P2P_DISABLE does not route around it. Restarting the service needs
  # root; --gpus is the no-root fallback. Keep global batch at 128 by raising
  # --accum as you drop GPUs (gpus x 16 x accum == 128).
  GPUS="${GPUS_OVERRIDE:-$(seq -s, 0 $((NGPU-1)))}"
  NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
  ACCUM="${ACCUM_OVERRIDE:-1}"
  say "train: launching ${NGPU}-GPU run $RUN_ID (gpus $GPUS, accum $ACCUM)"
  say "  global batch = $NGPU x $PER_DEVICE x $ACCUM = $((NGPU * 16 * ACCUM))"
  rsh "cat > \$HOME/launch_${RUN_ID}.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
source \\\$HOME/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME
cd $REPO_REMOTE
# TRAP 10: RUN_ID is a SHELL env var read by the launcher, and it is what
# names outputs/starvla/<RUN_ID>/ -- the yaml's own run_id only reaches
# --run_id and does NOT name the directory. Leave it unset and a run lands in
# the launcher's default 'starvla_gr00t_n1d7_pipette_g1', colliding with
# every other pipette run and breaking the bridge's checkpoint var, which
# resolves outputs/starvla/<RUN_ID><RUN_SUFFIX>/final_model/pytorch_model.pt.
export RUN_ID=${RUN_ID}${RUN_SUFFIX}
# TRAP 9: the repo is rsynced, not pip-installed, so \\\$PYTHONPATH is the only
# thing that makes \`import starVLA\` work. accelerate spawns each rank as a
# fresh process, so all 8 die with ModuleNotFoundError and the traceback you
# see is a torch.distributed rank-3 exit, not the real cause.
export PYTHONPATH=$REPO_REMOTE\${PYTHONPATH:+:\$PYTHONPATH}
export CONFIG_YAML=$CONFIG_YAML
export DATA_MIX=$DATA_MIX                 # trap 4: overrides the yaml
export CUDA_VISIBLE_DEVICES=$GPUS         # trap 3: never leave this default
export NUM_PROCESSES=$NGPU
export PER_DEVICE_BATCH_SIZE=$PER_DEVICE
export GRAD_ACCUM_STEPS=$ACCUM
export MAX_TRAIN_STEPS=10000
export SAVE_INTERVAL=5000
export DDP_FIND_UNUSED_PARAMETERS=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # trap 2
# TRAP 12: NVLink SHARP (NVLS) multicast. When nvidia-fabricmanager is
# started while GPUs hold a stale partition, the fabric comes up and the
# topology reads a full NV18 mesh, but every NCCL multicast team setup fails
# in /var/log/fabricmanager.log with: cannot find exporter GPU in partition
# ... All GPUs in the partition need to be reset to recover -- and NCCL HANGS
# instead of erroring, which reads as a dead job rather than a fabric fault.
# (Seen on 100.95.103.25, 2026-08-14.) Disabling NVLS skips
# multicast teams entirely and NCCL falls back to ring/tree over the same
# NVLinks: full bandwidth, minus the switch-side reduction offload, and no
# GPU reset or reboot needed. Remove this once the node has been reset and
# an 8-way allreduce passes without it.
export NCCL_NVLS_ENABLE=0
export WANDB_MODE=offline
bash examples/realRobots/UnitreeG1_Pipette/train_files/run_starvla_GR00T_N1d7_pipette_train.sh \\
  --trainer.optimizer.name adamw
EOF
chmod +x \$HOME/launch_${RUN_ID}.sh"
  rsh "cd \$HOME && nohup bash launch_${RUN_ID}.sh > \$HOME/train_${RUN_ID}.log 2>&1 &
       echo launched"
  say "  log: $NODE:~/train_${RUN_ID}.log"
  say "  relaunch by hand with: ~/launch_${RUN_ID}.sh"
fi

say "bootstrap complete for $NODE"
