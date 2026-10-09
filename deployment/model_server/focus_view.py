"""Serve-time FOCUS view for the five-step pipette checkpoints trained on ego + focus (v6, 2026-10-09).

The training focus view (dataset v1.2 + focus, ``observation.images.focus``) is a 300 x 300 window of
the 1280 x 720 ego frame around the phase's target, resized to 224 x 224:

    phase 1 / 5  pipette holder        OWLv2 image-guided (two exemplar crops)
    phase 2 / 4  blue tube rack        HSV blue blob + the OWLv2 text box that agrees with it
    phase 3      tube opening          OWLv2 text "the opening of a plastic tube"

tracked with SAM 2.1; the window FOLLOWS the target while it is clearly visible and HOLDS the last
clear centre while a hand covers it; EMA-smoothed; clamped inside the frame. The offline builder
(tools/build_focus_v13.py in the hub overlay jren313/g1-pipette-2view-teleop0925-5task-eerel-focus)
could look at the whole episode. This server-side twin is CAUSAL -- same detectors, thresholds and
geometry, with:

  offline                                    here
  prompt = the episode's clearest frame      the FIRST frame the detector accepts after a reset
  SAM 2.1 forward + reverse                  SAM 2.1 streaming, forward only
  visibility = area / episode p95 area       area / p95 of the areas seen since the lock
  before the first follow: back-filled       the acquire box's centre; before the lock, the
                                             target's typical training position (PRIOR_CENTRE)
  EMA a per 20 Hz sample (0.3; tube 0.5)     a_eff = 1 - (1 - a)^(20 dt): the same time constant

Replayed on 15 held-out episodes the live window sits within a few px of the training window (see
the v6 notes); its "following" flag agrees less (the visibility reference differs).

``FocusViewSynth`` is what the server calls: (ego frame, phase sentence) -> (224 crop, record). The
sentence picks the target; a new sentence (a phase switch) or ``reset()`` re-acquires.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

SIDE, OUT = 300, 224
KEEP = {"holder": 0.9, "rack": 0.9, "tube": 0.3}
EMA20 = {"holder": 0.3, "rack": 0.3, "tube": 0.5}
#: phase sentence prefix -> target (the five-step plan's sentences; build_focus_v13.py's PH table)
TARGET_BY_PREFIX = (("Pick up the pipette", "holder"), ("Pick up the tube", "rack"), ("Aim", "tube"),
                    ("Put the tube", "rack"), ("Put the pipette", "holder"))
#: Before the first lock: the target's typical frame-0 window centre over the TRAINING episodes
#: (median of observation.focus.center at frame 0, which the builder back-filled with the target's
#: real position; 379 train episodes, both capture venues pooled; p10-p90 spread about +-60 px).
#: The frame centre instead was 290 px off on held-out holder episode 11.
PRIOR_CENTRE = {"holder": (878.0, 602.0), "rack": (384.0, 578.0), "tube": (438.0, 460.0)}
SAM_IDS = {"tiny": "facebook/sam2.1-hiera-tiny", "small": "facebook/sam2.1-hiera-small",
           "base_plus": "facebook/sam2.1-hiera-base-plus", "large": "facebook/sam2.1-hiera-large"}
EXEMPLAR_DIR = Path(__file__).parent / "focus_exemplars"


def target_for(sentence: str) -> Optional[str]:
    s = str(sentence).strip()
    return next((t for p, t in TARGET_BY_PREFIX if s.startswith(p)), None)


def hsv_blue(rgb, area=False):
    m = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), (100, 150, 90), (125, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return 0 if area else None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = st[k]
    return int(a) if area else [float(x), float(y), float(x + w), float(y + h)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0.0


class FocusTracker:
    """Causal target tracker + window logic (see the module docstring)."""

    def __init__(self, sam="small", device="cuda", exemplar_dir=None,
                 owl_id="google/owlv2-base-patch16-ensemble"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2VideoModel, Sam2VideoProcessor
        sam_id = SAM_IDS.get(sam, sam)
        self.dev = device
        self.owl_p = Owlv2Processor.from_pretrained(owl_id)
        self.owl = Owlv2ForObjectDetection.from_pretrained(owl_id).to(device).eval()
        self.sam_p = Sam2VideoProcessor.from_pretrained(sam_id)
        self.sam = Sam2VideoModel.from_pretrained(sam_id).to(device, dtype=torch.bfloat16).eval()
        ed = Path(exemplar_dir or EXEMPLAR_DIR)
        self.exemplars = [cv2.cvtColor(cv2.imread(str(ed / f"holder_exemplar_{e}.png")), cv2.COLOR_BGR2RGB)
                          for e in (62, 262)]
        self.target = None
        logger.info("FocusTracker ready: SAM %s + OWLv2 %s on %s", sam_id, owl_id, device)

    # -- detection: build_focus_v13.py's detectors on ONE frame -----------------------------
    @torch.no_grad()
    def _owl_text(self, rgb, text):
        out = self.owl(**self.owl_p(text=[[text]], images=rgb, return_tensors="pt").to(self.dev))
        side = max(rgb.shape[:2])
        r = self.owl_p.post_process_object_detection(out, threshold=0.05, target_sizes=torch.tensor([[side, side]]))[0]
        return [([float(v) for v in b.tolist()], float(s)) for b, s in zip(r["boxes"], r["scores"])]

    @torch.no_grad()
    def _owl_image(self, rgb, ex):
        out = self.owl.image_guided_detection(**self.owl_p(images=rgb, query_images=ex, return_tensors="pt").to(self.dev))
        side = max(rgb.shape[:2])
        r = self.owl_p.post_process_image_guided_detection(out, threshold=0.5, nms_threshold=0.3,
                                                           target_sizes=torch.tensor([[side, side]]))[0]
        return None if len(r["scores"]) == 0 else r["boxes"][int(r["scores"].argmax())].tolist()

    def detect(self, rgb):
        """The target's box on this frame, or None (the next frame tries again)."""
        H, W = rgb.shape[:2]; A = H * W
        clip = lambda b: [max(0.0, b[0]), max(0.0, b[1]), min(W - 1.0, b[2]), min(H - 1.0, b[3])]
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        if self.target == "rack":
            hb = hsv_blue(rgb)
            if hb is None or area(hb) < 0.005 * A:
                return None
            best = max(self._owl_text(rgb, "a blue test tube rack"), key=lambda c: iou(c[0], hb), default=None)
            return clip(best[0] if best is not None and iou(best[0], hb) > 0.3 else hb)
        if self.target == "holder":
            dets = [b for b in (self._owl_image(rgb, e) for e in self.exemplars) if b and 0.002 * A < area(b) < 0.04 * A]
            return clip(np.median(np.array(dets), axis=0).tolist()) if dets else None
        c = [(b, s) for b, s in self._owl_text(rgb, "the opening of a plastic tube") if 0.001 * A < area(b) < 0.02 * A]
        return clip(max(c, key=lambda x: x[1])[0]) if c else None

    # -- tracking ---------------------------------------------------------------------------
    def reset(self, target):
        assert target in KEEP, target
        self.target = target
        self.sess, self.locked, self.last, self.c, self.t_prev = None, False, None, None, None
        self.areas = []

    @torch.no_grad()
    def _sam(self, rgb, box=None):
        inputs = self.sam_p(images=rgb, device=self.dev, return_tensors="pt")
        if box is not None:
            self.sess = self.sam_p.init_video_session(inference_device=self.dev, dtype=torch.bfloat16)
            self.sam_p.add_inputs_to_inference_session(inference_session=self.sess, frame_idx=0, obj_ids=1,
                                                       input_boxes=[[box]], original_size=inputs.original_sizes[0])
        out = self.sam(inference_session=self.sess, frame=inputs.pixel_values[0].to(torch.bfloat16))
        m = self.sam_p.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes,
                                          binarize=True)[0][0, 0].cpu().numpy()
        ys, xs = np.nonzero(m)
        return None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(len(xs))]

    def _centre(self, b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2 if self.target != "tube" else b[1] + 0.25 * (b[3] - b[1]))

    def step(self, rgb, t=None):
        """One frame (RGB, ego size). ``t`` (s) times the EMA; default: the wall clock."""
        t0 = time.perf_counter()
        t = time.monotonic() if t is None else float(t)
        H, W = rgb.shape[:2]
        acquired, b = False, None
        if not self.locked:
            box = self.detect(rgb)
            if box is not None:
                b = self._sam(rgb, box=box)
                self.locked, acquired, self.last = True, True, self._centre(box)
        else:
            b = self._sam(rgb)
        vis, fol = 0.0, 0
        if b is not None:
            # causal twin of the builder's episode-p95 reference: p95 of the areas seen so far
            # (the lock-frame area alone ran 3 % high on episode 11 and held where training followed)
            self.areas.append(float(b[4]))
            vis = b[4] / max(1.0, float(np.percentile(self.areas, 95)))
            if vis >= KEEP[self.target]:
                self.last, fol = self._centre(b), 1
        raw = np.asarray(self.last if self.last is not None else PRIOR_CENTRE[self.target], float)
        if self.c is None:
            self.c = raw
        else:
            dt = max(1e-3, t - self.t_prev)
            a = 1.0 - (1.0 - EMA20[self.target]) ** (20.0 * dt)
            self.c = a * raw + (1 - a) * self.c
        self.t_prev = t
        x0 = float(np.clip(np.round(self.c[0] - SIDE / 2), 0, W - SIDE))
        y0 = float(np.clip(np.round(self.c[1] - SIDE / 2), 0, H - SIDE))
        return {"target": self.target, "window": [x0, y0, x0 + SIDE, y0 + SIDE], "center": self.c.tolist(),
                "following": fol, "visibility": float(vis), "locked": self.locked, "acquired": acquired,
                "box": b, "ms": 1000 * (time.perf_counter() - t0)}

    @staticmethod
    def crop(rgb, rec):
        x0, y0 = int(rec["window"][0]), int(rec["window"][1])
        return cv2.resize(rgb[y0:y0 + SIDE, x0:x0 + SIDE], (OUT, OUT), interpolation=cv2.INTER_AREA)


class BinaryEgoCamera:
    """The robot's ego camera wire (TWIST2 / Orin ``binary``: int32 x4 header (w, h, jpeg_len,
    depth_len) + JPEG), a ZMQ SUB with CONFLATE -- a second subscriber beside the bridge, so the
    tracker sees every frame it has time for, not only the ones a policy request carries."""

    def __init__(self, address: str):
        import zmq
        host, port = address.replace("tcp://", "").rsplit(":", 1)
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.setsockopt(zmq.CONFLATE, True)
        self.sock.setsockopt(zmq.RCVHWM, 3)
        self.sock.connect(f"tcp://{host}:{int(port)}")

    def recv(self, timeout_ms: int = 200):
        import struct
        if not self.sock.poll(timeout=timeout_ms):
            return None
        raw = self.sock.recv()
        if len(raw) < 16:
            return None
        _w, _h, jlen, _d = struct.unpack("iiii", raw[:16])
        bgr = cv2.imdecode(np.frombuffer(raw[16:16 + jlen], np.uint8), cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FocusTap:
    """Operator feed of what the focus view did on each request: the served 224 crop (``focus``) and
    the ego frame with its window drawn (``focus_window``), as keyed JPEGs on a ZMQ PUB -- the bridge
    frame tap's multicam wire (``VIEWJPG1:<name>:<jpeg>``, one part each), so the bridge console
    shows them as two more policy-eye tiles. A PUB without subscribers drops at no cost. Drawing and
    encoding run on this class's own thread (newest offer wins): a request only hands over arrays."""

    WIDTH = 640                     # focus_window tile width (the 1280 x 720 ego frame at 1/2)
    COLOURS = {"follow": (40, 220, 90), "hold": (255, 170, 0), "prior": (170, 170, 170)}

    def __init__(self, port: int):
        import threading
        import zmq
        self._zmq = zmq
        self.sock = zmq.Context.instance().socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 4)
        self.sock.bind(f"tcp://*:{int(port)}")
        self._cv = threading.Condition()
        self._job = None
        threading.Thread(target=self._loop, name="focus-tap", daemon=True).start()
        logger.info("focus view: tap PUB tcp://*:%d (tiles focus + focus_window)", int(port))

    def offer(self, ego_rgb, crop, rec):
        with self._cv:
            self._job = (ego_rgb, crop, rec)
            self._cv.notify()

    @classmethod
    def window_view(cls, ego_rgb, rec):
        """The ego frame at WIDTH with the served window (green following / amber holding the last
        clear centre / grey the training prior, nothing locked yet) and the tracker's mask box."""
        H, W = ego_rgb.shape[:2]
        s = cls.WIDTH / W
        img = cv2.resize(ego_rgb, (cls.WIDTH, round(H * s)), interpolation=cv2.INTER_AREA)
        state = "follow" if rec["following"] else ("hold" if rec["locked"] else "prior")
        if rec.get("box") is not None:
            b = [round(v * s) for v in rec["box"][:4]]
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 200, 255), 1)
        w = [round(v * s) for v in rec["window"]]
        cv2.rectangle(img, (w[0], w[1]), (w[2] - 1, w[3] - 1), cls.COLOURS[state], 2)
        label = f"{rec['target']}  {state}  vis {rec['visibility']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
        cv2.putText(img, label, (5, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls.COLOURS[state], 1,
                    cv2.LINE_AA)
        return img

    def _loop(self):
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                (ego, crop, rec), self._job = self._job, None
            try:
                for name, img in (("focus", crop), ("focus_window", self.window_view(ego, rec))):
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ok:
                        self.sock.send(b"VIEWJPG1:" + name.encode() + b":" + jpg.tobytes(),
                                       self._zmq.NOBLOCK)
            except Exception as e:                     # the feed must never touch the request path
                logger.debug("focus tap: %s: %s", type(e).__name__, e)


class FocusViewSynth:
    """(ego frame, phase sentence) -> (224 x 224 focus crop, window record), for the policy server.

    The sentence picks the target; a different sentence (the bridge switched phase) or ``reset()``
    (GO / RESET / a new session) re-acquires it. The last record is kept for the response ``info``
    and the log.

    Two modes:
      * no ``camera``: the tracker steps on the frames the requests carry (the request rate, ~2.6 Hz
        on the robot) -- enough for the static holder / rack, short for the moving tube (one of
        three held-out aim episodes lost it at 5 Hz).
      * ``camera`` = the ego camera's ZMQ address: a background thread steps the tracker on that
        stream at up to ``rate_hz`` and the request only CROPS its own frame with the newest window.
        The thread YIELDS to inference -- the server holds ``gpu_lock`` around every model call, so
        the tracker never shares the GPU with PI0.5 (sharing it measured +31 ms per PI0.5 call).
    """

    def __init__(self, sam="small", device="cuda", exemplar_dir=None, camera: Optional[str] = None,
                 rate_hz: float = 10.0, detect_period_s: float = 0.5, tap_port: int = 0):
        import threading
        self.tracker = FocusTracker(sam=sam, device=device, exemplar_dir=exemplar_dir)
        self._tap = FocusTap(tap_port) if tap_port else None
        self._sentence, self._gen = None, 0
        self.last = None          # newest tracker record for the current sentence
        self.served = None        # the record the last REQUEST cropped with (response info)
        self.gpu_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cam = BinaryEgoCamera(camera) if camera else None
        self._period, self._detect_period = 1.0 / float(rate_hz), float(detect_period_s)
        self._stop = False
        self.n_bg = 0
        if self._cam is not None:
            self._thread = threading.Thread(target=self._loop, name="focus-tracker", daemon=True)
            self._thread.start()
            logger.info("focus view: background tracker on %s at <= %.0f Hz (yields to inference)",
                        camera, rate_hz)

    def reset(self):
        with self._state_lock:
            self._sentence, self.last = None, None
            self._gen += 1

    def _log_rec(self, rec):
        if rec["acquired"]:
            logger.info("focus view: %s acquired at window %s (%.0f ms)", rec["target"],
                        rec["window"][:2], rec["ms"])

    def _loop(self):
        """Camera mode: the ONLY thread that touches the tracker. Tracking steps hold gpu_lock
        (inference waits at most one ~30 ms step); detection while unlocked does NOT (a holder
        detection is 0.6-2 s and must not stall a request) and is paced to detect_period_s."""
        applied, last_detect = None, 0.0
        while not self._stop:
            t0 = time.monotonic()
            try:
                frame = self._cam.recv(timeout_ms=200)
                with self._state_lock:
                    sentence, gen = self._sentence, self._gen
                if frame is not None and sentence is not None:
                    if gen != applied:
                        self.tracker.reset(target_for(sentence))
                        applied, last_detect = gen, 0.0
                    rec = None
                    if self.tracker.locked:
                        with self.gpu_lock:
                            rec = self.tracker.step(frame)
                    elif t0 - last_detect >= self._detect_period:
                        last_detect = t0
                        rec = self.tracker.step(frame)
                    if rec is not None:
                        self._log_rec(rec)
                        self.n_bg += 1
                        with self._state_lock:
                            if gen == self._gen:
                                self.last = rec
            except Exception as e:                     # a tracker hiccup must not end the thread
                logger.warning("focus view: background step failed: %s: %s", type(e).__name__, e)
            time.sleep(max(0.0, self._period - (time.monotonic() - t0)))

    def _prior_rec(self, shape, target):
        """The window before anything is tracked for this phase: the target's training prior."""
        H, W = shape[:2]
        cx, cy = PRIOR_CENTRE[target]
        x0 = float(np.clip(round(cx - SIDE / 2), 0, W - SIDE)); y0 = float(np.clip(round(cy - SIDE / 2), 0, H - SIDE))
        return {"target": target, "window": [x0, y0, x0 + SIDE, y0 + SIDE], "center": [cx, cy],
                "following": 0, "visibility": 0.0, "locked": False, "acquired": False, "box": None, "ms": 0.0}

    def __call__(self, ego_rgb: np.ndarray, sentence: str):
        """Called by the request path (the caller does NOT hold gpu_lock yet)."""
        target = target_for(sentence)
        if target is None:
            raise ValueError(f"focus view: no target for the instruction {sentence!r} "
                             f"(known phase prefixes: {[p for p, _ in TARGET_BY_PREFIX]})")
        if self._cam is None:
            # no stream: track on the request's own frame (detection included)
            with self._state_lock, self.gpu_lock:
                if sentence != self._sentence:
                    self.tracker.reset(target)
                    self._sentence, self.last = sentence, None
                    logger.info("focus view: target %s for %r", target, sentence[:60])
                rec = self.tracker.step(np.ascontiguousarray(ego_rgb))
                self._log_rec(rec)
                self.last = rec
        else:
            # the background thread tracks; the request crops with its newest window or the prior
            with self._state_lock:
                if sentence != self._sentence:
                    self._sentence, self.last = sentence, None
                    self._gen += 1
                    logger.info("focus view: target %s for %r", target, sentence[:60])
                rec = self.last if self.last is not None else self._prior_rec(ego_rgb.shape, target)
        self.served = rec
        crop = self.tracker.crop(ego_rgb, rec)
        if self._tap is not None:
            self._tap.offer(ego_rgb, crop, rec)
        return crop, rec
