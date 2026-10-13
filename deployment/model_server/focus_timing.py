"""Thread-safe host-wall-time overlap diagnostics, not CUDA kernel profiling.

Pure telemetry: does not acquire the inference GPU lock or change detections.
"""
from collections import deque
import threading
import time


class FocusTiming:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.lock = threading.Lock()
        self.spans = deque(maxlen=4096)
        self.active = {}
        self.sequence = 0
        self.counts = {"detection": 0, "tracking": 0}

    def begin(self, kind, target, generation):
        with self.lock:
            self.sequence += 1
            self.active[self.sequence] = dict(kind=kind, target=target, generation=generation,
                                              start=self.clock())
            return self.sequence

    def end(self, token):
        with self.lock:
            span = self.active.pop(token)
            span["end"] = self.clock()
            self.counts[span["kind"]] += 1
            self.spans.append(span)

    def snapshot(self, start=None, end=None):
        with self.lock:
            now = self.clock() if end is None else end
            spans = list(self.spans) + [dict(s, end=now) for s in self.active.values()]
            result = dict(counts=dict(self.counts), active=[dict(s, elapsed_ms=(now-s["start"])*1000)
                                                         for s in self.active.values()])
            for kind in self.counts:
                rows = [s for s in spans if s["kind"] == kind]
                result[kind+"_overlap_ms"] = (sum(max(0., min(now,s["end"])-max(start,s["start"]))
                                                  for s in rows)*1000 if start is not None else 0.)
                finished = [s for s in self.spans if s["kind"] == kind]
                result[kind+"_last_ms"] = ((finished[-1]["end"]-finished[-1]["start"])*1000
                                           if finished else None)
            return result
