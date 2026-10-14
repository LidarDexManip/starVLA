"""Bound SAM2 history for forward-only streaming, without changing its memory inputs.

The installed Transformers SAM2 uses all conditioning outputs, the preceding
num_maskmem-1 outputs and max_object_pointers_in_encoder-1 object pointers.
It otherwise retains every RGB frame/output indefinitely. Never use this helper
for reverse propagation, offline video iteration or historical prompt editing.
"""


def prune_streaming_history(session, frame_idx, *, num_maskmem, max_object_pointers):
    keep = max(1, int(num_maskmem), int(max_object_pointers))
    cutoff = frame_idx - keep + 1
    conditioning = set()
    for outputs in session.output_dict_per_obj.values():
        conditioning.update(outputs["cond_frame_outputs"])
        history = outputs["non_cond_frame_outputs"]
        for index in list(history):
            if index < cutoff:
                del history[index]
    # Old raw images are not used by forward memory attention. Retain prompt
    # images and the current image; the one-entry vision feature cache is bounded
    # by Transformers itself. The caller MUST supply a monotonic frame_idx:
    # SAM2's default len(processed_frames) would reuse indices after pruning.
    if session.processed_frames is not None:
        for index in list(session.processed_frames):
            if index != frame_idx and index not in conditioning:
                del session.processed_frames[index]
    for history in session.frames_tracked_per_obj.values():
        for index in list(history):
            if index < cutoff and index not in conditioning:
                del history[index]
    return dict(frame_idx=frame_idx, history_limit=keep,
                raw_frames=len(session.processed_frames or {}),
                conditioning_outputs=sum(len(o["cond_frame_outputs"]) for o in session.output_dict_per_obj.values()),
                recent_outputs=sum(len(o["non_cond_frame_outputs"]) for o in session.output_dict_per_obj.values()))
