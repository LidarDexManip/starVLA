# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""PolicyNormProcessor — reuse the training-time ComposedModalityTransform.

This class replaces the hand-rolled un-normalization math that previously
lived in every eval client. It rebuilds the *exact* ``ComposedModalityTransform``
used at training time from a checkpoint:

  1. Read ``config.yaml`` next to the ckpt → resolve ``data_mix`` →
     look up ``robot_type`` from ``DATASET_NAMED_MIXTURES`` →
     fetch the ``DataConfig`` from ``ROBOT_TYPE_CONFIG_MAP``.
  2. Build the transform pipeline via ``data_config.transform()``.
  3. Reconstruct a ``DatasetMetadata`` from the saved
     ``dataset_statistics.json`` (which stores **combined** per-modality
     arrays of length ``D``) by splitting it into per-key entries that match
     ``data_config.action_keys`` / ``state_keys``.
  4. ``set_metadata(...)`` binds the metadata into every transform.

After construction the caller simply invokes :meth:`unapply_actions` /
:meth:`apply_state` to get/normalize tensors using the same code path as
training — there is no second source of truth for normalization math.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    StateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.model.framework.share_tools import read_mode_config

logger = logging.getLogger(__name__)


def _resolve_robot_type(
    model_cfg: dict,
    unnorm_key: Optional[str] = None,
) -> str:
    """Look up the training robot_type from the saved cfg.

    Convention used by starVLA train scripts:
      ``cfg.datasets.vla_data.data_mix`` is a key in
      ``DATASET_NAMED_MIXTURES`` whose value is a list of
      ``(dataset_name, weight, robot_type)`` tuples.

    When a data_mix contains entries from multiple robot types (e.g.
    ``bridge_rt_1`` covers ``oxe_bridge`` + ``oxe_rt1``), ``unnorm_key``
    is used to identify which embodiment is requested.  In those mixtures
    the ``robot_type`` field of each entry **matches** the top-level key in
    ``dataset_statistics.json``, so ``unnorm_key`` serves as the selector.
    """
    try:
        data_mix = model_cfg["datasets"]["vla_data"]["data_mix"]
    except (KeyError, TypeError) as e:
        raise KeyError(
            "ckpt config.yaml is missing `datasets.vla_data.data_mix`; "
            "cannot resolve training-time robot_type."
        ) from e

    if data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(
            f"data_mix={data_mix!r} not in DATASET_NAMED_MIXTURES "
            f"(available: {sorted(DATASET_NAMED_MIXTURES.keys())[:20]} ...). "
            "Did you forget to register the example under examples/<bench>/train_files/data_registry/?"
        )

    mixture = DATASET_NAMED_MIXTURES[data_mix]
    robot_types = sorted({entry[2] for entry in mixture})

    if len(robot_types) == 1:
        return robot_types[0]

    # Multiple robot types in the mixture.
    # Use unnorm_key as a direct selector: for multi-robot mixtures the
    # dataset_statistics.json top-level keys equal the robot_type values.
    if unnorm_key is not None and unnorm_key in robot_types:
        return unnorm_key

    raise ValueError(
        f"data_mix={data_mix!r} contains multiple robot_types {robot_types}. "
        "Pass `unnorm_key` matching one of them to disambiguate "
        f"(e.g. unnorm_key={robot_types[0]!r})."
    )


def _infer_key_dims(
    data_config: Any,
    combined_stats: Dict[str, Any],
    modality_keys: Sequence[str],
    modality: str,
) -> Dict[str, int]:
    """Compute per-key dimensions for splitting combined stats arrays.

    Lookup priority:
      1. ``data_config.<modality>_key_dims`` — explicit dict classvar on the DataConfig
         (required for DataConfigs with non-uniform or multi-dim keys).
      2. Infer uniformly from stats array length: if ``D_total / n_keys`` is an
         integer, use that as the uniform per-key dim.
      3. Fall back to dim=1 when no stats are available (empty combined dict).

    Args:
        data_config: The DataConfig instance for the current robot type.
        combined_stats: The raw stats dict for the chosen unnorm_key (the value
            of ``norm_stats[unnorm_key]`` from ``dataset_statistics.json``).
        modality_keys: Ordered list of full keys for this modality
            (e.g. ``["action.left_joints", ...]``).
        modality: ``"action"`` or ``"state"``.

    Returns:
        Dict mapping each full key to its integer dimension.
    """
    attr = f"{modality}_key_dims"
    if hasattr(data_config, attr):
        return dict(getattr(data_config, attr))

    combined = combined_stats.get(modality, {})
    stat_arr = next((v for k, v in combined.items() if k != "mask"), None)
    n_keys = len(modality_keys)
    if stat_arr is not None and n_keys > 0:
        D_total = len(stat_arr)
        if D_total == n_keys:
            return {k: 1 for k in modality_keys}
        elif n_keys > 0 and D_total % n_keys == 0:
            dim = D_total // n_keys
            return {k: dim for k in modality_keys}
        else:
            raise ValueError(
                f"Cannot infer per-key dims for modality={modality!r}: "
                f"D_total={D_total} is not evenly divisible by n_keys={n_keys}. "
                f"Add `{attr} = {{key: dim, ...}}` to the DataConfig "
                f"(keys={list(modality_keys)})."
            )
    return {k: 1 for k in modality_keys}


def _build_dataset_metadata(
    stats_for_key: Dict[str, Any],
    embodiment_tag: Any,
    action_keys: Sequence[str],
    state_keys: Sequence[str],
    action_key_dims: Optional[Dict[str, int]] = None,
    state_key_dims: Optional[Dict[str, int]] = None,
    video_keys: Sequence[str] = (),
    video_resolution: Sequence[int] = (224, 224),
    video_fps: float = 30.0,
    video_channels: int = 3,
) -> DatasetMetadata:
    """Convert the *combined* stats arrays from ``dataset_statistics.json``
    back into a per-subkey :class:`DatasetMetadata` matching what the training
    pipeline produced.

    The saved ``dataset_statistics.json`` stores stats as flat arrays of
    length ``D = sum(per_key_dims)``.  For each ``"action.<sub>"`` key we
    slice out ``dim_k`` elements starting at the current cursor and store them
    under ``statistics.action.<sub> = {"min": [v0..v_{k-1}], ...}``.

    Args:
        stats_for_key: ``norm_stats[unnorm_key]`` from ``dataset_statistics.json``.
        embodiment_tag: Embodiment tag from the DataConfig.
        action_keys: Ordered list of full action keys.
        state_keys: Ordered list of full state keys.
        action_key_dims: Per-key dimension dict (``{full_key: dim_k}``).
            Defaults to dim=1 for every key.
        state_key_dims: Per-key dimension dict for state keys.
            Defaults to dim=1 for every key.
    """
    if action_key_dims is None:
        action_key_dims = {k: 1 for k in action_keys}
    if state_key_dims is None:
        state_key_dims = {k: 1 for k in state_keys}

    def _split_combined(
        combined: Dict[str, Sequence[float]],
        keys: Sequence[str],
        key_dims: Dict[str, int],
    ):
        """Split combined arrays into per-subkey dicts using per-key dims.

        ``combined`` looks like ``{"min": [..D..], "max": [..D..], "mask": [..D..], ...}``.
        ``keys`` is the ordered list of full keys.
        ``key_dims`` maps each full key to its integer dimension.
        Returns ``(stats_per_subkey, meta_per_subkey)``.
        """
        stats_per_subkey: Dict[str, Dict[str, List[float]]] = {}
        meta_per_subkey: Dict[str, StateActionMetadata] = {}
        cursor = 0
        for full_key in keys:
            subkey = full_key.split(".", 1)[1]
            dim_k = key_dims.get(full_key, 1)
            per_key: Dict[str, List[float]] = {}
            for stat_name, arr in combined.items():
                if stat_name == "mask":
                    continue
                end = cursor + dim_k
                if end > len(arr):
                    # Saved combined array shorter than expected (truncated
                    # pad channels etc.). Skip this stat field silently.
                    continue
                per_key[stat_name] = [float(v) for v in arr[cursor:end]]
            stats_per_subkey[subkey] = per_key
            meta_per_subkey[subkey] = StateActionMetadata(
                absolute=True,
                rotation_type=None,
                shape=(dim_k,),
                continuous=True,
            )
            cursor += dim_k
        return stats_per_subkey, meta_per_subkey

    action_combined = stats_for_key.get("action", {})
    state_combined = stats_for_key.get("state", {})

    action_stats, action_meta = _split_combined(action_combined, action_keys, action_key_dims)

    # Populate video modality metadata for any video sub-keys referenced by the
    # training-time transform pipeline. dataset_statistics.json carries no video
    # info, but data_configs whose transform() includes video augmentation (e.g.
    # VideoColorJitter in the QwenOFT piston config) call set_metadata on those
    # transforms and require the sub-key + resolution to exist. Un-normalization
    # never touches video (VideoTransforms are not InvertibleModalityTransform,
    # so unapply() skips them), so these values only need to be schema-valid.
    video_meta: Dict[str, Any] = {}
    for full_key in video_keys:
        subkey = full_key.split(".", 1)[1] if "." in full_key else full_key
        video_meta[subkey] = {
            "resolution": (int(video_resolution[0]), int(video_resolution[1])),
            "channels": int(video_channels),
            "fps": float(video_fps),
        }

    statistics: Dict[str, Any] = {"action": action_stats}
    modalities: Dict[str, Any] = {"video": video_meta, "action": action_meta}
    if state_combined:
        state_stats, state_meta = _split_combined(state_combined, state_keys, state_key_dims)
        statistics["state"] = state_stats
        modalities["state"] = state_meta

    return DatasetMetadata.model_validate(
        {
            "statistics": statistics,
            "modalities": modalities,
            "embodiment_tag": embodiment_tag.value if hasattr(embodiment_tag, "value") else embodiment_tag,
        }
    )


class PolicyNormProcessor:
    """Server-side normalization helper backed by training-time transforms.

    Construct once per checkpoint; call :meth:`unapply_actions` to convert
    a normalized action chunk ``(T, D)`` back to env-space.

    Args:
        ckpt_path: Path to the ``*.pt`` checkpoint (the loader looks for
            ``config.yaml`` and ``dataset_statistics.json`` two dirs up).
        unnorm_key: Which top-level key in ``dataset_statistics.json`` to use.
            ``None`` → auto-pick the only key.
    """

    def __init__(self, ckpt_path: str, unnorm_key: Optional[str] = None) -> None:
        self._ckpt_path = str(ckpt_path)
        cfg, norm_stats = read_mode_config(self._ckpt_path)
        self._model_cfg = cfg
        self._norm_stats = norm_stats

        # 3-early) Pick the requested unnorm_key (or auto-select) BEFORE
        # resolving robot_type so we can use it as a hint for multi-robot mixtures.
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            # else: defer error to step 3 below after robot_type resolution attempt
        elif unnorm_key not in norm_stats:
            raise KeyError(
                f"unnorm_key={unnorm_key!r} not in {list(norm_stats.keys())}"
            )
        self._unnorm_key = unnorm_key  # may still be None for multi-key case

        # 1) Resolve which DataConfig was used at training.
        robot_type = _resolve_robot_type(cfg, unnorm_key=unnorm_key)
        if robot_type not in ROBOT_TYPE_CONFIG_MAP:
            raise KeyError(
                f"robot_type={robot_type!r} not in ROBOT_TYPE_CONFIG_MAP "
                f"(available: {sorted(ROBOT_TYPE_CONFIG_MAP.keys())}). "
                "Make sure the example's data_registry/data_config.py is importable."
            )
        self._data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        self._action_keys: List[str] = list(self._data_config.action_keys)
        self._state_keys: List[str] = list(getattr(self._data_config, "state_keys", []))
        self._video_keys: List[str] = list(getattr(self._data_config, "video_keys", []))

        # Training image size -> a schema-valid video resolution for transforms
        # whose set_metadata reads it (un-norm itself never uses video). Falls
        # back to a square default when the cfg does not specify a size.
        _vla_cfg = (cfg.get("datasets", {}) or {}).get("vla_data", {}) or {}
        _img_size = _vla_cfg.get("obs_image_size") or _vla_cfg.get("image_size")
        if isinstance(_img_size, (list, tuple)) and len(_img_size) >= 2:
            self._video_resolution = (int(_img_size[-2]), int(_img_size[-1]))
        else:
            self._video_resolution = (224, 224)

        # 2) Build training-time transform pipeline.
        transform = self._data_config.transform()
        if not isinstance(transform, ComposedModalityTransform):
            transform = ComposedModalityTransform(transforms=[transform])
        self._transform = transform

        # 3) Pick the requested unnorm_key (finalize; error if still None here).
        if self._unnorm_key is None:
            raise ValueError(
                f"Multiple unnorm_keys in dataset_statistics.json: "
                f"{list(norm_stats.keys())}. Pass unnorm_key explicitly."
            )
        unnorm_key = self._unnorm_key

        # 4) Resolve per-key dims (handles multi-d action/state keys).
        stats_for_unnorm = norm_stats[unnorm_key]
        self._action_key_dims: Dict[str, int] = _infer_key_dims(
            self._data_config, stats_for_unnorm, self._action_keys, "action"
        )
        self._state_key_dims: Dict[str, int] = _infer_key_dims(
            self._data_config, stats_for_unnorm, self._state_keys, "state"
        )

        # 5) Build & bind metadata.
        ds_meta = _build_dataset_metadata(
            stats_for_key=stats_for_unnorm,
            embodiment_tag=self._data_config.embodiment_tag,
            action_keys=self._action_keys,
            state_keys=self._state_keys,
            action_key_dims=self._action_key_dims,
            state_key_dims=self._state_key_dims,
            video_keys=self._video_keys,
            video_resolution=self._video_resolution,
        )
        self._transform.set_metadata(ds_meta)
        self._transform.eval()  # mark transforms as eval-mode

        logger.info(
            "PolicyNormProcessor ready: robot_type=%s, unnorm_key=%s, "
            "action_keys=%s (dims=%s), state_keys=%s",
            robot_type,
            unnorm_key,
            self._action_keys,
            [self._action_key_dims[k] for k in self._action_keys],
            self._state_keys,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def action_keys(self) -> List[str]:
        return list(self._action_keys)

    @property
    def state_keys(self) -> List[str]:
        return list(self._state_keys)

    @property
    def video_keys(self) -> List[str]:
        """Camera views in TRAINING order (``DataConfig.video_keys``).

        The order is a serve-time contract: nothing downstream identifies a view
        by name, so the server must lay images out exactly as training did.
        Empty only for a DataConfig that declares no video keys.
        """
        return list(self._video_keys)

    @property
    def action_key_dims(self) -> Dict[str, int]:
        return dict(self._action_key_dims)

    @property
    def state_key_dims(self) -> Dict[str, int]:
        return dict(self._state_key_dims)

    @property
    def unnorm_key(self) -> str:
        return self._unnorm_key

    @property
    def available_unnorm_keys(self) -> List[str]:
        return list(self._norm_stats.keys())

    @property
    def transform(self) -> ComposedModalityTransform:
        return self._transform

    # ------------------------------------------------------------------
    # Forward path (env proprioception → model input)
    # ------------------------------------------------------------------
    def apply_state(self, raw_state: np.ndarray) -> np.ndarray:
        """Normalize proprioception with the TRAINING pipeline.

        The module docstring has promised this method since the class was
        written, but it did not exist — so every client that switched state on
        sent RAW joint angles to a checkpoint trained on q99-normalised ones.
        That is why the pipette lane serves ``--no_state``: not because state
        was useless, but because the only available spelling of it was wrong.

        Args:
            raw_state: ``(T, D)`` or ``(D,)`` in ENV units (radians, registers),
                laid out as ``state_keys`` concatenated in DataConfig order.
                ``T`` is the state history depth; every frame is normalised
                with the same per-key statistics, exactly as the dataloader
                does when ``state_indices`` has more than one entry.

        Returns:
            The same shape, normalised — ready to hand to the action head.

        Only the state-side transforms run. Video transforms are skipped
        (there are no video keys in ``data``), and the state ones document
        that they "allow some keys to be missing", so restricting the dict to
        state keys is the pipeline's own supported behaviour rather than a
        parallel implementation of the normalisation math.
        """
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
            StateActionToTensor,
            StateActionTransform,
        )

        arr = np.asarray(raw_state, dtype=np.float32)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected (T, D) or (D,); got shape {arr.shape}")

        sum_dims = sum(self._state_key_dims.get(k, 1) for k in self._state_keys)
        if arr.shape[-1] != sum_dims:
            raise ValueError(
                f"state has {arr.shape[-1]} dims but the DataConfig's state "
                f"keys require {sum_dims}. state_keys={self._state_keys}, "
                f"state_key_dims={self._state_key_dims}"
            )

        data: Dict[str, Any] = {}
        cursor = 0
        for full_key in self._state_keys:
            dim_k = self._state_key_dims.get(full_key, 1)
            data[full_key] = arr[:, cursor : cursor + dim_k]
            cursor += dim_k

        state_side = set(self._state_keys)
        for transform in self._transform.transforms:
            if not isinstance(transform, (StateActionToTensor, StateActionTransform)):
                continue
            if not state_side.intersection(getattr(transform, "apply_to", [])):
                continue
            data = transform(data)

        parts: List[np.ndarray] = []
        for full_key in self._state_keys:
            v = data[full_key]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            parts.append(np.asarray(v, dtype=np.float32))
        out = np.concatenate(parts, axis=-1)
        return out[0] if squeeze else out

    # ------------------------------------------------------------------
    # Inverse path (model output → env action)
    # ------------------------------------------------------------------
    def unapply_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        """Invert action normalization using the training-time pipeline.

        Args:
            normalized_actions: shape ``(T, D)`` where ``D`` is at least
                ``sum(action_key_dims.values())``. Fixed-action-dim backbones
                (PI0/PI05 use ``action_dim=32``) right-pad the action to a
                constant width; the extra trailing columns are zero pad and are
                dropped here. GR00T emits exactly the summed width (no pad).

        Returns:
            ``(T, sum(action_key_dims))`` un-normalized actions in env coords.
        """
        normalized_actions = np.asarray(normalized_actions)
        assert normalized_actions.ndim == 2, (
            f"Expected (T, D); got shape {normalized_actions.shape}"
        )

        # The un-normalization pipeline is defined on the un-padded action
        # (one slot per action key). PI0/PI05 pad the model's action to a fixed
        # 32 by appending zeros AFTER the real dims, so the true content is the
        # first ``sum_dims`` columns. Drop any trailing pad before splitting;
        # for GR00T (model width == sum_dims) this is a no-op.
        sum_dims = sum(self._action_key_dims.get(k, 1) for k in self._action_keys)
        model_dim = normalized_actions.shape[-1]
        if model_dim < sum_dims:
            raise ValueError(
                f"Model emitted {model_dim} action dims but the DataConfig's "
                f"action keys require {sum_dims}. "
                f"action_keys={self._action_keys}, "
                f"action_key_dims={self._action_key_dims}"
            )
        if model_dim > sum_dims:
            normalized_actions = normalized_actions[..., :sum_dims]

        # Split (T, sum_dims) into per-key {full_key: torch.Tensor[T, dim_k]}.
        data: Dict[str, torch.Tensor] = {}
        cursor = 0
        for full_key in self._action_keys:
            dim_k = self._action_key_dims.get(full_key, 1)
            slice_ = normalized_actions[..., cursor : cursor + dim_k]
            data[full_key] = torch.as_tensor(slice_, dtype=torch.float32)
            cursor += dim_k

        out = self._transform.unapply(data)

        parts: List[np.ndarray] = []
        for full_key in self._action_keys:
            v = out[full_key]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            parts.append(np.asarray(v))
        return np.concatenate(parts, axis=-1)
