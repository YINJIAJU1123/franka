"""Cache contract for the standalone 8D ACT gripper policy."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

import torch

STANDARD_IMAGE_SIZE = 64
STANDARD_CROP_MODE = "right"
PREPROCESSING_VERSION = "torch-bilinear-v1"
STANDARD_CAM_MODES = {"cam_head": "center", "cam_wrist": "pad"}
CAMERA_KEYS = tuple(f"observation.images.{cam}" for cam in STANDARD_CAM_MODES)


def standard_image_preprocessing(image_size: int = STANDARD_IMAGE_SIZE) -> dict[str, Any]:
    return {
        "preprocessing_version": PREPROCESSING_VERSION,
        "image_size": int(image_size),
        "crop_mode": STANDARD_CROP_MODE,
        "cam_modes": dict(STANDARD_CAM_MODES),
    }


def standard_precompute_command(dataset_root, image_size=STANDARD_IMAGE_SIZE) -> str:
    return (
        "python scripts/cache/precompute_buffer_cache.py "
        f"--dataset {shlex.quote(str(dataset_root))} --image-size {int(image_size)}"
    )


def validate_gripper_features(features):
    for key in ("observation.state", "action"):
        if list(features.get(key, {}).get("shape", [])) != [8]:
            raise ValueError(f"{key} must have shape [8] for ACT-gripper")
    cameras = {
        key for key, value in features.items()
        if value.get("dtype") in ("image", "video")
    }
    if cameras != set(CAMERA_KEYS):
        raise ValueError(f"ACT-gripper requires exactly {CAMERA_KEYS}, got {sorted(cameras)}")


def validate_buffer_cache_image_preprocessing(cache, *, cache_path=None, expected=None):
    expected = {**standard_image_preprocessing(), **(expected or {})}
    actual = cache.get("image_preprocessing")
    source = str(cache_path or "buffer_cache.pt")
    errors = []
    if not isinstance(actual, dict):
        errors.append("missing image_preprocessing metadata")
    else:
        for key in ("preprocessing_version", "image_size", "crop_mode", "cam_modes"):
            if actual.get(key) != expected[key]:
                errors.append(f"{key}: actual={actual.get(key)!r}, expected={expected[key]!r}")
    if errors:
        raise ValueError(
            f"{source}: invalid image preprocessing: {'; '.join(errors)}. "
            "Regenerate the cache with "
            + standard_precompute_command(
                Path(source).parent if cache_path else "<dataset_root>", expected["image_size"]
            )
        )
    return actual


def validate_gripper_buffer_cache(cache, episode_data_index, *, image_size=STANDARD_IMAGE_SIZE):
    """Require a complete cache in metadata row order, without skipped episodes."""
    starts = torch.as_tensor(episode_data_index["from"], dtype=torch.long).cpu()
    ends = torch.as_tensor(episode_data_index["to"], dtype=torch.long).cpu()
    if starts.ndim != 1 or ends.shape != starts.shape or len(starts) < 2:
        raise ValueError("ACT-gripper training requires at least two episodes")
    if (
        starts[0].item() != 0
        or torch.any(ends <= starts)
        or not torch.equal(starts[1:], ends[:-1])
    ):
        raise ValueError("Episode metadata must describe contiguous, nonempty episodes")
    total = int(ends[-1])
    required = {"state", "action", "done", "episode_index", *CAMERA_KEYS}
    missing = required - cache.keys()
    if missing:
        raise ValueError(f"Cache is missing {sorted(missing)}")
    cameras = {key for key in cache if key.startswith("observation.images.")}
    if cameras != set(CAMERA_KEYS):
        raise ValueError(f"Cache must contain exactly {CAMERA_KEYS}")
    for key, value in cache.items():
        if torch.is_tensor(value) and (value.ndim == 0 or value.shape[0] != total):
            raise ValueError(f"Cache {key} length does not match dataset length {total}")
    for key in ("state", "action"):
        value = cache[key]
        if not torch.is_tensor(value) or value.shape != (total, 8):
            raise ValueError(f"Cache {key} must have shape ({total}, 8)")
        if not torch.isfinite(value).all():
            raise ValueError(f"Cache {key} contains nonfinite values")
    expected_ids = torch.repeat_interleave(torch.arange(len(starts)), ends - starts)
    ids = cache["episode_index"]
    if not torch.is_tensor(ids) or ids.dtype != torch.long or not torch.equal(ids.cpu(), expected_ids):
        raise ValueError("Cache episode_index is not aligned with dataset episode metadata")
    expected_done = torch.zeros(total, dtype=torch.bool)
    expected_done[ends - 1] = True
    done = cache["done"]
    if not torch.is_tensor(done) or done.dtype != torch.bool or not torch.equal(done.cpu(), expected_done):
        raise ValueError("Cache done flags do not match episode boundaries")
    for key in CAMERA_KEYS:
        images = cache[key]
        if (
            not torch.is_tensor(images)
            or images.shape != (total, 3, image_size, image_size)
            or images.dtype != torch.float32
        ):
            raise ValueError(f"{key} must be float32 [{total}, 3, {image_size}, {image_size}]")
        # Check one episode at a time to avoid a full-cache temporary allocation.
        for start, end in zip(starts.tolist(), ends.tolist()):
            frames = images[start:end]
            if not torch.isfinite(frames).all() or torch.any(frames < 0) or torch.any(frames > 1):
                raise ValueError(f"{key} must contain finite RGB values in [0, 1]")


def load_standard_buffer_cache(cache_path, *, map_location="cpu", expected=None):
    cache = torch.load(cache_path, map_location=map_location, weights_only=False)
    validate_buffer_cache_image_preprocessing(cache, cache_path=cache_path, expected=expected)
    return cache
