"""ACT policy loading and camera preprocessing for the 8D Franka gripper model."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_LEROBOT_SRC = _THIS_DIR / "lerobot" / "src"
if (_LEROBOT_SRC / "lerobot").is_dir():
    sys.path.insert(0, str(_LEROBOT_SRC))

import draccus  # noqa: F401  # required by LeRobot's config registration
from lerobot.configs.types import NormalizationMode
from lerobot.policies.factory import make_policy, make_policy_config


from image_preprocessing import preprocess_image


def load_act_gripper_policy(
    config_path: str | os.PathLike[str],
    checkpoint_dir: str | os.PathLike[str],
    device: torch.device,
    checkpoint_file: str | os.PathLike[str] | None = None,
):
    """Load a self-contained 8D checkpoint without the training dataset."""
    config_path = Path(config_path).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    raw_config = json.loads(config_path.read_text())
    policy_raw = dict(raw_config["policy"])
    policy_raw.pop("type", None)
    if "normalization_mapping" in policy_raw:
        policy_raw["normalization_mapping"] = {
            key: NormalizationMode(value)
            for key, value in policy_raw["normalization_mapping"].items()
        }
    n_action_steps = int(os.environ.get("N_ACTION_STEPS", policy_raw.get("n_action_steps", 1)))
    policy_raw["n_action_steps"] = n_action_steps

    stats_path = checkpoint_dir / "dataset_stats.pth"
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    resize = raw_config.get("dataset", {}).get("image_resize", [64, 64])
    if len(resize) != 2 or resize[0] != resize[1]:
        raise ValueError("ACT-gripper requires a square image_resize")
    image_size = int(resize[0])
    features = {
        "observation.images.cam_head": {"dtype": "video", "shape": [image_size, image_size, 3], "names": ["height", "width", "channels"]},
        "observation.images.cam_wrist": {"dtype": "video", "shape": [image_size, image_size, 3], "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": [8]},
        "action": {"dtype": "float32", "shape": [8]},
    }
    for key in ("observation.state", "action"):
        for name in ("min", "max", "mean", "std"):
            if tuple(torch.as_tensor(stats[key][name]).shape) != (8,):
                raise ValueError(f"{key}.{name} must have 8 values")
    policy_raw["device"] = str(device)
    # Full checkpoint contains the backbone; inference never downloads ImageNet weights.
    policy_raw["pretrained_backbone_weights"] = None
    mock_meta = types.SimpleNamespace(features=features, stats=stats)
    policy_config = make_policy_config("act", **policy_raw)
    policy = make_policy(policy_config, ds_meta=mock_meta).to(device)

    if checkpoint_file:
        checkpoint_path = Path(checkpoint_file).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = checkpoint_dir / checkpoint_path
    else:
        checkpoint_path = checkpoint_dir / "act_policy_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ACT checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("policy_state_dict", checkpoint)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    print(
        f"[ACT-Gripper] loaded {checkpoint_path} step={checkpoint.get('step', '?')} "
        f"image_size={image_size} n_action_steps={n_action_steps}",
        flush=True,
    )
    if "val_loss" in checkpoint:
        print(f"[ACT-Gripper] checkpoint val_loss={checkpoint['val_loss']:.6f}", flush=True)
    return policy, raw_config, image_size
