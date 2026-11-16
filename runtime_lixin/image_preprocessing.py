"""Shared RGB preprocessing for cache generation and real-time inference."""
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN_CHW = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)


def preprocess_image(
    image_rgb: np.ndarray,
    target_size: int,
    crop_mode: str,
) -> torch.Tensor:
    """Convert ROS RGB uint8 image to one ACT image tensor in [0, 1]."""
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image_rgb.shape}")
    if target_size <= 0 or crop_mode not in ("none", "pad", "center", "right"):
        raise ValueError(f"invalid image preprocessing: {target_size}, {crop_mode}")
    tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0) / 255.0
    _, channels, height, width = tensor.shape
    if crop_mode == "pad" and height != width:
        side = max(height, width)
        padded = IMAGENET_MEAN_CHW.expand(1, channels, side, side).clone()
        y0 = (side - height) // 2
        x0 = (side - width) // 2
        padded[:, :, y0:y0 + height, x0:x0 + width] = tensor
        tensor = padded
    elif crop_mode not in ("none", "pad") and height != width:
        side = min(height, width)
        top = (height - side) // 2
        left = width - side if crop_mode == "right" else (width - side) // 2
        tensor = tensor[..., top:top + side, left:left + side]
    if tensor.shape[-2:] != (target_size, target_size):
        tensor = F.interpolate(
            tensor, size=(target_size, target_size), mode="bilinear", align_corners=False
        )
    return tensor


