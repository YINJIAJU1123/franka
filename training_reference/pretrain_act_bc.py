#!/usr/bin/env python3
"""
pretrain_act_bc.py — ACT BC 预训练脚本（改进版）

改进点:
  1. 非backbone参数用更高学习率(1e-4)，backbone保持低学习率(1e-5)
  2. Cosine LR schedule + linear warmup
  3. ColorJitter 数据增强（真机小数据集必需）
  4. 按 validation loss 保存最优模型
  5. EMA (Exponential Moving Average) 平滑权重
  6. Epoch 感知训练，确保数据充分利用
  7. 更大的 batch_size (32)

用法:
    python3 pretrain_act_bc.py \
        --config train_config_act_franka_gripper.json \
        --data_dir datasets/franka_gripper \
        --output_dir experiments/pick_apple/act_bc \
        --steps 50000

输出:
    experiments/pick_apple/act_bc/
        ├── act_policy.pth          # 最终 policy state_dict
        ├── act_policy_best.pth     # 最优 val loss 的 checkpoint
        ├── act_policy_ema.pth      # EMA 权重
        ├── act_config.json         # 训练时用的 config
        └── dataset_stats.pth       # 归一化统计量（真机推理时要用）
"""
import argparse
import copy
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from gripper_training_validation import observation_only_validation, transition_sampling_weights
import torchvision.transforms as T
import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LEROBOT_SRC = os.path.join(_THIS_DIR, "lerobot", "src")
if os.path.isdir(_LEROBOT_SRC) and _LEROBOT_SRC not in sys.path:
    sys.path.insert(0, _LEROBOT_SRC)

from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.configs.types import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed
from buffer_cache_utils import (
    CAMERA_KEYS,
    load_standard_buffer_cache,
    standard_image_preprocessing,
    validate_gripper_buffer_cache,
    validate_gripper_features,
)


# ── EMA (Exponential Moving Average) ──────────────────────────────────
class EMAModel:
    """Exponential Moving Average of model parameters for stabler inference."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.clone()

    def state_dict(self):
        return self.shadow


# ── Cosine LR Scheduler with Linear Warmup ────────────────────────────
class CosineWarmupScheduler:
    """Cosine decay with linear warmup. Adjusts all param groups."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def apply_color_jitter(batch, jitter_fn):
    """对 batch 里所有 observation.images.* 施加 ColorJitter 增强。"""
    for k in list(batch.keys()):
        v = batch[k]
        if "images" not in k:
            continue
        if not torch.is_tensor(v) or v.ndim != 4:
            continue
        # ColorJitter 期望 (B, C, H, W) float [0,1]
        batch[k] = jitter_fn(v)
    return batch


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def compute_vector_stats(tensor: torch.Tensor) -> dict[str, np.ndarray]:
    arr = tensor.cpu().numpy().astype(np.float64, copy=False)
    return {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "count": np.array([arr.shape[0]], dtype=np.int64),
    }


class CachedActDataset(torch.utils.data.Dataset):
    def __init__(self, indices, states, actions, image_cache, cam_keys, chunk_size, ep_to, frame_to_ep, action_offset=0):
        self.action_offset = int(action_offset)
        if self.action_offset < 0:
            raise ValueError("action_offset must be nonnegative")
        self.indices = [i for i in indices if i + self.action_offset < int(ep_to[int(frame_to_ep[i])])]
        self.states = states
        self.actions = actions
        self.image_cache = image_cache
        self.cam_keys = cam_keys
        self.chunk_size = chunk_size
        self.ep_to = ep_to
        self.frame_to_ep = frame_to_ep

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        gi = self.indices[idx]
        ep = int(self.frame_to_ep[gi])
        ep_end = int(self.ep_to[ep])
        item = {
            "observation.state": self.states[gi],
            "index": torch.tensor(gi, dtype=torch.long),
        }
        action_indices = [min(gi + self.action_offset + k, ep_end - 1) for k in range(self.chunk_size)]
        item["action"] = self.actions[action_indices]
        item["action_is_pad"] = torch.tensor(
            [gi + self.action_offset + k >= ep_end for k in range(self.chunk_size)], dtype=torch.bool
        )
        for cam_key in self.cam_keys:
            item[cam_key] = self.image_cache[cam_key][gi]
        return item


# ── 主训练循环 ──────────────────────────────────────────────────────────
def main(args):
    # 1. 读 config
    with open(args.config) as f:
        raw_cfg = json.load(f)

    policy_raw = dict(raw_cfg["policy"])
    policy_raw.pop("type", None)
    if "normalization_mapping" in policy_raw:
        policy_raw["normalization_mapping"] = {
            k: NormalizationMode(v) for k, v in policy_raw["normalization_mapping"].items()
        }

    steps       = args.steps if args.steps is not None else raw_cfg.get("steps", 50000)
    batch_size  = args.batch_size if args.batch_size is not None else raw_cfg.get("batch_size", 32)
    num_workers = args.num_workers if args.num_workers is not None else raw_cfg.get("num_workers", 4)
    log_freq    = raw_cfg.get("log_freq", 100)
    save_freq   = raw_cfg.get("save_freq", 2000)
    device_str  = args.device or raw_cfg.get("device", "cuda")
    device      = get_safe_torch_device(try_device=device_str, log=True)
    # make_policy moves to policy_cfg.device during construction.
    policy_raw["device"] = str(device)
    seed = int(raw_cfg.get("seed", 1000))
    set_seed(seed)
    if steps < 1 or batch_size < 1 or num_workers < 0 or log_freq < 1 or save_freq < 1:
        raise ValueError("steps, batch_size, log/save frequency must be positive; workers >= 0")

    image_shape = raw_cfg.get("dataset", {}).get("image_resize", [64, 64])
    if len(image_shape) != 2 or image_shape[0] != image_shape[1] or image_shape[0] <= 0:
        raise ValueError("dataset.image_resize must be a positive square [size, size]")
    image_target = int(image_shape[0])
    crop_mode = raw_cfg.get("dataset", {}).get("crop_mode", "right")
    # per-camera 处理模式: {"cam_wrist": "pad", "cam_head": "right"}
    cam_modes = raw_cfg.get("dataset", {}).get("cam_modes", standard_image_preprocessing()["cam_modes"])

    # Training settings
    train_cfg = raw_cfg.get("training", {})
    warmup_steps    = train_cfg.get("warmup_steps", 500)
    min_lr_ratio    = train_cfg.get("min_lr_ratio", 0.01)
    use_color_jitter = train_cfg.get("color_jitter", True)
    val_split_ratio = train_cfg.get("val_split_ratio", 0.05)
    val_freq        = train_cfg.get("val_freq", 500)
    ema_decay       = train_cfg.get("ema_decay", 0.999)
    if not 0 < val_split_ratio < 1 or val_freq < 1:
        raise ValueError("val_split_ratio must be in (0, 1), val_freq must be positive")

    # 2. 输出目录 + TensorBoard
    os.makedirs(args.output_dir, exist_ok=True)
    tb_dir = os.path.join(args.output_dir, "tb_logs")
    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=tb_dir)
        print(f"[pretrain_act_bc] tensorboard  : {tb_dir}")
    else:
        tb_writer = None
        print(f"[pretrain_act_bc] tensorboard  : not available (pip install tensorboard)")
    # CSV 日志（始终写入，tensorboard 不可用时也能画图）
    csv_path = os.path.join(args.output_dir, "train_log.csv")
    init_ckpt_path = args.init_checkpoint
    csv_file = open(csv_path, "w")
    if csv_file.tell() == 0:
        csv_file.write("step,train_loss,train_l1,train_kld,lr,epoch,val_loss,best_val_loss\n")
    print(f"[pretrain_act_bc] csv log      : {csv_path}")
    print(f"[pretrain_act_bc] output dir   : {args.output_dir}")
    print(f"[pretrain_act_bc] steps        : {steps}")
    if init_ckpt_path:
        print(f"[pretrain_act_bc] initialize weights from: {init_ckpt_path}; training starts at step 0")
    print(f"[pretrain_act_bc] batch_size   : {batch_size}")
    print(f"[pretrain_act_bc] warmup_steps : {warmup_steps}")
    print(f"[pretrain_act_bc] color_jitter : {use_color_jitter}")
    print(f"[pretrain_act_bc] ema_decay    : {ema_decay}")

    # 3. 加载数据集
    chunk_size = policy_raw.get("chunk_size", 32)
    action_offset = int(raw_cfg.get("dataset", {}).get("action_offset_frames", 0))
    data_abs   = os.path.abspath(args.data_dir)
    repo_id    = os.path.basename(data_abs)

    with open(os.path.join(data_abs, "meta", "info.json")) as _f:
        _info = json.load(_f)
    validate_gripper_features(_info["features"])
    if int(_info["total_episodes"]) < 2:
        raise ValueError("ACT-gripper training requires at least two episodes")
    fps = _info["fps"]
    delta_timestamps = {
        "action": [(i + action_offset) / fps for i in range(chunk_size)],
    }

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=data_abs,
        revision="v2.1",
        delta_timestamps=delta_timestamps,
    )
    print(f"[pretrain_act_bc] dataset: {len(dataset)} frames, "
          f"{dataset.num_episodes} episodes, fps={fps}")

    # 4. Train/val split (按 episode 切分，避免数据泄露)
    all_episodes = list(range(dataset.num_episodes))
    n_val = min(len(all_episodes) - 1, max(1, int(len(all_episodes) * val_split_ratio)))
    rng = np.random.RandomState(seed)
    rng.shuffle(all_episodes)
    val_episodes = set(all_episodes[:n_val])
    train_episodes = set(all_episodes[n_val:])
    print(f"[pretrain_act_bc] train episodes: {len(train_episodes)}, "
          f"val episodes: {len(val_episodes)}")

    # Build indices for train/val (利用 episode_data_index 避免逐样本遍历)
    train_indices, val_indices = [], []
    ep_index = dataset.episode_data_index
    for ep in range(dataset.num_episodes):
        start = int(ep_index["from"][ep].item())
        end   = int(ep_index["to"][ep].item())
        indices = list(range(start, end))
        if ep in val_episodes:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    # 4.5 统一走 buffer_cache.pt，避免训练时再解码视频
    cache_path = os.path.join(data_abs, "buffer_cache.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"buffer_cache not found: {cache_path}. "
            "Generate it first with scripts/cache/precompute_buffer_cache.py."
        )
    print(f"[pretrain_act_bc] Loading buffer cache: {cache_path}")
    cache = load_standard_buffer_cache(
        cache_path,
        map_location="cpu",
        expected={
            **standard_image_preprocessing(image_target),
            "image_size": image_target,
            "crop_mode": crop_mode,
            "cam_modes": cam_modes,
        },
    )
    if len(dataset) != int(_info["total_frames"]):
        raise ValueError("Loaded dataset length does not match metadata total_frames")
    if dataset.num_episodes != int(_info["total_episodes"]):
        raise ValueError("Loaded dataset episode count does not match metadata")
    validate_gripper_buffer_cache(cache, dataset.episode_data_index, image_size=image_target)
    if cache["state"].shape[0] != len(dataset):
        raise ValueError("Cache length does not match dataset")
    dataset_episode_ids = torch.stack(dataset.hf_dataset["episode_index"]).reshape(-1).long().cpu()
    if not torch.equal(cache["episode_index"], dataset_episode_ids):
        raise ValueError("Cache episode_index does not match parquet row order")
    cam_keys_in_cache = list(CAMERA_KEYS)
    for k in cam_keys_in_cache:
        print(f"  {k}: {tuple(cache[k].shape)}")
    all_states = cache["state"].to(dtype=torch.float32)
    all_actions = cache["action"].to(dtype=torch.float32)
    print(f"  states: {tuple(all_states.shape)}, actions: {tuple(all_actions.shape)}")
    ep_from = dataset.episode_data_index["from"].cpu().numpy().astype(np.int64)
    ep_to = dataset.episode_data_index["to"].cpu().numpy().astype(np.int64)
    frame_to_ep = np.empty(len(dataset), dtype=np.int64)
    for ep in range(dataset.num_episodes):
        frame_to_ep[int(ep_from[ep]):int(ep_to[ep])] = ep

    image_cache = {k: cache[k] for k in cam_keys_in_cache}
    train_dataset = CachedActDataset(
        train_indices, all_states, all_actions, image_cache, cam_keys_in_cache,
        chunk_size, ep_to, frame_to_ep, action_offset,
    )
    val_dataset = CachedActDataset(
        val_indices, all_states, all_actions, image_cache, cam_keys_in_cache,
        chunk_size, ep_to, frame_to_ep, action_offset,
    )
    print(f"[pretrain_act_bc] Using cached dataset! train={len(train_dataset)}, val={len(val_dataset)}")

    # 5. 构建 ACT policy
    policy_cfg = make_policy_config("act", **policy_raw)

    patched_features = copy.deepcopy(dataset.meta.features)
    for feat_key, feat_info in patched_features.items():
        if feat_info.get("dtype") in ("image", "video"):
            fi = dict(feat_info)
            fi["shape"] = [image_target, image_target, 3]
            patched_features[feat_key] = fi
    patched_stats = copy.deepcopy(dataset.meta.stats)
    patched_stats["observation.state"] = compute_vector_stats(all_states[train_indices])
    patched_stats["action"] = compute_vector_stats(all_actions[train_indices])
    mock_meta = types.SimpleNamespace(
        features=patched_features,
        stats=patched_stats,
    )
    policy = make_policy(policy_cfg, ds_meta=mock_meta).to(device)
    if init_ckpt_path:
        ckpt = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["policy_state_dict"] if "policy_state_dict" in ckpt else ckpt
        policy.load_state_dict(state_dict, strict=True)
    policy.train()
    print(f"[pretrain_act_bc] policy: ACT, chunk_size={chunk_size}, "
          f"dim_model={policy_cfg.dim_model}, use_vae={policy_cfg.use_vae}")
    print(f"[pretrain_act_bc] lr={policy_cfg.optimizer_lr}, "
          f"lr_backbone={policy_cfg.optimizer_lr_backbone}")

    # Write deployable metadata before any best checkpoint can be produced.
    save_cfg = copy.deepcopy(raw_cfg)
    save_cfg.update(
        data_dir=data_abs, steps=steps, batch_size=batch_size, num_workers=num_workers,
        device=str(device), seed=seed, init_checkpoint=init_ckpt_path,
    )
    save_cfg["policy"]["device"] = str(device)
    save_cfg["dataset"] = {
        **save_cfg.get("dataset", {}),
        "image_resize": [image_target, image_target],
        "crop_mode": crop_mode, "cam_modes": cam_modes,
    }
    save_cfg["image_preprocessing"] = cache["image_preprocessing"]
    with open(os.path.join(args.output_dir, "act_config.json"), "w") as config_file:
        json.dump(save_cfg, config_file, indent=2)
    torch.save(patched_stats, os.path.join(args.output_dir, "dataset_stats.pth"))

    # 6. DataLoaders
    sampler = None
    transition_weight = float(train_cfg.get("transition_sampling_weight", 1.0))
    if transition_weight > 1.0:
        weights = transition_sampling_weights(train_dataset, transition_weight,
            int(train_cfg.get("transition_sampling_radius_frames", 45)))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print(f"[pretrain_act_bc] transition sampler: weight={transition_weight}, emphasized={int((weights > 1).sum())}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )

    # 7. Optimizer (分 backbone / non-backbone 学习率)
    optim_params = policy.get_optim_params()
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=policy_cfg.optimizer_lr,
        weight_decay=policy_cfg.optimizer_weight_decay,
    )

    # 8. LR Scheduler (cosine + warmup)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=warmup_steps,
        total_steps=steps, min_lr_ratio=min_lr_ratio,
    )

    # 9. EMA
    ema = EMAModel(policy, decay=ema_decay)

    # 10. ColorJitter + 光照鲁棒性增强（brightness/contrast/hue 加大 + 额外空间增强）
    jitter_fn = None
    if use_color_jitter:
        jitter_fn = T.Compose([
            T.ColorJitter(
                brightness=train_cfg.get("color_jitter_brightness", 0.3),
                contrast=train_cfg.get("color_jitter_contrast", 0.3),
                saturation=train_cfg.get("color_jitter_saturation", 0.3),
                hue=train_cfg.get("color_jitter_hue", 0.05),
            ),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            T.RandomGrayscale(p=0.05),
            T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),
            T.RandomAutocontrast(p=0.2),
        ])
        print(f"[pretrain_act_bc] color_jitter + extras: {jitter_fn}")

    # 11. 训练循环
    train_iter = iter(train_loader)
    loss_running = {"total": 0.0, "l1": 0.0, "kld": 0.0, "arm_l1": 0.0, "hand_l1": 0.0}
    t0 = time.time()
    best_val_loss = float("inf")
    best_event_score = None
    epoch = 0
    samples_seen = 0
    total_dataset_size = len(train_dataset)

    pbar = tqdm.tqdm(range(1, steps + 1), desc="ACT BC pretrain")
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = move_to_device(batch, device)
        # 训练时数据增强
        if jitter_fn is not None:
            batch = apply_color_jitter(batch, jitter_fn)

        loss, loss_dict = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        # EMA update
        ema.update(policy)

        samples_seen += batch["observation.state"].shape[0]
        cur_epoch = samples_seen / total_dataset_size

        loss_running["total"] += loss.item()
        loss_running["l1"]    += loss_dict.get("l1_loss", 0.0)
        loss_running["kld"]   += loss_dict.get("kld_loss", 0.0)
        loss_running["arm_l1"]  += loss_dict.get("arm_l1_loss", 0.0)
        loss_running["hand_l1"] += loss_dict.get("hand_l1_loss", 0.0)

        if step % log_freq == 0:
            avg = {k: v / log_freq for k, v in loss_running.items()}
            elapsed = time.time() - t0
            lr_now = scheduler.get_lr()[0]
            pbar.set_postfix({
                "loss": f"{avg['total']:.4f}",
                "l1":   f"{avg['l1']:.4f}",
                "kld":  f"{avg['kld']:.4f}",
                "arm":  f"{avg['arm_l1']:.4f}",
                "hand": f"{avg['hand_l1']:.4f}",
                "epoch": f"{cur_epoch:.1f}",
                "lr":   f"{lr_now:.2e}",
            })
            # TensorBoard + CSV: train losses + lr
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss_total", avg["total"], step)
                tb_writer.add_scalar("train/loss_l1", avg["l1"], step)
                tb_writer.add_scalar("train/loss_kld", avg["kld"], step)
                tb_writer.add_scalar("train/loss_arm_l1", avg["arm_l1"], step)
                tb_writer.add_scalar("train/loss_hand_l1", avg["hand_l1"], step)
                tb_writer.add_scalar("train/lr", lr_now, step)
                tb_writer.add_scalar("train/epoch", cur_epoch, step)
            csv_file.write(f"{step},{avg['total']:.6f},{avg['l1']:.6f},{avg['kld']:.6f},"
                           f"{lr_now:.2e},{cur_epoch:.2f},,\n")
            csv_file.flush()
            loss_running = {"total": 0.0, "l1": 0.0, "kld": 0.0, "arm_l1": 0.0, "hand_l1": 0.0}

        # Validation
        if step % val_freq == 0 or step == steps:
            event_metrics = None
            if train_cfg.get("deployment_validation", False):
                val_loss, event_metrics = observation_only_validation(policy, val_loader, device, fps)
                event_metrics.update(step=step, val_loss=val_loss)
                with open(os.path.join(args.output_dir, "gripper_validation.jsonl"), "a") as event_log:
                    event_log.write(json.dumps(event_metrics) + "\n")
                event_score = tuple(event_metrics["selection_score"])
                if best_event_score is None or event_score < best_event_score:
                    best_event_score = event_score
                    torch.save({"step": step, "policy_state_dict": policy.state_dict(),
                        "gripper_validation": event_metrics}, os.path.join(args.output_dir, "act_policy_event_best.pth"))
                    print(f"  [gripper] new event best: {event_metrics}", flush=True)
            else:
                val_loss = _run_validation(policy, val_loader, device)
            print(f"\n  [val] step={step}, val_loss={val_loss:.4f}, "
                  f"best={best_val_loss:.4f}, epoch={cur_epoch:.1f}")
            # TensorBoard: val loss
            if tb_writer is not None:
                tb_writer.add_scalar("val/loss", val_loss, step)
                tb_writer.add_scalar("val/best_loss", best_val_loss, step)
            # CSV: 回填 val_loss 到最近一行
            csv_file.write(f"{step},,,,,,{val_loss:.6f},{min(best_val_loss, val_loss):.6f}\n")
            csv_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_best = {
                    "step": step,
                    "val_loss": val_loss,
                    "policy_state_dict": policy.state_dict(),
                }
                torch.save(ckpt_best, os.path.join(args.output_dir, "act_policy_best.pth"))
                print(f"  [val] ★ new best model saved (val_loss={val_loss:.4f})")
            policy.train()

        # Periodic save
        if step % save_freq == 0 or step == steps:
            ckpt_data = {"step": step, "policy_state_dict": policy.state_dict()}
            torch.save(ckpt_data, os.path.join(
                args.output_dir, f"act_policy_step{step:06d}.pth"))
            torch.save(ckpt_data, os.path.join(args.output_dir, "act_policy.pth"))

    # 12. 保存 EMA 权重
    ema_ckpt = {"step": steps, "policy_state_dict": ema.state_dict()}
    torch.save(ema_ckpt, os.path.join(args.output_dir, "act_policy_ema.pth"))

    if tb_writer is not None:
        tb_writer.close()
    csv_file.close()
    print(f"\n{'='*60}")
    print(f"Training complete! epochs={cur_epoch:.1f}, best_val_loss={best_val_loss:.4f}")
    print(f"  act_policy.pth      — final checkpoint")
    print(f"  act_policy_best.pth — best validation loss")
    print(f"  act_policy_ema.pth  — EMA smoothed weights")
    print(f"  dataset_stats.pth   — normalization stats")
    print(f"{'='*60}")


@torch.no_grad()
def _run_validation(policy, val_loader, device):
    """Compute sample-weighted loss without updating BatchNorm or enabling dropout."""
    was_training = policy.training
    policy.eval()
    total_loss, samples = 0.0, 0
    try:
        for batch in val_loader:
            batch = move_to_device(batch, device)
            loss, _ = policy.forward(batch)
            count = batch["observation.state"].shape[0]
            total_loss += loss.item() * count
            samples += count
    finally:
        policy.train(was_training)
    if samples == 0:
        raise ValueError("Validation dataset is empty")
    result = total_loss / samples
    if not math.isfinite(result):
        raise ValueError(f"Validation loss is nonfinite: {result}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     required=True,
                    help="train_config_act_franka_gripper.json")
    ap.add_argument("--data_dir",   required=True,
                    help="Path to a complete 8D, dual-camera LeRobot dataset")
    ap.add_argument("--output_dir", required=True,
                    help="Where to save act_policy.pth and dataset_stats.pth")
    ap.add_argument("--steps",       type=int, default=None)
    ap.add_argument("--batch_size",  type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--device", default=None,
                    help="Override config device (e.g. cpu for a smoke test).")
    ap.add_argument("--init_checkpoint", default=None,
                    help="Strictly load model weights; start a new training run at step 0.")
    args = ap.parse_args()
    main(args)
