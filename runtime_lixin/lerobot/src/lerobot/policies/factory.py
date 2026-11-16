"""Construct the extracted ACT policy from dataset feature metadata."""
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

def get_policy_class(name):
    if name != "act":
        raise ValueError(f"Only ACT is included, got {name!r}")
    return ACTPolicy

def make_policy_config(policy_type, **kwargs):
    get_policy_class(policy_type)
    return ACTConfig(**kwargs)

def make_policy(cfg, ds_meta=None, env_cfg=None):
    if ds_meta is None or env_cfg is not None:
        raise ValueError("ACT-gripper requires dataset metadata")
    features = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    cfg.input_features = {k: v for k, v in features.items() if k not in cfg.output_features}
    return ACTPolicy(config=cfg, dataset_stats=ds_meta.stats).to(cfg.device)
