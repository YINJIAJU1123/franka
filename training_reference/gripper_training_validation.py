"""Deployment-only validation and transition sampling for Franka ACT training."""
import math
import numpy as np
import torch


def transition_sampling_weights(dataset, multiplier, radius):
    if multiplier < 1 or radius < 0:
        raise ValueError('Invalid transition sampling settings')
    action = dataset.actions[:, 7].cpu().numpy() >= .5
    changes = np.flatnonzero((action[1:] != action[:-1]) &
                            (dataset.frame_to_ep[1:] == dataset.frame_to_ep[:-1])) + 1
    anchors = np.asarray(dataset.indices)
    weights = np.ones(len(anchors), dtype=np.float64)
    for change in changes:
        near = (np.abs(anchors + dataset.action_offset - change) <= radius)
        near &= dataset.frame_to_ep[anchors] == dataset.frame_to_ep[change]
        weights[near] = multiplier
    return torch.as_tensor(weights)


def binary_trace(raw, initial=False):
    result = []
    previous = initial
    for value in raw:
        previous = bool(value > .45) if previous else bool(value >= .55)
        result.append(previous)
    return np.asarray(result, dtype=bool)


def events(trace, frames):
    changed = np.flatnonzero(trace != np.r_[False, trace[:-1]])
    return [{'type': 'close' if trace[i] else 'open', 'frame': int(frames[i])} for i in changed]


def event_metrics(indices, predicted, target, dataset, fps):
    indices = np.asarray(indices)
    predicted, target = np.asarray(predicted), np.asarray(target) >= .5
    ids = dataset.frame_to_ep[indices]
    episodes, errors = [], []
    count_errors = 0
    all_binary = np.zeros(len(indices), dtype=bool)
    for ep in np.unique(ids):
        mask = ids == ep
        ep_start = int(np.flatnonzero(dataset.frame_to_ep == ep)[0])
        frames = indices[mask] - ep_start
        binary = binary_trace(predicted[mask])
        all_binary[mask] = binary
        pe, te = events(binary, frames), events(target[mask], frames)
        pair_errors = []
        for kind in ['close', 'open']:
            pp = [e['frame'] for e in pe if e['type'] == kind]
            tt = [e['frame'] for e in te if e['type'] == kind]
            count_errors += abs(len(pp) - len(tt))
            for pf, tf in zip(pp, tt):
                error = (pf - tf) / fps
                errors.append(abs(error)); pair_errors.append({'type': kind,
                    'expected_command_frame': tf, 'recorded_label_frame': tf + dataset.action_offset,
                    'predicted_frame': pf, 'error_s': error})
        episodes.append({'episode': int(ep), 'predicted_events': pe, 'expected_events': te,
                         'event_timing': pair_errors,
                         'binary_accuracy': float((binary == target[mask]).mean())})
    return {'episodes': episodes, 'event_count_errors': count_errors,
            'max_abs_timing_error_s': max(errors) if errors else None,
            'mean_abs_timing_error_s': float(np.mean(errors)) if errors else None,
            'gripper_accuracy': float((all_binary == target).mean()),
            'gripper_balanced_accuracy': float(np.mean([(all_binary[target == x] == x).mean()
                for x in [False, True] if np.any(target == x)])),
            'action_offset_frames': dataset.action_offset}


@torch.no_grad()
def observation_only_validation(policy, loader, device, fps):
    was_training = policy.training
    policy.eval()
    error_sum = 0.; valid_components = 0
    indices, predictions, targets = [], [], []
    try:
        for original in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in original.items()}
            observations = {k: v for k, v in batch.items() if k.startswith('observation.')}
            assert 'action' not in observations and 'action_is_pad' not in observations
            predicted = policy.predict_action_chunk(observations)
            if not bool(torch.isfinite(predicted).all()):
                raise ValueError('Nonfinite deployment prediction')
            normal_pred = policy.normalize_targets({'action': predicted})['action']
            normal_target = policy.normalize_targets({'action': batch['action']})['action']
            valid = ~batch['action_is_pad']
            weight = torch.ones(predicted.shape[-1], device=device)
            weight[:policy.config.arm_action_dim] = policy.config.arm_loss_weight
            weight[policy.config.arm_action_dim:] = policy.config.hand_loss_weight
            error_sum += float(((normal_pred-normal_target).abs()*weight*valid[..., None]).sum())
            valid_components += int(valid.sum())*predicted.shape[-1]
            indices.extend(original['index'].tolist())
            predictions.extend(predicted[:, 0, 7].cpu().tolist())
            targets.extend(batch['action'][:, 0, 7].cpu().tolist())
    finally:
        policy.train(was_training)
    loss = error_sum / valid_components
    if not math.isfinite(loss):
        raise ValueError('Nonfinite deployment validation loss')
    metrics = event_metrics(indices, predictions, targets, loader.dataset, fps)
    metrics['selection_score'] = [metrics['event_count_errors'],
        metrics['max_abs_timing_error_s'] if metrics['max_abs_timing_error_s'] is not None else 100.,
        metrics['mean_abs_timing_error_s'] if metrics['mean_abs_timing_error_s'] is not None else 100., loss]
    return loss, metrics
