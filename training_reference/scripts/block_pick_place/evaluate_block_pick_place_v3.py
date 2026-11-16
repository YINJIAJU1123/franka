"""Offline ACT evaluation from cached validation episodes; never imports a robot transport."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def summarize(pred, target, mask, current):
    pred, target = pred[mask], target[mask]
    error = pred[:, :7] - target[:, :7]
    y, yh = target[:, 7] >= .5, pred[:, 7] >= .5
    recalls = [float((yh[y == v] == v).mean()) for v in [False, True] if np.any(y == v)]
    result = dict(samples=int(len(pred)), arm_mae_rad=float(np.abs(error).mean()),
                  arm_rmse_rad=float(np.sqrt(np.square(error).mean())),
                  gripper_accuracy=float((yh == y).mean()), gripper_balanced_accuracy=float(np.mean(recalls)))
    moving = np.max(np.abs(target[:, :7] - current[mask, :7]), axis=1) > .001
    transition = y != (current[mask, 7] >= .5)
    result['moving_samples'] = int(moving.sum())
    result['moving_arm_mae_rad'] = float(np.abs(error[moving]).mean()) if moving.any() else None
    result['transition_samples'] = int(transition.sum())
    result['transition_accuracy'] = float((yh[transition] == y[transition]).mean()) if transition.any() else None
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default='/home/lixin/act_gripper')
    ap.add_argument('--run', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()
    project, run, output = Path(args.project).resolve(), Path(args.run).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new evaluation output directory: {output}')
    output.mkdir(parents=True)
    sys.path.insert(0, str(project))
    from act_gripper_policy import load_act_gripper_policy
    from buffer_cache_utils import load_standard_buffer_cache
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(8)
    os.environ['N_ACTION_STEPS'] = '1'
    cfg = json.loads((run / 'act_config.json').read_text())
    root = Path(cfg['data_dir'])
    info = json.loads((root / 'meta/info.json').read_text())
    entries = [json.loads(x) for x in (root / 'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
    split = json.loads((project / 'configs/block_pick_place_v3_split.json').read_text())
    cache = load_standard_buffer_cache(root / 'buffer_cache.pt')
    states, actions = cache['state'].float(), cache['action'].float()
    cameras = ['observation.images.cam_head', 'observation.images.cam_wrist']
    ends = np.cumsum([x['length'] for x in entries])
    starts = np.r_[0, ends[:-1]]
    val_ids = split['validation_new_ids']
    indices = np.concatenate([np.arange(starts[ep], ends[ep]) for ep in val_ids])
    ep_for_row = np.concatenate([np.full(ends[ep] - starts[ep], ep) for ep in val_ids])
    current = states[indices].numpy()
    device = torch.device(args.device)
    stats = torch.load(run / 'dataset_stats.pth', map_location='cpu', weights_only=False)
    amin, amax = [np.asarray(stats['action'][key], dtype=np.float32) for key in ['min', 'max']]
    smin, smax = [torch.as_tensor(stats['observation.state'][key], dtype=torch.float32, device=device) for key in ['min', 'max']]
    lo, hi = [torch.as_tensor(x, device=device) for x in [amin, amax]]
    chunk = int(cfg['policy']['chunk_size'])
    offsets = [0, 1, 5, 10, 30, chunk - 1]
    all_results, stored = [], {}
    val_log = [r for r in csv.DictReader((run / 'train_log.csv').open()) if r.get('val_loss')]
    best_log = min(val_log, key=lambda r: float(r['val_loss']))
    for filename in ['act_policy_best.pth', 'act_policy_ema.pth', 'act_policy.pth']:
        policy, _, _ = load_act_gripper_policy(run / 'act_config.json', run, device, filename)
        policy.eval()
        predictions = {k: [] for k in offsets}
        targets = {k: [] for k in offsets}
        masks = {k: [] for k in offsets}
        norm_sum = 0.; valid_components = 0; outside = 0; component_count = 0
        for first in range(0, len(indices), args.batch_size):
            ix = indices[first:first + args.batch_size]
            eps = ep_for_row[first:first + args.batch_size]
            batch = {'observation.state': torch.minimum(torch.maximum(states[ix].to(device), smin), smax)}
            batch.update({key: cache[key][ix].to(device) for key in cameras})
            # No future action or padding fields reach the policy: this matches deployment's zero latent.
            assert set(batch) == {'observation.state', *cameras}
            with torch.inference_mode():
                raw = policy.predict_action_chunk(batch)
                assert raw.shape == (len(ix), chunk, 8) and bool(torch.isfinite(raw).all())
                outside += int(((raw < lo) | (raw > hi)).sum())
                component_count += raw.numel()
                predicted = torch.minimum(torch.maximum(raw, lo), hi).cpu().numpy()
            target_ix = ix[:, None] + np.arange(chunk)[None, :]
            valid = target_ix < ends[eps, None]
            target_ix = np.minimum(target_ix, ends[eps, None] - 1)
            target = actions[target_ix].numpy()
            error = 2 * np.abs(predicted - target) / (amax - amin + 1e-8)
            norm_sum += float((error * valid[..., None]).sum())
            valid_components += int(valid.sum()) * 8
            for k in offsets:
                predictions[k].append(predicted[:, k]); targets[k].append(target[:, k]); masks[k].append(valid[:, k])
        row = dict(checkpoint=filename, step=int(best_log['step']) if 'best' in filename else int(cfg['steps']),
                   validation_frames=len(indices), deployment_normalized_l1=norm_sum / valid_components,
                   raw_prediction_outside_training_bounds_fraction=outside / component_count, horizons={})
        for k in offsets:
            p, t, m = np.concatenate(predictions[k]), np.concatenate(targets[k]), np.concatenate(masks[k])
            row['horizons'][str(k)] = summarize(p, t, m, current)
            stored[filename + '_h' + str(k)] = p
        example = {'observation.state': torch.minimum(torch.maximum(states[indices[:1]].to(device), smin), smax)}
        example.update({key: cache[key][indices[:1]].to(device) for key in cameras})
        latency = []
        with torch.inference_mode():
            for repetition in range(110):
                if device.type == 'cuda': torch.cuda.synchronize()
                start = time.perf_counter(); policy.predict_action_chunk(example)
                if device.type == 'cuda': torch.cuda.synchronize()
                if repetition >= 10: latency.append((time.perf_counter() - start) * 1000)
        row['batch1_model_latency_ms'] = {'median': float(np.median(latency)), 'p95': float(np.percentile(latency, 95))}
        all_results.append(row)
        print('DEPLOYMENT_VALIDATION', json.dumps(row), flush=True)
        del policy
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
    chosen = min(all_results, key=lambda x: x['deployment_normalized_l1'])['checkpoint']
    report = {'run': str(run), 'dataset': str(root), 'validation_episode_ids': val_ids,
              'validation_original_episode_ids': split['validation_original_ids'],
              'state_action_identity_max_abs_error': float(torch.max(torch.abs(states[indices] - actions[indices]))),
              'runtime_state_and_action_clamping': True,
              'evaluation_mode': 'Observation-only predict_action_chunk, zero VAE latent, valid-action masks; no robot interface.',
              'models': all_results, 'recommended_checkpoint_by_deployment_l1': chosen,
              'limitation': 'Demonstration observations supplied each step; these metrics are not autonomous task success rates.'}
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(output / 'predictions.npz', frame_index=indices, episode_index=ep_for_row, state=current, **stored)
    rows = list(csv.DictReader((run / 'train_log.csv').open()))
    training = [r for r in rows if r.get('train_loss')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for name, label in [('train_loss', 'Train total: L1 + 10 KL'), ('train_l1', 'Train L1')]:
        axes[0].semilogy([int(x['step']) for x in training], [float(x[name]) for x in training], label=label)
    axes[0].set(xlabel='Optimization step', ylabel='Loss (log scale)', title='Training objective'); axes[0].legend()
    vs = [int(x['step']) for x in val_log]; vl = [float(x['val_loss']) for x in val_log]
    axes[1].plot(vs, vl, label='Logged VAE validation objective')
    axes[1].scatter([int(best_log['step'])], [float(best_log['val_loss'])], color='tab:red', zorder=3)
    axes[1].annotate(f'Best {float(best_log["val_loss"]):.5f} at step {best_log["step"]}',
                     (int(best_log['step']), float(best_log['val_loss'])), xytext=(.36, .75), textcoords='axes fraction',
                     arrowprops={'arrowstyle': '->', 'color': 'tab:red'})
    axes[1].set(xlabel='Optimization step', ylabel='Validation loss', title='Episode holdout: original 2 and 4')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(output / 'training_losses.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(len(val_ids), 2, figsize=(12, 3.2 * len(val_ids)), constrained_layout=True)
    for r, ep in enumerate(val_ids):
        mask = ep_for_row == ep; t = np.arange(mask.sum()) / float(info['fps']); pred = stored[chosen + '_h10'][mask]
        target_indices = np.minimum(indices[mask] + 10, ends[ep] - 1); target = actions[target_indices].numpy()
        axes[r, 0].plot(t, target[:, 3], label='Recorded q4 at t+10'); axes[r, 0].plot(t, pred[:, 3], label='Predicted q4 at t+10', alpha=.85)
        axes[r, 0].set(ylabel='Joint 4 (rad)', title=f'Validation episode {ep}: future joint target')
        axes[r, 1].plot(t, target[:, 7], label='Recorded gripper at t+10'); axes[r, 1].plot(t, pred[:, 7], label='Predicted gripper at t+10', alpha=.85)
        axes[r, 1].axhline(.5, color='gray', ls=':', lw=1); axes[r, 1].set(ylabel='Gripper: 0=open, 1=closed', title=chosen)
        for ax in axes[r]: ax.set_xlabel('Episode time (s)'); ax.grid(alpha=.2); ax.legend()
    fig.savefig(output / 'deployment_predictions.png', dpi=180); plt.close(fig)
    print('OFFLINE_EVALUATION_COMPLETE', str(output), 'recommended', chosen, flush=True)


if __name__ == '__main__':
    main()
