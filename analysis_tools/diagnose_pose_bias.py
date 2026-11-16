"""Read-only attribution of held-out ACT target errors; no live inputs or controls."""
import contextlib
import hashlib
import json
from pathlib import Path
import socket
import sys

import av
import numpy as np
import torch

ROOT = Path('/home/lixin/act_gripper_temporal_run02_20261112')
SOURCE = Path('/home/lixin/act_gripper_baseline_run02_20261112')
OUT = ROOT / 'reports/pose_bias_diagnosis'
sys.path.insert(0, str(ROOT))
from act_gripper_policy import load_act_gripper_policy, preprocess_image
from buffer_cache_utils import load_standard_buffer_cache
from franka_gripper_control import CartesianServo
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler


@contextlib.contextmanager
def no_network():
    original = socket.socket
    attempts = []
    def denied(family=socket.AF_INET, *a, **kw):
        if family in (socket.AF_INET, socket.AF_INET6):
            attempts.append(int(family))
            raise AssertionError('No IP sockets allowed')
        return original(family, *a, **kw)
    socket.socket = denied
    try:
        yield attempts
    finally:
        socket.socket = original


def summarize(v):
    v = np.asarray(v)
    return {'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95)), 'max': float(np.max(v))}


def gripper_events(action):
    closed = False
    found = []
    for i, value in enumerate(action):
        new = value > .45 if closed else value >= .55
        if new != closed:
            found.append({'type': 'close' if new else 'open', 'frame': i, 'time_s': i / 30})
        closed = new
    return found


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'runtime_config.json').read_text())
    data = Path(json.loads((ROOT / 'offline_sources.json').read_text())['dataset'])
    cache = load_standard_buffer_cache(data / 'buffer_cache.pt')
    states = cache['state'].float().numpy()
    actions = cache['action'].float().numpy()
    assert np.max(np.abs(states - actions)) == 0
    rows = [json.loads(x) for x in (ROOT / 'recorded_metadata/meta/episodes.jsonl').read_text().splitlines()]
    ends = np.cumsum([r['length'] for r in rows]); starts = np.r_[0, ends[:-1]]
    split = json.loads((ROOT / 'recorded_metadata/split.json').read_text())
    assert split['validation_new_ids'] == [2, 4]
    stats = torch.load(ROOT / 'checkpoint/dataset_stats.pth', map_location='cpu', weights_only=False)
    lo = torch.as_tensor(stats['observation.state']['min'], dtype=torch.float32, device='cuda')
    hi = torch.as_tensor(stats['observation.state']['max'], dtype=torch.float32, device='cuda')
    alo = np.asarray(stats['action']['min'], dtype=np.float32); ahi = np.asarray(stats['action']['max'], dtype=np.float32)
    raw = np.load(SOURCE / 'reports/execution_comparison_20261112/raw_chunks.npy', mmap_mode='r')
    servo = CartesianServo(str(ROOT / 'fr3.urdf'))
    def fk(q):
        return np.stack([servo.fk(x[:7])[0] for x in q])
    def execute(chunks, method):
        if method == 'first_1':
            return np.clip(chunks[:, 0], alo, ahi)
        e = ACTTemporalEnsembler(.01, 100)
        return np.clip(np.stack([e.update(torch.from_numpy(np.array(x, copy=True))[None])[0].numpy().copy() for x in chunks]), alo, ahi)
    def pose_metrics(pred, reference):
        error = (fk(pred) - fk(reference)) * 1000
        return {'distance_mm': summarize(np.linalg.norm(error, axis=1)),
                'rmse_mm': float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
                'signed_xyz_mean_mm': error.mean(0).tolist(),
                'abs_xyz_median_mm': np.median(np.abs(error), axis=0).tolist(),
                'joint_mae_rad': np.abs(pred[:, :7] - reference[:, :7]).mean(0).tolist(),
                'joint_signed_mean_rad': (pred[:, :7] - reference[:, :7]).mean(0).tolist()}
    result = {'model': 'run02 EMA step 30000', 'temporal_coefficient': .01, 'held_out_episodes': [2, 4],
              'action_lead_frames': 10, 'same_frame_action_equals_state': True,
              'limitations': ['Predicted actions never generate the next observation.',
                             'Oracle joint replacements and q3 offsets are attribution probes, not deployable corrections.',
                             'Offline target error exists before any controller, network, or robot response.'],
              'episodes': {}, 'q3_episode_distribution': [], 'preprocessing_checks': []}
    plots = {}
    for ep, (a, b) in enumerate(zip(starts, ends)):
        a, b = int(a), int(b)
        result['q3_episode_distribution'].append({'episode': ep, 'split': 'held_out' if ep in [2, 4] else 'train',
            'mean': float(states[a:b, 2].mean()), 'std': float(states[a:b, 2].std()), 'min': float(states[a:b, 2].min()), 'max': float(states[a:b, 2].max())})
    result['q3_normalization_std_rad'] = float(np.asarray(stats['observation.state']['std'])[2])
    with no_network() as attempts:
        policy, _, size = load_act_gripper_policy(ROOT / 'runtime_config.json', ROOT / 'checkpoint', torch.device('cuda'), 'act_policy_ema.pth')
        assert policy.config.temporal_ensemble_coeff == .01 and policy.config.n_action_steps == 1
        for ep in [2, 4]:
            a, b = int(starts[ep]), int(ends[ep]); n = b - a - 10
            ref = actions[a + 10:b]; all_xyz = fk(actions[a:b])
            ev = gripper_events(ref[:, 7]); close = ev[0]['frame']; opened = ev[1]['frame']
            phases = {'approach': (0, close + 1), 'closed_carry': (close + 1, opened), 'after_release': (opened, n)}
            er = {'reference_events': ev, 'phases': phases, 'methods': {}, 'input_q3_offset_probes': []}
            for name in ['first_1', 'ensemble_0.01']:
                selected = execute(raw[a:a + n], name); xyz = fk(selected)
                detail = {'whole': pose_metrics(selected, ref), 'phase_metrics': {}, 'single_joint_oracle': []}
                for phase, (left, right) in phases.items():
                    detail['phase_metrics'][phase] = pose_metrics(selected[left:right], ref[left:right])
                # A common evaluation interval ensures each tested time shift sees the same observations.
                indices = np.arange(60, n - 60); alignment = []
                for lag in range(-60, 61):
                    diff = (xyz[indices] - all_xyz[indices + 10 + lag]) * 1000
                    alignment.append({'extra_lag_frames': lag, 'extra_lag_s': lag / 30,
                                      'rmse_mm': float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))),
                                      'distance_median_mm': float(np.median(np.linalg.norm(diff, axis=1)))})
                best = min(alignment, key=lambda x: x['rmse_mm'])
                detail['time_alignment_search'] = {'lag_range_frames': [-60, 60], 'original': alignment[60], 'best_rmse': best}
                plots[f'ep{ep}_{name}_lag_rmse'] = np.array([[x['extra_lag_s'], x['rmse_mm']] for x in alignment])
                for joint in range(7):
                    replaced = selected.copy(); replaced[:, joint] = ref[:, joint]
                    detail['single_joint_oracle'].append({'joint': joint + 1, 'whole': pose_metrics(replaced, ref),
                        'phase_distance_median_mm': {phase: pose_metrics(replaced[l:r], ref[l:r])['distance_mm']['p50'] for phase, (l, r) in phases.items()}})
                detail['at_reference_close_xyz_error_mm'] = ((xyz[close] - all_xyz[close + 10]) * 1000).tolist()
                er['methods'][name] = detail
                plots[f'ep{ep}_{name}_error_xyz_mm'] = (xyz - all_xyz[10:]) * 1000
            # Deliberately perturb only the model input q3; leave labels and all other inputs unchanged.
            # Best values selected using reference labels are diagnostic and must never be used as state calibration.
            for shift in [-.001, -.0005, 0., .0005, .001]:
                if shift == 0:
                    altered = np.array(raw[a:a + n], copy=True)
                else:
                    altered = np.empty((n, 100, 8), np.float32)
                    with torch.inference_mode():
                        for offset in range(0, n, 64):
                            ix = np.arange(a + offset, min(a + n, a + offset + 64))
                            state = torch.from_numpy(states[ix].copy()).cuda(); state[:, 2] += shift
                            obs = {'observation.state': state.clamp(lo, hi), **{f'observation.images.{c}': cache[f'observation.images.{c}'][ix].float().cuda() for c in ['cam_head', 'cam_wrist']}}
                            altered[offset:offset + len(ix)] = policy.predict_action_chunk(obs).cpu().numpy()
                selected = execute(altered, 'ensemble_0.01')
                er['input_q3_offset_probes'].append({'offset_rad': shift,
                    'clipped_input_fraction': float(np.mean((states[a:a+n, 2] + shift < float(lo[2])) | (states[a:a+n, 2] + shift > float(hi[2])))),
                    'whole': pose_metrics(selected, ref),
                    'phase_metrics': {p: pose_metrics(selected[l:r], ref[l:r]) for p, (l, r) in phases.items()},
                    'gripper_events': gripper_events(selected[:, 7])})
                print('Q3_PROBE', ep, shift, er['input_q3_offset_probes'][-1]['whole']['distance_mm']['p50'], flush=True)
            # Verify recorded video decoding/preprocessing for each held-out episode, both cameras, all phases.
            frames = sorted(set([0, close + 10, (close + opened) // 2 + 10, opened + 10, b - a - 1]))
            for camera in ['cam_head', 'cam_wrist']:
                differences = []
                file = data / f'videos/chunk-000/observation.images.{camera}/episode_{ep:06d}.mp4'
                with av.open(str(file)) as video:
                    for i, frame in enumerate(video.decode(video=0)):
                        if i in frames:
                            processed = preprocess_image(frame.to_ndarray(format='rgb24'), size, cfg['dataset']['cam_modes'][camera])[0]
                            differences.append(float((processed - cache[f'observation.images.{camera}'][a + i]).abs().max()))
                        if i >= frames[-1]:
                            break
                assert len(differences) == len(frames) and max(differences) < 1/255 + 1e-6
                result['preprocessing_checks'].append({'episode': ep, 'camera': camera, 'frames': frames, 'max_abs_difference': max(differences)})
            result['episodes'][str(ep)] = er
        result['ip_socket_attempts'] = attempts
    assert not attempts
    integrity = json.loads((ROOT / 'source_integrity.json').read_text())
    result['source_files_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in integrity.items())
    assert result['source_files_unchanged']
    (OUT / 'pose_bias_results.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT / 'pose_bias_traces.npz', **plots)
    print('POSE_DIAGNOSIS_DONE', OUT, flush=True)


if __name__ == '__main__':
    main()
