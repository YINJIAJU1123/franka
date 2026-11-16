"""Inspect frozen code and run saved-data inference without creating ROS nodes."""
from pathlib import Path
import ast
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import json
import os
import socket
import sys
import threading
import time
from types import SimpleNamespace

import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from act_gripper_policy import load_act_gripper_policy, preprocess_image
from buffer_cache_utils import load_standard_buffer_cache
from franka_gripper_control import CartesianServo
from franka_gripper_transport import FrankaGripperTransport


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def block_python_network():
    real = socket.socket
    attempts = []

    def denied(family=socket.AF_INET, *args, **kwargs):
        if family in (socket.AF_INET, socket.AF_INET6):
            attempts.append(str(family))
            raise AssertionError('Offline audit attempted an IP socket')
        return real(family, *args, **kwargs)

    socket.socket = denied
    try:
        yield attempts
    finally:
        socket.socket = real


def main():
    output = ROOT / 'reports'
    output.mkdir(exist_ok=True)
    result = {'software_checks_passed': False, 'physical_motion_performed': False,
              'ros_graph_created': False, 'method': 'Recorded observations, unchanged original runner, native temporal ensemble k=0.01',
              'limitations': ['No live camera/state validation', 'No actuator/controller test',
                              'Recorded-observation replay is not autonomous rollout',
                              'Only two episodes are held out; they were previously used to choose EMA']}
    protected = json.loads((ROOT / 'protected_source_hashes.json').read_text())
    protected_bad = [p for p, digest in protected.items() if sha(p) != digest]
    assert not protected_bad, protected_bad
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    assert all(sha(ROOT / p) == v['sha256'] for p, v in manifest['files'].items())
    assert sha(ROOT / 'original_runner.py') == manifest['original_runner_expected_sha256']
    assert sha(ROOT / 'checkpoint/act_policy_ema.pth') == manifest['ema_sha256']
    training = json.loads((ROOT / 'checkpoint/act_config.json').read_text())
    runtime = json.loads((ROOT / 'runtime_config.json').read_text())
    differences = [k for k in set(training) | set(runtime) if training.get(k) != runtime.get(k)]
    assert set(differences) == {'data_dir', 'policy'}
    expected = json.loads(json.dumps(training)); expected['data_dir'] = runtime['data_dir']
    expected['policy']['temporal_ensemble_coeff'] = 0.01
    assert runtime == expected
    tree = ast.parse((ROOT / 'original_runner.py').read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(x in imports for x in ['franka_action_execution', 'franka_joint_client'])
    assert 'select_action' in {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    os.environ['N_ACTION_STEPS'] = '1'
    torch.set_num_threads(8)
    policy, cfg, size = load_act_gripper_policy(ROOT / 'runtime_config.json', ROOT / 'checkpoint',
                                               torch.device('cuda'), 'act_policy_ema.pth')
    assert policy.config.n_action_steps == 1 and policy.config.temporal_ensemble_coeff == 0.01
    assert policy.config.mask_gripper_state and cfg['dataset']['action_offset_frames'] == 10
    import act_gripper_policy, franka_gripper_control, franka_gripper_transport, image_preprocessing
    modules = [act_gripper_policy, franka_gripper_control, franka_gripper_transport, image_preprocessing,
               sys.modules[type(policy).__module__], sys.modules[type(policy.model).__module__]]
    module_paths = [str(Path(inspect.getfile(module)).resolve()) for module in modules]
    assert all(Path(p).is_relative_to(ROOT) for p in module_paths), module_paths
    result['isolation'] = {'original_runner_byte_identical': True, 'model_sha256': manifest['ema_sha256'],
        'runtime_config_changed_keys': differences, 'module_paths': module_paths,
        'protected_source_files_unchanged': len(protected), 'no_joint_client_or_action_executor': True}
    print('PASS frozen artifacts, original loop and isolated imports', flush=True)

    callback = SimpleNamespace(_joint_map=FrankaGripperTransport._joint_map,
        _lock=threading.Lock(), _gripper_binary=None, _gripper_close_threshold=.0693,
        _gripper_open_threshold=.074, _callback_counts={'gripper': 0})
    widths = [.079744, .058849, .071, .079744]
    bits = []
    for width in widths:
        msg = SimpleNamespace(name=['fr3_finger_joint1', 'fr3_finger_joint2'], position=[width / 2] * 2)
        FrankaGripperTransport._gripper_cb(callback, msg)
        bits.append(callback._gripper_binary)
    assert bits == [0., 1., 1., 0.], bits
    result['gripper_calibration'] = {'widths_m': widths, 'binary_sequence': bits, 'passed': True}

    data = Path(json.loads((ROOT / 'offline_sources.json').read_text())['dataset'])
    selection = json.loads((ROOT / 'recorded_metadata/episode_selection.json').read_text())
    assert selection['excluded_source_episodes'] == [6]
    assert len(selection['mapping']) == 14
    assert [x['source_episode'] for x in selection['mapping']] == list(range(6)) + list(range(7, 15))
    entries = [json.loads(x) for x in (ROOT / 'recorded_metadata/meta/episodes.jsonl').read_text().splitlines() if x]
    ends = np.cumsum([x['length'] for x in entries])
    starts = np.r_[0, ends[:-1]]
    assert ends[-1] == 24054
    cache = load_standard_buffer_cache(data / 'buffer_cache.pt')
    states = cache['state'].float()
    actions = cache['action'].float()
    stats = torch.load(ROOT / 'checkpoint/dataset_stats.pth', map_location='cpu', weights_only=False)
    lo, hi = [torch.as_tensor(stats['observation.state'][x]).float().cuda() for x in ('min', 'max')]
    alo, ahi = [torch.as_tensor(stats['action'][x]).float().cuda() for x in ('min', 'max')]
    cameras = ['cam_head', 'cam_wrist']

    def batch(indices):
        state = states[indices].cuda()
        return {'observation.state': state.clamp(lo, hi),
                **{f'observation.images.{c}': cache[f'observation.images.{c}'][indices].float().cuda() for c in cameras}}

    # Validate RGB decode + real preprocessing against cached training inputs.
    ep = 2
    local_frames = [0, 100, 210, 216, 226, 300, 800, 1290, 1300, 1310, 1500, int(ends[ep] - starts[ep] - 1)]
    frames = {c: {} for c in cameras}
    preprocessing = {}
    for camera in cameras:
        file = data / f'videos/chunk-000/observation.images.{camera}/episode_{ep:06d}.mp4'
        with av.open(str(file)) as video:
            for i, frame in enumerate(video.decode(video=0)):
                if i in local_frames:
                    raw = frame.to_ndarray(format='rgb24')
                    frames[camera][i] = raw
                if i >= max(local_frames):
                    break
        differences = []
        for i in local_frames:
            processed = preprocess_image(frames[camera][i], size, cfg['dataset']['cam_modes'][camera])[0]
            differences.append(float((processed - cache[f'observation.images.{camera}'][int(starts[ep]) + i]).abs().max()))
        assert max(differences) < 1 / 255 + 1e-6, differences
        preprocessing[camera] = {'raw_rgb_shape': list(frames[camera][0].shape),
            'mode': cfg['dataset']['cam_modes'][camera], 'output_shape': [3, size, size],
            'max_abs_difference_from_training_cache': max(differences)}
    result['camera_preprocessing'] = preprocessing
    print('PASS raw recorded RGB preprocessing and gripper calibration', flush=True)

    # Use the actual original main loop with the actual EMA, preprocessing and FK.
    # Only observation transport, clock and ROS lifecycle are replaced; all sends fail.
    spec = importlib.util.spec_from_file_location('baseline_original_runner', ROOT / 'original_runner.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    servo = CartesianServo(str(ROOT / 'fr3.urdf'))
    observations = []
    for i in local_frames:
        state = states[int(starts[ep]) + i].numpy()
        pos, quat = servo.fk(state[:7])
        observations.append(SimpleNamespace(arm_joints=state[:7], gripper_binary=float(state[7]),
            gripper_width=.058849 if state[7] >= .5 else .079744,
            ee_pose=np.r_[pos, quat], images={c: frames[c][i] for c in cameras},
            state_age_s=0., image_age_s={c: 0. for c in cameras}, sequence=i))
    send_attempts = []
    case = 'normal'

    class SavedTransport:
        def __init__(self, **kwargs):
            assert kwargs['enable_commands'] is False
            assert kwargs['gripper_close_threshold'] == .0693 and kwargs['gripper_open_threshold'] == .074
            self.index = 0
        def wait_ready(self, **kwargs):
            return observations[0]
        def snapshot(self):
            value = observations[min(self.index, len(observations) - 1)]
            self.index += 1
            if case == 'stale':
                return SimpleNamespace(**{**vars(value), 'state_age_s': 1.})
            return value
        def send_delta(self, *args, **kwargs):
            send_attempts.append('delta'); raise AssertionError('No sends allowed')
        def send_zero(self, *args, **kwargs):
            send_attempts.append('zero'); raise AssertionError('No sends allowed')
        def probe_command_receiver(self, *args, **kwargs):
            send_attempts.append('probe'); raise AssertionError('No probes allowed')
        def destroy_node(self):
            pass

    class Clock:
        value = 0.
        def monotonic(self):
            return self.value
        def sleep(self, delay):
            self.value += max(.001, delay)

    runner.FrankaGripperTransport = SavedTransport
    runner.rclpy = SimpleNamespace(init=lambda: None, shutdown=lambda: None, ok=lambda: True)
    runner.time = Clock()
    runner.load_act_gripper_policy = lambda *args, **kwargs: (policy, cfg, size)
    args = ['original_runner', '--act_ckpt_dir', str(ROOT / 'checkpoint'), '--act_ckpt_file', 'act_policy_ema.pth',
            '--config', str(ROOT / 'runtime_config.json'), '--urdf', str(ROOT / 'fr3.urdf'),
            '--allow-motion', '0', '--dry-run-steps', str(len(observations)), '--auto-reset', '0',
            '--gripper-reset', 'hold', '--gripper-close-threshold', '.0693', '--gripper-open-threshold', '.074']
    with block_python_network() as network_attempts:
        sys.argv = args + ['--log-dir', str(output / 'original_loop')]
        runner.main()
        case = 'stale'
        sys.argv = args + ['--log-dir', str(output / 'stale_observation')]
        try:
            runner.main()
            raise AssertionError('Stale observations were accepted')
        except RuntimeError as exc:
            assert 'stale' in str(exc)
    assert not send_attempts and not network_attempts
    loop_log = next((output / 'original_loop').glob('*.csv'))
    rows = list(csv.DictReader(loop_log.open()))
    assert len(rows) == len(observations) and all(x['command_sent'] == '0' for x in rows)
    expected = []
    policy.reset()
    with torch.inference_mode():
        for i in local_frames:
            expected.append(policy.select_action(batch([int(starts[ep]) + i]))[0].clamp(alo, ahi).cpu().numpy())
    loop_actions = np.stack([np.fromstring(x['q_target'].strip('[]'), sep=' ') for x in rows])
    target_difference = float(np.abs(loop_actions - np.stack(expected)[:, :7]).max())
    assert target_difference < 2e-5, target_difference
    result['original_loop'] = {'raw_observation_steps': len(rows), 'real_ema_and_fk': True,
        'command_send_attempts': len(send_attempts), 'ip_socket_attempts': len(network_attempts),
        'stale_observation_rejected': True, 'max_joint_difference_from_direct_select_action_rad': target_difference}
    print('PASS original main loop with real model/FK, no IP sockets or command calls, stale observation rejected', flush=True)

    first = np.full((len(states), 8), np.nan, dtype=np.float32)
    raw_chunks = np.full((len(states), 100, 8), np.nan, dtype=np.float32)
    lead = int(cfg['dataset']['action_offset_frames'])
    indices = np.concatenate([np.arange(a, b - lead) for a, b in zip(starts, ends)])
    timings = []
    with torch.inference_mode(), block_python_network() as network_attempts:
        for offset in range(0, len(indices), 64):
            ix = indices[offset:offset + 64]
            t = time.perf_counter()
            pred = policy.predict_action_chunk(batch(ix))
            assert tuple(pred.shape) == (len(ix), 100, 8) and bool(torch.isfinite(pred).all())
            raw_chunks[ix] = pred.cpu().numpy()
            timings.append((time.perf_counter() - t) * 1000)
            if offset % 4096 == 0:
                print(f'OFFLINE_REPLAY {offset}/{len(indices)}', flush=True)
    assert not network_attempts
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
    for a, b in zip(starts, ends):
        ensemble = ACTTemporalEnsembler(0.01, 100)
        for i in range(int(a), int(b) - lead):
            value = ensemble.update(torch.from_numpy(raw_chunks[i]).unsqueeze(0))[0].numpy()
            first[i] = np.clip(value, alo.cpu().numpy(), ahi.cpu().numpy())
    event_rows = []
    correct = total = 0
    arm_error_sum = 0.
    for ep, (a, b) in enumerate(zip(starts, ends)):
        a, b = int(a), int(b)
        ix = np.arange(a, b - lead)
        target = actions[ix + lead].numpy()
        raw = first[ix, 7]
        current = False
        binary = []
        for value in raw:
            current = bool(value > .45) if current else bool(value >= .55)
            binary.append(current)
        binary = np.asarray(binary)
        target_binary = target[:, 7] >= .5
        pe = np.flatnonzero(binary != np.r_[False, binary[:-1]])
        te = np.flatnonzero(target_binary != np.r_[False, target_binary[:-1]])
        types_pred = ['close' if binary[x] else 'open' for x in pe]
        types_target = ['close' if target_binary[x] else 'open' for x in te]
        errors = ((pe - te) / 30).tolist() if types_pred == types_target else None
        event_rows.append({'training_episode': ep, 'source_episode': selection['mapping'][ep]['source_episode'],
            'predicted_types': types_pred, 'predicted_frames': pe.tolist(), 'expected_frames': te.tolist(),
            'timing_error_s': errors, 'within_one_second': errors is not None and len(errors) == 2 and max(map(abs, errors)) <= 1})
        correct += int((binary == target_binary).sum()); total += len(ix)
        arm_error_sum += float(np.abs(first[ix, :7] - target[:, :7]).sum())
    result['recorded_replay'] = {'dataset_episodes': 14, 'dataset_total_frames': len(states),
        'evaluated_frames': total, 'excluded_source_episode': 6, 'action_lead_frames': lead,
        'runtime_action_index': 'native_temporal_ensemble', 'gripper_accuracy': correct / total, 'arm_mae_rad': arm_error_sum / (total * 7),
        'episodes_with_one_close_one_open_within_1s': sum(x['within_one_second'] for x in event_rows),
        'max_abs_event_timing_error_s': max(abs(e) for x in event_rows for e in (x['timing_error_s'] or [])),
        'episodes': event_rows, 'median_64_observation_batch_ms': float(np.median(timings))}
    with (output / 'event_timing.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(event_rows[0])); writer.writeheader(); writer.writerows(event_rows)
    np.savez_compressed(output / 'first_action_predictions.npz', indices=indices, predictions=first[indices])
    print('RECORDED_EVENTS', result['recorded_replay']['episodes_with_one_close_one_open_within_1s'], '/14', flush=True)

    saved = np.load(ROOT / 'fixtures/saved_near_grasp_inputs.npz')
    reference = states[saved['indices']].clone()
    live = torch.tensor(saved['live_state']).float().repeat(14, 1)
    corrected = live.clone(); corrected[:, 2] = reference[:, 2]
    images = {f'observation.images.{c}': torch.tensor(saved[c]).unsqueeze(0).repeat(14, 1, 1, 1).cuda() for c in cameras}
    predictions = {}
    with torch.inference_mode(), block_python_network() as network_attempts:
        for label, state in [('recorded', reference), ('live', live), ('live_q3_corrected', corrected)]:
            predictions[label] = policy.predict_action_chunk({**images, 'observation.state': state.cuda().clamp(lo, hi)}).cpu().numpy()
        flipped = reference.clone(); flipped[:, 7] = 1 - flipped[:, 7]
        flip_pred = policy.predict_action_chunk({**images, 'observation.state': flipped.cuda().clamp(lo, hi)}).cpu().numpy()
    assert not network_attempts
    mask_difference = float(np.abs(flip_pred - predictions['recorded']).max())
    assert mask_difference == 0., mask_difference
    live_pos = servo.fk(saved['live_state'][:7])[0]
    distances = {name: float(np.mean([np.linalg.norm(servo.fk(x[0, :7])[0] - live_pos) for x in values]) * 1000)
                 for name, values in predictions.items()}
    result['model_input_contract'] = {'masked_gripper_flip_max_output_difference': mask_difference, 'passed': True}
    result['known_model_limitation'] = {'saved_near_grasp_input_first_target_distance_mm': distances,
        'q3_sensitivity_still_present': distances['live'] > 80 and distances['live_q3_corrected'] < 20,
        'physical_state_was_not_changed': True, 'not_a_state_spoofing_fix': True}
    assert not [p for p, digest in protected.items() if sha(p) != digest]
    assert all(sha(ROOT / p) == v['sha256'] for p, v in manifest['files'].items())
    result['software_checks_passed'] = True
    result['recorded_event_check_passed'] = all(x['within_one_second'] for x in event_rows)
    result['autonomous_task_success_established'] = False
    (output / 'offline_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('OFFLINE_COMPLETE', json.dumps({k: result[k] for k in ['software_checks_passed', 'recorded_event_check_passed',
        'known_model_limitation', 'autonomous_task_success_established']}), flush=True)


if __name__ == '__main__':
    main()
