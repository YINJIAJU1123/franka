"""Launch the frozen original runner with explicit run02 and calibration arguments."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_command():
    return [sys.executable, '-u', str(ROOT / 'original_runner.py'),
        '--act_ckpt_dir', str(ROOT / 'checkpoint'), '--act_ckpt_file', 'act_policy_ema.pth',
        '--config', str(ROOT / 'runtime_config.json'), '--urdf', str(ROOT / 'fr3.urdf'),
        '--device', 'cuda', '--hz', '30', '--allow-motion', '1', '--dry-run-steps', '0',
        '--yin-host', '10.10.10.2', '--command-port', '8888',
        '--head-topic', '/camera/cam_head/color/image_raw',
        '--wrist-topic', '/camera/cam_wrist/color/image_raw',
        '--joint-topic', '/fr3/filtered_joint_states',
        '--pose-topic', '/franka_robot_state_broadcaster/current_pose',
        '--gripper-topic', '/franka_gripper/joint_states',
        '--auto-reset', '0', '--gripper-reset', 'hold', '--strict-reset', '0',
        '--gripper-close-threshold', '.0693', '--gripper-open-threshold', '.074',
        '--gripper-threshold', '.5', '--gripper-hysteresis', '.05',
        '--max-linear-velocity', '.12', '--max-angular-velocity', '.35',
        '--log-dir', str(ROOT / 'reports/real_temporal')]


def check():
    launch_manifest = json.loads((ROOT / 'real_launch_manifest.json').read_text())
    for relative, expected in launch_manifest['new_files'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError('Real-launch file changed: ' + relative)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for relative, details in manifest['files'].items():
        if digest(ROOT / relative) != details['sha256']:
            raise RuntimeError('Frozen baseline file changed: ' + relative)
    protected = json.loads((ROOT / 'protected_source_hashes.json').read_text())
    changed = [p for p, value in protected.items() if digest(p) != value]
    if changed:
        raise RuntimeError('Production source/artifact changed: ' + ', '.join(changed))
    if digest(ROOT / 'original_runner.py') != manifest['original_runner_expected_sha256']:
        raise RuntimeError('Original runner provenance mismatch')
    if digest(ROOT / 'checkpoint/act_policy_ema.pth') != manifest['ema_sha256']:
        raise RuntimeError('EMA checkpoint mismatch')
    training = json.loads((ROOT / 'checkpoint/act_config.json').read_text())
    runtime = json.loads((ROOT / 'runtime_config.json').read_text())
    expected = json.loads(json.dumps(training))
    expected['data_dir'] = runtime['data_dir']
    expected['policy']['temporal_ensemble_coeff'] = 0.01
    if runtime != expected:
        raise RuntimeError('Unexpected temporal runtime/model configuration changes')
    if runtime['policy'].get('mask_gripper_state') is not True:
        raise RuntimeError('run02 requires its trained gripper-input mask')
    if os.environ.get('N_ACTION_STEPS') != '1':
        raise RuntimeError('Original-loop baseline requires N_ACTION_STEPS=1')
    if runtime['policy']['temporal_ensemble_coeff'] != 0.01:
        raise RuntimeError('This independent version requires native temporal ensemble k=0.01')
    previous = json.loads((ROOT / 'reports/offline_results.json').read_text())
    if not previous['software_checks_passed'] or not previous['recorded_event_check_passed']:
        raise RuntimeError('Saved-data offline checks have not passed')
    spec = importlib.util.spec_from_file_location('launch_check_original', ROOT / 'original_runner.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    command = build_command()
    args = runner._build_parser().parse_args(command[3:])
    assert args.allow_motion == 1 and args.dry_run_steps == 0 and args.auto_reset == 0
    assert args.gripper_reset == 'hold' and args.command_port == 8888
    assert args.gripper_close_threshold == .0693 and args.gripper_open_threshold == .074
    for name in ['act_gripper_policy', 'franka_gripper_transport', 'franka_gripper_control']:
        if not Path(inspect.getfile(sys.modules[name])).resolve().is_relative_to(ROOT):
            raise RuntimeError('Mixed source import: ' + name)
    result = {'passed': True, 'mode': 'static_no_robot', 'original_runner_unchanged': True,
        'original_runner_sha256': digest(ROOT / 'original_runner.py'), 'ema_sha256': manifest['ema_sha256'],
        'source_files_unchanged': len(protected), 'command': command,
        'n_action_steps': 1, 'temporal_ensemble_coeff': 0.01, 'training_action_lead_frames': 10, 'extra_runtime_offset_frames': 0,
        'auto_reset': False, 'gripper_reset': 'hold', 'command_target': '10.10.10.2:8888',
        'linear_velocity_component_limit_m_s': .12, 'angular_velocity_component_limit_rad_s': .35,
        'robot_connection_created': False, 'ros_graph_created': False,
        'onsite_keyboard_required': True,
        'original_behavior_before_B': 'Bridge probe, then zero arm delta packets carrying current gripper binary command',
        'known_limitation': 'run02 q3 sensitivity persists; this launcher does not correct predictions'}
    (ROOT / 'reports/real_launch_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print('CHECK PASSED: original code + run02 EMA + calibrated arguments; no robot connection.', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='check', choices=['check', 'run'])
    mode = parser.parse_args().mode
    result = check()
    if mode == 'check':
        print('To run later from the lixin Linux desktop terminal: bash ' + str(ROOT / 'run_real.sh') + ' run')
        return 0
    if not os.environ.get('DISPLAY') or not sys.stdin.isatty():
        raise RuntimeError('Run mode needs an interactive terminal on the lixin Linux desktop for B/C keys.')
    print('Starting ORIGINAL Cartesian ACT loop: EMA step 30000, native temporal k=0.01, 30 Hz, automatic reset OFF.', flush=True)
    print('Before B: original code probes the bridge and sends zero arm delta + current gripper command.', flush=True)
    print('Manually open the gripper and restore the demonstrated joint start before launching.', flush=True)
    print('B starts/ends an episode; C exits. Maximum component speeds: .12 m/s and .35 rad/s.', flush=True)
    os.execv(sys.executable, result['command'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
