"""Offline source, configuration and asset checks using only the standard library."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EMA_SHA256 = 'd9319c1b1e6f329f8a31d739cfd0593069622b949f3544a7b2bc0231dc39c852'
RUNNER_SHA256 = '5deea3cf877b51dce0380e1064ac86e78de5888c9e2bbd3efef0e9efd558167c'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside repository/snapshot: {name}')
    return path


def verify(require_runtime_assets=False, check_git_index=False):
    manifest = json.loads((ROOT / 'reproduction_manifest.json').read_text())
    checked = 0
    for name, expected in manifest['verified_source_files'].items():
        path = contained_path(ROOT, name)
        if not path.is_file() or sha(path) != expected['sha256']:
            raise ValueError(f'Frozen source mismatch: {name}')
        if check_git_index:
            result = subprocess.run(['git', 'show', ':'+name], cwd=ROOT, capture_output=True)
            if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected['sha256']:
                raise ValueError(f'Frozen source missing or changed in Git index: {name}')
        checked += 1
    cfg = json.loads((ROOT / 'runtime_lixin/runtime_config.json').read_text())
    assert cfg['policy']['temporal_ensemble_coeff'] == .01
    assert cfg['policy']['chunk_size'] == 100 and cfg['policy']['n_action_steps'] == 1
    assert cfg['policy']['mask_gripper_state'] is True
    assert cfg['dataset']['action_offset_frames'] == 10
    assert manifest['ema_sha256'] == EMA_SHA256 and manifest['model_step'] == 30000
    assert sha(ROOT / 'runtime_lixin/original_runner.py') == RUNNER_SHA256
    selection = json.loads((ROOT / 'runtime_lixin/recorded_metadata/episode_selection.json').read_text())
    assert selection['excluded_source_episodes'] == [6] and len(selection['mapping']) == 14
    assets = json.loads((ROOT / 'external_assets_manifest.json').read_text())['assets']
    external_present = 0; required_missing = []
    for asset in assets:
        path = contained_path(ROOT, asset['path'])
        if path.exists():
            if sha(path) != asset['sha256']:
                raise ValueError(f'External asset hash mismatch: {asset["path"]}')
            external_present += 1
        elif asset['required_by_original_runtime_manifest']:
            required_missing.append(asset['path'])
    if require_runtime_assets and required_missing:
        raise ValueError('Missing runtime assets: ' + ', '.join(required_missing))
    parsed = 0
    secret_patterns = [
        re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}'),
        re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),
        re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
        re.compile(r'(?i)(?:password|passwd|api_key)\s*=\s*[\"\x27](?!\$|<|example|placeholder)[A-Za-z0-9+/_=.-]{6,}[\"\x27]'),
    ]
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.relative_to(ROOT).parts or '__pycache__' in path.parts:
            continue
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            parsed += 1
        if path.suffix in ['.py', '.cpp', '.hpp', '.h', '.sh', '.json', '.jsonl', '.md', '.txt', '.yaml', '.yml']:
            text = path.read_text(encoding='utf-8')
            if any(pattern.search(text) for pattern in secret_patterns):
                raise ValueError(f'Possible credential material in {path.relative_to(ROOT)}')
    return {'passed': True, 'frozen_source_files_checked': checked, 'python_files_syntax_checked': parsed,
            'external_assets_present': external_present, 'required_runtime_assets_missing': len(required_missing),
            'model_step': 30000, 'temporal_ensemble_coeff': .01,
            'hardware_connections_created': 0, 'ROS_nodes_started': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-runtime-assets', action='store_true')
    parser.add_argument('--check-git-index', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.require_runtime_assets, args.check_git_index), indent=2))


if __name__ == '__main__':
    main()
