"""Restore hash-matched, Git-ignored assets from an existing trusted snapshot."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from verify_repository import ROOT, contained_path, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--include-logs', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source_root = args.snapshot.resolve()
    assets = json.loads((ROOT / 'external_assets_manifest.json').read_text())['assets']
    planned = []; already = 0
    # Validate every selected source/destination before starting to copy.
    for asset in assets:
        if not args.include_logs and not asset['required_by_original_runtime_manifest']:
            continue
        source = contained_path(source_root, asset['snapshot_relative_path'])
        destination = contained_path(ROOT, asset['path'])
        if not source.is_file() or sha(source) != asset['sha256']:
            raise ValueError(f'Trusted source asset missing/mismatched: {asset["path"]}')
        if destination.exists():
            if sha(destination) != asset['sha256']:
                raise ValueError(f'Refusing to overwrite different content: {asset["path"]}')
            already += 1
        else:
            planned.append((source, destination, asset['sha256']))
    if not args.dry_run:
        for source, destination, expected in planned:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.restore-', delete=False) as f:
                temporary = Path(f.name).resolve()
                assert temporary.is_relative_to(ROOT.resolve())
                with source.open('rb') as original:
                    shutil.copyfileobj(original, f, length=2**20)
            try:
                assert sha(temporary) == expected
                if destination.exists():
                    raise ValueError(f'Destination appeared during restore: {destination.relative_to(ROOT)}')
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
    print(json.dumps({'dry_run': args.dry_run, 'files_to_restore': len(planned),
                      'already_matching': already, 'hardware_connections_created': 0}, indent=2))


if __name__ == '__main__':
    main()
