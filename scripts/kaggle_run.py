#!/usr/bin/env python3
"""Submit exactly one explicit run. Git commits alone never trigger Kaggle."""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrypoint', default='kaggle/smoke_test.py')
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--repository', default='nikitapr1/pinn-sirius')
    parser.add_argument('--gpu', action='store_true', help='Explicitly opt into GPU quota consumption')
    parser.add_argument('--requirements', help='Optional requirements file; built-in packages are preferred')
    parser.add_argument('--wheelhouse', help='Optional /kaggle/input/... directory containing prepared wheels')
    parser.add_argument('--dataset', action='append', default=[], help='Kaggle dataset owner/slug; repeatable')
    parser.add_argument('--checkpoint-kernel', action='append', default=[], help='Previous owner/kernel whose outputs to mount')
    parser.add_argument('--max-training-minutes', type=float, default=60, help='Stop training subprocess after this duration')
    parser.add_argument('--wait', action='store_true', help='Wait and download outputs locally')
    parser.add_argument('--timeout-minutes', type=float, default=30, help='Client wait timeout, NOT a remote training limit')
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if git('status', '--porcelain', '--untracked-files=normal'):
        parser.error('Commit and push changes before submitting; working tree must be clean.')
    if args.max_training_minutes <= 0 or args.timeout_minutes <= 0:
        parser.error('Time limits must be positive')
    sha = git('rev-parse', args.revision + '^{commit}')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', args.repository):
        parser.error('Invalid repository owner/name')
    for name in [args.entrypoint, args.requirements]:
        if name:
            if Path(name).is_absolute() or '..' in Path(name).parts:
                parser.error('Entrypoint and requirements must be repository-relative paths')
            git('cat-file', '-e', sha + ':' + name)
    # Validate that the public source is already available before reserving Kaggle compute.
    import urllib.request
    with urllib.request.urlopen(f'https://api.github.com/repos/{args.repository}/commits/{sha}', timeout=30) as r:
        if json.load(r)['sha'] != sha:
            raise RuntimeError('Remote commit mismatch')
    from kaggle import api
    owner = api.get_config_value(api.CONFIG_NAME_USER)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    slug = f'pinn-{"gpu" if args.gpu else "cpu"}-{sha[:8]}-{stamp}'
    ref = f'{owner}/{slug}'
    folder = ROOT / '.kaggle-runs' / slug
    folder.mkdir(parents=True)
    config = dict(repository=args.repository, commit=sha, entrypoint=args.entrypoint,
                  requirements=args.requirements, wheelhouse=args.wheelhouse,
                  args=args.training_args[1:] if args.training_args[:1] == ['--'] else args.training_args,
                  gpu=args.gpu, datasets=args.dataset, checkpoint_kernels=args.checkpoint_kernel,
                  max_training_minutes=args.max_training_minutes)
    bootstrap = (ROOT / 'kaggle/bootstrap.py').read_text()
    (folder / 'main.py').write_text('RUN_CONFIG = ' + repr(config) + '\n' + bootstrap)
    metadata = dict(id=ref, title=slug, code_file='main.py', language='python', kernel_type='script',
                    is_private=True, enable_gpu=args.gpu, enable_internet=True,
                    dataset_sources=args.dataset, competition_sources=[], kernel_sources=args.checkpoint_kernel)
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    result = api.kernels_push(str(folder))
    print(result)
    print('https://www.kaggle.com/code/' + ref, flush=True)
    (folder / 'submission.json').write_text(json.dumps(dict(kernel=ref, **config), indent=2))
    if args.wait:
        import time
        deadline = time.monotonic() + args.timeout_minutes * 60
        while time.monotonic() < deadline:
            response = api.kernels_status(ref)
            status = str(getattr(response, 'status', response)).upper()
            print(status, flush=True)
            if any(word in status for word in ['COMPLETE', 'ERROR', 'CANCEL', 'FAILED']):
                token = None
                while True:
                    _, token = api.kernels_output(ref, path=str(folder / 'output'), page_token=token)
                    if not token:
                        break
                if 'COMPLETE' not in status:
                    raise RuntimeError('Kaggle run failed: ' + status)
                print('Outputs:', folder / 'output')
                return
            time.sleep(15)
        sys.exit('Wait timed out. Remote job may still be running; check Kaggle before submitting another.')

if __name__ == '__main__':
    main()
