"""Executed in Kaggle; RUN_CONFIG is prepended by scripts/kaggle_run.py."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(config):
    output = Path('/kaggle/working')
    output.mkdir(parents=True, exist_ok=True)
    source = Path('/kaggle/temp/project')
    source.mkdir(parents=True, exist_ok=True)
    record = dict(config, started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='running')
    start = time.monotonic()
    def run(args):
        subprocess.run(args, cwd=source, check=True)
    try:
        run(['git', 'init', '-q'])
        run(['git', 'remote', 'add', 'origin', 'https://github.com/' + config['repository'] + '.git'])
        run(['git', 'fetch', '-q', '--depth=1', 'origin', config['commit']])
        run(['git', 'checkout', '-q', '--detach', 'FETCH_HEAD'])
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
        if actual != config['commit']:
            raise RuntimeError('Downloaded commit does not match requested commit')
        os.environ['OUTPUT_DIR'] = str(output)
        os.environ['INPUT_DIR'] = '/kaggle/input'
        os.environ['RUN_COMMIT'] = actual
        os.environ['PYTHONUNBUFFERED'] = '1'
        if config['requirements']:
            command = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-r', config['requirements']]
            if config['wheelhouse']:
                command += ['--no-index', '--find-links', config['wheelhouse']]
            run(command)
        record['setup_seconds'] = round(time.monotonic() - start, 2)
        print('Running commit:', actual, flush=True)
        subprocess.run([sys.executable, config['entrypoint'], *config['args']], cwd=source,
                       check=True, timeout=config['max_training_minutes'] * 60)
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        record['elapsed_seconds'] = round(time.monotonic() - start, 2)
        (output / 'run-manifest.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main(RUN_CONFIG)
