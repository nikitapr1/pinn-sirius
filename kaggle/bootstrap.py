"""Executed in Kaggle; RUN_CONFIG is prepended by scripts/kaggle_run.py."""
import base64
import io
import zipfile
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
        archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE)))
        prefix = archive.namelist()[0].split('/')[0] + '/'
        for item in archive.infolist():
            relative = item.filename.removeprefix(prefix)
            target = (source / relative).resolve()
            if not target.is_relative_to(source.resolve()):
                raise RuntimeError('Unsafe path in source archive')
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
        actual = config['commit']
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
