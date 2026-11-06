"""Command line entry point. Run from any working directory."""
import argparse
import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))

from src.experiments import run


def main():
    parser = argparse.ArgumentParser(description='BTL01 PyTorch Gaussian approximation')
    parser.add_argument('experiment', choices=['A', 'B', 'C', 'all'], nargs='?', default='all')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/official.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    output = args.output.resolve()
    data_dir = args.data_dir or (ROOT / 'data' if output == ROOT / 'results' else output / 'data')
    # Paths in manifests are repository-relative and stable even when invoked elsewhere.
    os.chdir(ROOT)
    run(config, output, data_dir, args.experiment)


if __name__ == '__main__':
    main()
