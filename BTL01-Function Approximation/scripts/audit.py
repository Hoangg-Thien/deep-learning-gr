"""Independently re-evaluate every checkpoint and audit the actual deliverables."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from pypdf import PdfReader
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_csv
from src.experiments import experiment_specs, read_csv, select_model, summarize, write_json
from src.metrics import regression_metrics
from src.training import load_checkpoint, predict, seed_everything


def resolve_data_directory(output, manifest, project_root=ROOT):
    output, project_root = Path(output).resolve(), Path(project_root).resolve()
    local = project_root / 'data' if output == project_root / 'results' else output / 'data'
    if local.is_dir():
        return local
    return Path(manifest['data_directory'])


def expected_runs(config):
    runs = {}
    for spec in experiment_specs(config):
        for seed in config['seeds']:
            run_id = f"{spec['experiment']}_{spec['value']:04d}_seed{seed}"
            params = dict(config['training'], hidden_sizes=spec['hidden_sizes'], seed=seed,
                          data_seed=config['data_seed'], n_samples=spec['n_samples'])
            runs[run_id] = dict(spec=spec, config=params)
    return runs


def validate_run_config(run_id, record, saved_config, checkpoint_config, expected):
    planned = expected[run_id]
    assert saved_config == checkpoint_config == planned['config'], f'{run_id}: config differs from experiment plan'
    spec = planned['spec']
    for key in ['experiment', 'value', 'n_samples']:
        assert record[key] == spec[key], (run_id, key)
    assert record['width'] == spec['hidden_sizes'][0]
    assert record['depth'] == len(spec['hidden_sizes'])
    assert record['seed'] == planned['config']['seed']


def audit(output, documents=True):
    output = Path(output)
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    seed_everything(config['seeds'][0], config['training'].get('threads', 1))
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete'
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    assert digest == manifest['config_sha256']
    for path, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    data_dir = resolve_data_directory(output, manifest)
    splits = {name: load_csv(data_dir / f'{name}.csv') for name in ['train', 'validation', 'test']}
    for name, (x, _) in splits.items():
        assert len(x) == config['n_' + name]
        assert hashlib.sha256((data_dir / f'{name}.csv').read_bytes()).hexdigest() == manifest['dataset_sha256'][name]
    assert len(splits['test'][0]) == 100
    sets = [set(map(tuple, x)) for x, _ in splits.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    specifications = experiment_specs(config)
    expected = expected_runs(config)
    rows = read_csv(output / 'tables/all_runs.csv')
    assert len(rows) == len(expected) and {r['run_id'] for r in rows} == set(expected)
    records = []
    for row in rows:
        run_id = row['run_id']
        record = dict(row)
        for key in ['value', 'seed', 'n_samples', 'width', 'depth', 'parameters', 'best_epoch', 'epochs_ran']:
            record[key] = int(row[key])
        record['training_seconds'] = float(row['training_seconds'])
        model, metadata = load_checkpoint(output / f'checkpoints/{run_id}.pt')
        assert model.hidden_sizes == [record['width']] * record['depth']
        assert metadata['config']['seed'] == record['seed']
        saved_config = json.loads((output / f'logs/{run_id}_config.json').read_text())
        validate_run_config(run_id, record, saved_config, metadata['config'], expected)
        assert sum(p.numel() for p in model.parameters()) == record['parameters']
        history = read_csv(output / f'logs/{run_id}.csv')
        assert len(history) == record['epochs_ran'] <= config['training']['epochs']
        assert [int(h['epoch']) for h in history] == list(range(1, len(history) + 1))
        values = np.array([float(h['validation_loss']) for h in history])
        assert int(values.argmin()) + 1 == record['best_epoch'] == metadata['best_epoch']
        assert values.min() == metadata['best_validation_loss']
        assert (output / f'figures/loss/{run_id}.png').stat().st_size > 1000
        for name, (x, y) in splits.items():
            if name == 'train':
                x, y = x[:record['n_samples']], y[:record['n_samples']]
            actual = regression_metrics(y, predict(model, x))
            for metric, value in actual.items():
                assert np.isclose(value, float(row[f'{name}_{metric}']), rtol=1e-10, atol=1e-15), (run_id, name, metric)
                record[f'{name}_{metric}'] = value
        assert np.isclose(record['validation_mse'], values.min(), rtol=1e-5, atol=1e-12)
        predictions = read_csv(output / f'tables/predictions/{run_id}.csv')
        assert len(predictions) == 100
        np.testing.assert_allclose([[float(p['y_pred'])] for p in predictions], predict(model, splits['test'][0]), atol=1e-15, rtol=1e-14)
        np.testing.assert_array_equal([[float(p['y_true'])] for p in predictions], splits['test'][1])
        records.append(record)
    saved_summary = read_csv(output / 'tables/summary.csv')
    recalculated = summarize(records)
    assert len(saved_summary) == len(recalculated) == len(specifications)
    for saved, actual in zip(saved_summary, recalculated):
        for key, value in actual.items():
            if isinstance(value, (int, float)):
                assert np.isclose(float(saved[key]), value, rtol=1e-10, atol=1e-15), (key, saved[key], value)
            else:
                assert saved[key] == value
    selected = json.loads((output / 'selection.json').read_text())
    assert selected['run_id'] == select_model(records)['run_id']
    assert selected['locked_before_test'] and selected['locked_utc'] <= manifest['completed_utc']
    validation_records = json.loads((output / 'validation_records.json').read_text())
    assert all(not any(k.startswith('test_') for k in r) for r in validation_records)
    if output.resolve() == (ROOT / 'results').resolve():
        assert config == json.loads((ROOT / 'configs/official.json').read_text(encoding='utf-8'))
        assert config['experiments']['A'] == list(range(200, 2001, 200))
        assert config['experiments']['B'] == list(range(1, 11))
        assert all(depth in config['experiments']['C'] for depth in [1, 2, 3, 4])
        for seed in config['seeds']:
            baselines = [next(r for r in records if r['run_id'] == name) for name in
                         [f'A_2000_seed{seed}', f'B_0010_seed{seed}', f'C_0001_seed{seed}']]
            assert len({r['test_mse'] for r in baselines}) == 1
        originals = json.loads((ROOT / 'reports/source_materials/manifest.json').read_text())
        for entry in originals:
            assert hashlib.sha256((ROOT / entry['file']).read_bytes()).hexdigest() == entry['sha256']
    result = dict(status='PASS', runs_verified=len(records), configurations_verified=len(specifications),
                  seeds=config['seeds'], test_samples=100, checkpoint_predictions_recomputed=True,
                  metrics_recomputed=['MSE', 'MAE', 'RMSE'], summary_recomputed=True,
                  no_cross_split_duplicates=True, validation_only_selection=True)
    if documents:
        pages = len(PdfReader(ROOT / 'reports/ACADEMIC_REPORT.pdf').pages)
        slides = len(Presentation(ROOT / 'slides/BTL01_PRESENTATION.pptx').slides)
        assert 10 <= pages <= 20, pages
        assert 10 <= slides <= 15, slides
        assert (ROOT / 'reports/ACADEMIC_REPORT.docx').stat().st_size > 10000
        for file in ['README.md', 'reports/IMPLEMENTATION_REPORT.md', 'reports/REQUIREMENT_TRACEABILITY.md']:
            assert (ROOT / file).stat().st_size > 1000
        junit = ET.parse(output / 'verification/pytest.xml').getroot()
        suites = list(junit.iter('testsuite'))
        assert suites and all(int(s.get('failures', 0)) == 0 and int(s.get('errors', 0)) == 0
                              and int(s.get('skipped', 0)) == 0 for s in suites)
        result.update(report_pdf_pages=pages, pptx_slides=slides,
                      tests_passed=sum(int(s.get('tests', 0)) for s in suites))
    write_json(output / 'verification/audit.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--no-documents', action='store_true')
    args = parser.parse_args()
    audit(args.output, not args.no_documents)
