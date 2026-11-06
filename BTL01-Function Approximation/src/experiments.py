"""Controlled A/B/C experiments; lock validation selection before testing."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from .data import generate_splits, load_csv, save_csv
from .metrics import regression_metrics
from .training import fit, load_checkpoint, predict
from .visualization import comparison_figures, data_figures, evaluation_figures, loss_figure


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def experiment_specs(config, group='all'):
    baseline = config['baseline']
    specs = []
    for name, values in config['experiments'].items():
        if group != 'all' and group != name:
            continue
        for value in values:
            n = value if name == 'A' else baseline['n_samples']
            width = value if name == 'B' else baseline['width']
            depth = value if name == 'C' else baseline['depth']
            if n > config['n_train'] or n < 1 or width < 1 or depth < 1:
                raise ValueError('Experiment exceeds train pool or has invalid sizes')
            specs.append(dict(experiment=name, value=value, n_samples=n,
                              hidden_sizes=[width] * depth))
    if not specs:
        raise ValueError('No experiment configurations')
    return specs


def select_model(records):
    groups = {}
    for row in records:
        groups.setdefault((row['experiment'], row['value']), []).append(row)
    key = min(groups, key=lambda k: (np.mean([r['validation_mse'] for r in groups[k]]), k))
    representative = min(groups[key], key=lambda r: (r['validation_mse'], r['seed']))
    return dict(experiment=key[0], value=key[1], run_id=representative['run_id'],
                seed=representative['seed'], mean_validation_mse=float(np.mean(
                    [r['validation_mse'] for r in groups[key]])),
                criterion='minimum mean validation MSE across seeds, then minimum validation seed',
                locked_before_test=True)


def summarize(records):
    rows = []
    keys = sorted(set((r['experiment'], r['value']) for r in records))
    metric_keys = [f'{split}_{metric}' for split in ['train', 'validation', 'test']
                   for metric in ['mse', 'mae', 'rmse']]
    for group, value in keys:
        subset = [r for r in records if r['experiment'] == group and r['value'] == value]
        row = dict(experiment=group, value=value, seeds=len(subset),
                   n_samples=subset[0]['n_samples'], width=subset[0]['width'], depth=subset[0]['depth'])
        for key in metric_keys + ['training_seconds', 'epochs_ran', 'best_epoch']:
            array = np.array([r[key] for r in subset])
            row[f'{key}_mean'] = float(array.mean())
            row[f'{key}_std'] = float(array.std(ddof=1)) if len(array) > 1 else 0.
        rows.append(row)
    return rows


def run(config, output, data_dir, group='all'):
    output, data_dir = Path(output), Path(data_dir)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    marker = output / 'config.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8')) != config:
        raise ValueError('Output contains a different config; choose a new --output directory')
    write_json(marker, config)
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), status='running',
                    config_sha256=digest, python=sys.version, platform=platform.platform(),
                    torch=torch.__version__, numpy=np.__version__, cuda_available=torch.cuda.is_available(),
                    data_directory=str(data_dir.resolve()), requested_group=group)
    write_json(output / 'manifest.json', manifest)
    # Generation is independent. Training below deliberately reloads train/validation only.
    splits = generate_splits(config['n_train'], config['n_validation'], config['n_test'], config['data_seed'])
    for name, (x, y) in splits.items():
        save_csv(data_dir / f'{name}.csv', x, y)
    data_figures(splits, output / 'figures')
    statistics = []
    for name, (x, y) in splits.items():
        for index, label in enumerate(['x1', 'x2', 'y']):
            values = np.column_stack([x, y])[:, index]
            statistics.append(dict(split=name, variable=label, n=len(values), min=float(values.min()),
                                   max=float(values.max()), mean=float(values.mean()), std=float(values.std(ddof=1))))
    write_csv(output / 'tables/data_statistics.csv', statistics)
    del splits
    train_pool = load_csv(data_dir / 'train.csv')
    validation = load_csv(data_dir / 'validation.csv')
    records = []
    existing = output / 'validation_records.json'
    if group != 'all' and existing.exists():
        records = [r for r in json.loads(existing.read_text(encoding='utf-8')) if r['experiment'] != group]
    for spec in experiment_specs(config, group):
        for seed in config['seeds']:
            run_id = f"{spec['experiment']}_{spec['value']:04d}_seed{seed}"
            params = dict(config['training'], hidden_sizes=spec['hidden_sizes'], seed=seed)
            params.update(data_seed=config['data_seed'], n_samples=spec['n_samples'])
            write_json(output / f'logs/{run_id}_config.json', params)
            n = spec['n_samples']
            train = (train_pool[0][:n], train_pool[1][:n])
            result = fit(train, validation, params, output / f'checkpoints/{run_id}.pt')
            write_csv(output / f'logs/{run_id}.csv', result['history'])
            loss_figure(result['history'], output / f'figures/loss/{run_id}.png', run_id)
            record = dict(run_id=run_id, experiment=spec['experiment'], value=spec['value'], seed=seed,
                          n_samples=n, width=spec['hidden_sizes'][0], depth=len(spec['hidden_sizes']),
                          parameters=sum(p.numel() for p in result['model'].parameters()),
                          best_epoch=result['best_epoch'], epochs_ran=result['epochs_ran'],
                          stopped_early=result['stopped_early'], training_seconds=result['training_seconds'])
            for split_name, observations in [('train', train), ('validation', validation)]:
                metrics = regression_metrics(observations[1], predict(result['model'], observations[0]))
                record.update({f'{split_name}_{key}': val for key, val in metrics.items()})
            records.append(record)
            write_json(existing, records)
            print(f"{run_id}: epochs={result['epochs_ran']}, best={result['best_epoch']}, "
                  f"train={record['train_mse']:.6g}, val={record['validation_mse']:.6g}, "
                  f"time={record['training_seconds']:.2f}s", flush=True)
    selection = select_model(records)
    selection['locked_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output / 'selection.json', selection)
    # Only now open the independent test set; no subsequent hyperparameter search.
    test = load_csv(data_dir / 'test.csv')
    histories = {}
    for record in records:
        run_id = record['run_id']
        model, _ = load_checkpoint(output / f'checkpoints/{run_id}.pt')
        prediction = predict(model, test[0])
        record.update({f'test_{key}': val for key, val in regression_metrics(test[1], prediction).items()})
        write_csv(output / f'tables/predictions/{run_id}.csv', [
            dict(x1=float(x[0]), x2=float(x[1]), y_true=float(y[0]), y_pred=float(p[0]), residual=float(p[0] - y[0]))
            for x, y, p in zip(test[0], test[1], prediction)])
        histories[run_id] = [{k: float(v) for k, v in row.items()} for row in read_csv(output / f'logs/{run_id}.csv')]
        if run_id == selection['run_id']:
            evaluation_figures(model, test[0], test[1], prediction, output / 'figures')
    summary = summarize(records)
    write_csv(output / 'tables/all_runs.csv', records)
    write_csv(output / 'tables/summary.csv', summary)
    for name in 'ABC':
        subset = [r for r in records if r['experiment'] == name]
        if subset:
            write_csv(output / f'tables/experiment_{name}.csv', subset)
    comparison_figures(summary, records, histories, output / 'figures')
    manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                    run_count=len(records), selection=selection,
                    dataset_sha256={name: hashlib.sha256((data_dir / f'{name}.csv').read_bytes()).hexdigest()
                                    for name in ['train', 'validation', 'test']},
                    source_sha256={str(p).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(Path('src').glob('*.py'))})
    write_json(output / 'manifest.json', manifest)
    print(f"Completed {len(records)} runs; validation-selected model: {selection['run_id']}", flush=True)
    return records, selection
