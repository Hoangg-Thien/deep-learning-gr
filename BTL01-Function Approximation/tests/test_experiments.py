import json
from pathlib import Path

from src.experiments import experiment_specs, select_model


def test_official_ranges_and_controlled_variables():
    config = json.loads(Path('configs/official.json').read_text())
    specs = experiment_specs(config)
    assert len(specs) == 24
    for group, expected in [('A', list(range(200, 2001, 200))), ('B', list(range(1, 11))), ('C', [1, 2, 3, 4])]:
        rows = [s for s in specs if s['experiment'] == group]
        assert [r['value'] for r in rows] == expected
        for r in rows:
            if group == 'A':
                assert r['hidden_sizes'] == [10] and r['n_samples'] == r['value']
            elif group == 'B':
                assert r['hidden_sizes'] == [r['value']] and r['n_samples'] == 2000
            else:
                assert r['hidden_sizes'] == [10] * r['value'] and r['n_samples'] == 2000


def test_model_selection_ignores_test_and_uses_mean_over_seeds():
    rows = [dict(run_id='x1', experiment='A', value=200, seed=1, validation_mse=0.1, test_mse=100),
            dict(run_id='x2', experiment='A', value=200, seed=2, validation_mse=0.2, test_mse=100),
            dict(run_id='y1', experiment='B', value=2, seed=1, validation_mse=0.3, test_mse=0)]
    selection = select_model(rows)
    assert selection['run_id'] == 'x1'
    assert selection['mean_validation_mse'] == (0.1 + 0.2) / 2
