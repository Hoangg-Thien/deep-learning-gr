import json
from pathlib import Path
import pytest

from scripts.audit import expected_runs, resolve_data_directory, validate_run_config


def test_relocated_audit_prefers_local_csv_even_if_old_checkout_exists(tmp_path):
    old = tmp_path / 'old/data'
    local_root = tmp_path / 'submission'
    local = local_root / 'data'
    old.mkdir(parents=True)
    local.mkdir(parents=True)
    assert resolve_data_directory(local_root / 'results', {'data_directory': str(old)}, local_root) == local


def test_custom_output_prefers_its_own_data(tmp_path):
    local = tmp_path / 'results/other/data'
    local.mkdir(parents=True)
    assert resolve_data_directory(local.parent, {'data_directory': 'missing'}, tmp_path) == local


@pytest.mark.parametrize('key,value', [('lr', 0.3), ('hidden_sizes', [9])])
def test_audit_rejects_consistently_tampered_checkpoint_and_config(key, value):
    config = json.loads(Path('configs/official.json').read_text())
    expected = expected_runs(config)
    run_id = 'B_0010_seed11'
    actual = dict(expected[run_id]['config'])
    actual[key] = value
    record = dict(run_id=run_id, experiment='B', value=10, n_samples=2000, width=10, depth=1, seed=11)
    if key == 'hidden_sizes':
        record['width'] = 9
    with pytest.raises(AssertionError):
        validate_run_config(run_id, record, actual, actual, expected)
