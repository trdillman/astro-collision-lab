"""The paired comparison must reject invalid evidence and avoid frame pseudoreplication."""
import copy
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def api():
    path = ROOT / 'validation' / 'paired_solver_probe.py'
    assert path.is_file(), 'Executable paired solver comparison is missing'
    spec = importlib.util.spec_from_file_location('paired_solver_probe_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_is_reproducible_matched_and_counterbalanced():
    a = api()
    plan = a.make_plan([100000, 400000], pairs=8, seed=41)
    assert plan == a.make_plan([100000, 400000], pairs=8, seed=41)
    assert len(plan) == 64
    for count in (100000, 400000):
        rows = [row for row in plan if row['count'] == count]
        for phase in ('aa', 'ab'):
            pairs = [rows[i:i+2] for i in range(0, len(rows), 2) if rows[i]['phase'] == phase]
            assert len(pairs) == 8
            assert all(x[0]['seed'] == x[1]['seed'] for x in pairs)
            assert all(x[0]['pair'] == x[1]['pair'] for x in pairs)
            assert sum(x[0]['arm'] == 'a' for x in pairs) == 4
            assert all(x['variant'] == 'baseline' for p in pairs for x in p) if phase == 'aa' else True
    with pytest.raises(ValueError):
        a.make_plan([100000], pairs=7, seed=41)


def test_eight_pair_interval_reports_actual_conservative_coverage():
    result = api().median_interval([.98, 1.01, 1.03, 1.02, .99, 1.04, 1.05, 1.0])
    assert result['low'] == .98
    assert result['high'] == 1.05
    assert result['coverage'] == pytest.approx(.9921875)
    assert result['independent_pairs'] == 8


def test_improvement_regression_and_noise_are_distinct():
    a = api()
    assert a.classify([1.01] * 8, [1.4] * 8)['status'] == 'IMPROVEMENT'
    assert a.classify([1.0] * 8, [.7] * 8)['status'] == 'REGRESSION'
    assert a.classify([1.0] * 8, [.98, 1.03] * 4)['status'] == 'INCONCLUSIVE'
    assert a.classify([.8, 1.2] * 4, [1.4] * 8)['status'] == 'NOISY_BASELINE'


def test_nearest_rank_p95_keeps_the_tail():
    assert api().p95(list(range(1, 21))) == 19


def good_record():
    row = dict(sequence=0, count=100000, phase='ab', pair=0, arm='a',
               variant='baseline', seed=11)
    record = dict(plan=row, contract_sha256='contract', source_sha256='source',
                  actual_backend='cuda', quality_status='PASS',
                  measured_samples_ms=[2.0] * 20, warmup_samples_ms=[3.0] * 2,
                  count=100000, active_count=100000, max_depth=16, leaf_capacity=8,
                  theta=.6, softening=.03, precision='float32', substeps=1,
                  gravitational_constant=1.0,
                  physical_dt=.02, physical_advance=.4, initial_state_sha256='initial',
                  process_nonce='fresh', lease_verdict='clean',
                  monitor_before=dict(id='lease', samples=1), monitor_after=dict(id='lease', samples=2))
    return row, record


@pytest.mark.parametrize('field,value,reason', [
    ('actual_backend', 'cpu', 'backend'),
    ('quality_status', 'FAIL', 'quality'),
    ('active_count', 99999, 'count'),
    ('max_depth', 9, 'depth'),
    ('leaf_capacity', 4, 'leaf'),
    ('theta', .8, 'theta'),
    ('softening', .1, 'softening'),
    ('precision', 'float64', 'precision'),
    ('substeps', 2, 'substeps'),
    ('gravitational_constant', 2., 'gravity'),
    ('physical_dt', .01, 'dt'),
    ('physical_advance', .2, 'physical'),
    ('contract_sha256', 'stale', 'contract'),
    ('source_sha256', 'stale', 'source'),
    ('lease_verdict', 'contaminated', 'lease'),
])
def test_trust_boundary_rejects_changed_work_or_bad_evidence(field, value, reason):
    a = api()
    row, record = good_record()
    record[field] = value
    with pytest.raises(ValueError, match=reason):
        a.validate_record(record, row, contract_sha256='contract', source_sha256='source',
                          frames=20, warmup=2, dt=.02)


def test_valid_record_accepts_and_invalid_raw_timing_rejects():
    a = api()
    row, record = good_record()
    a.validate_record(record, row, contract_sha256='contract', source_sha256='source',
                      frames=20, warmup=2, dt=.02)
    for samples in ([2.0] * 19, [float('nan')] * 20, [-1.0] * 20):
        broken = copy.deepcopy(record)
        broken['measured_samples_ms'] = samples
        with pytest.raises(ValueError, match='sample'):
            a.validate_record(broken, row, contract_sha256='contract', source_sha256='source',
                              frames=20, warmup=2, dt=.02)


def test_paired_output_oracle_rejects_corruption_and_changed_initial_state():
    a = api()
    baseline = dict(initial_state_sha256='same', sample_indices=[0, 1],
                    positions_sample=[[0., 0., 0.], [1., 2., 3.]],
                    velocities_sample=[[.1, .2, .3], [.4, .5, .6]])
    assert a.validate_pair(baseline, copy.deepcopy(baseline), .001)['positions_sample'] == 0
    for field, value in [('initial_state_sha256', 'different'),
                         ('positions_sample', [[float('nan'), 0., 0.], [1., 2., 3.]]),
                         ('velocities_sample', [[.1, .2, .3]]),
                         ('positions_sample', [[0., 0., 0.], [2., 4., 6.]])]:
        candidate = copy.deepcopy(baseline)
        candidate[field] = value
        with pytest.raises(ValueError):
            a.validate_pair(baseline, candidate, .001)


def test_execution_evidence_rejects_tampering(tmp_path):
    a = api()
    directory = tmp_path/'trials'/'one'
    directory.mkdir(parents=True)
    lease = dict(verdict='clean', cls='exclusive')
    record = dict(directory=str(directory), lease=lease, exit_code=0)
    a.write_json(directory/'lease.json', lease)
    a.write_json(directory/'result.json', record)
    record.update(evidence_sha256=a.file_sha(directory/'result.json'),
                  lease_sha256=a.file_sha(directory/'lease.json'))
    a.verify_evidence(tmp_path, record)
    changed = copy.deepcopy(record)
    changed['exit_code'] = 1
    with pytest.raises(ValueError, match='aggregate'):
        a.verify_evidence(tmp_path, changed)
    a.write_json(directory/'lease.json', dict(verdict='contaminated'))
    with pytest.raises(ValueError, match='corrupt'):
        a.verify_evidence(tmp_path, record)


def test_analysis_rejects_missing_trials(tmp_path):
    a = api()
    contract = dict(verifier=dict(sha256=a.file_sha(a.__file__),
                        measurer_sha256=a.file_sha(a.MEASURER)), variants={}, plan=[{'sequence': 0}])
    a.write_json(tmp_path/'contract.json', contract)
    a.write_json(tmp_path/'records.json', [])
    with pytest.raises(ValueError, match='missing trials'):
        a.analyze_study(tmp_path)
