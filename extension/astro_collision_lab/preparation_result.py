"""CPU-only preparation failure reporting; readiness remains separately verified."""
import json
import re
from pathlib import Path


def _display_reason(reason):
    # Full diagnostics stay in the owned log; arbitrary exception paths do not
    # belong in the status panel. The owned log link is appended separately.
    reason = re.sub(r"[A-Za-z]:[\\/][^\r\n;]+", '[path in worker.log]', reason)
    reason = re.sub(r"(?<!\S)(?:\\\\|/)[^\s;]+", '[path in worker.log]', reason)
    return reason[:12000]


def isolation_reason(report):
    reasons = []
    for name, group in report.get('groups', {}).items():
        if group.get('pass'):
            continue
        if group.get('error'):
            reasons.append(f"{name}: {group['error']}")
        for metric, limit in (('max_radius_drift', .05), ('max_energy_drift', .01), ('max_angular_drift', .01)):
            value = group.get(metric)
            if isinstance(value, (int, float)) and value > limit:
                reasons.append(f'{name}: {metric} {value:.2%} exceeds {limit:.0%}')
        for component, result in group.get('components', {}).items():
            if result.get('pass'):
                continue
            # Report measured values without applying new acceptance criteria.
            reasons.append(f"{name}/{component}: radius drift {result.get('radius_drift', 'unknown')}; height drift {result.get('thickness_drift', 'unknown')}; thresholds {result.get('thresholds', {})}")
        if not reasons:
            reasons.append(f'{name}: isolation validation failed')
    return '; '.join(reasons)


def write_failure(request_path, error):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    reason = f'{type(error).__name__}: {error}'
    try:
        if not str(error).startswith('Galaxy isolation gate failed; inspect '):
            raise ValueError('No isolation report belongs to this failure')
        report = json.loads(Path(str(request['marker'])+'.isolation.json').read_text(encoding='utf-8'))
        if report.get('pass') is False:
            reason += '; ' + isolation_reason(report)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    result = {'schema': 1, 'request_id': request['identity']['request_id'], 'reason': _display_reason(reason)}
    temporary = request_path.parent / 'failure.tmp'
    temporary.write_text(json.dumps(result), encoding='utf-8')
    temporary.replace(request_path.parent / 'failure.json')


def failure_message(owned, identity):
    owned = Path(owned)
    reason = 'Preparation failed'
    try:
        result = json.loads((owned/'failure.json').read_text(encoding='utf-8'))
        if result.get('schema') == 1 and result.get('request_id') == identity['request_id'] and isinstance(result.get('reason'), str):
            reason = _display_reason(result['reason'])
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return f'{reason}. Edit settings and use Start to rebuild; see {owned / "worker.log"}'
