import json
from astro_collision_lab.preparation_result import write_failure, failure_message


def test_failed_isolation_round_trip_preserves_log_and_measurements(tmp_path):
    marker=tmp_path/'ready.json'
    request=tmp_path/'request.json'
    request.write_text(json.dumps({'marker':str(marker),'identity':{'request_id':'current'}}))
    report={'pass':False,'groups':{'Galaxy B':{'pass':False,'max_radius_drift':.1829,
        'components':{'Disk':{'pass':False,'radius_drift':.1829,'thickness_drift':1.549,
                             'thresholds':{'disk_height_drift':.1}}}}}}
    (tmp_path/'ready.json.isolation.json').write_text(json.dumps(report))
    log=tmp_path/'worker.log';log.write_text('original traceback')
    write_failure(request, RuntimeError('Galaxy isolation gate failed; inspect '+str(marker)+'.isolation.json'))
    message=failure_message(tmp_path,{'request_id':'current'})
    assert 'Galaxy B' in message and '18.29%' in message and '1.549' in message
    assert 'Start' in message and log.read_text()=='original traceback'


def test_missing_result_and_log_fallback(tmp_path):
    assert 'worker.log' in failure_message(tmp_path,{'request_id':'missing'})


def test_corrupt_isolation_report_retains_original_exception(tmp_path):
    request=tmp_path/'request.json'
    request.write_text(json.dumps({'marker':str(tmp_path/'ready'),'identity':{'request_id':'r'}}))
    (tmp_path/'ready.isolation.json').write_text('{')
    write_failure(request,ValueError('invalid configuration'))
    assert 'invalid configuration' in failure_message(tmp_path,{'request_id':'r'})


def test_failure_serialization_is_bounded_and_does_not_surface_unrelated_path(tmp_path):
    request=tmp_path/'request.json'
    request.write_text(json.dumps({'marker':str(tmp_path/'ready'),'identity':{'request_id':'r'}}))
    write_failure(request,RuntimeError('Cannot open C:\\private\\unrelated.txt ' + 'x'*20000))
    result=json.loads((tmp_path/'failure.json').read_text())
    assert result['schema']==1
    assert len(result['reason'])<=12000
    assert 'unrelated.txt' not in result['reason']
