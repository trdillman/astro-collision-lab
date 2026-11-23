import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def api():
    p=Path(__file__).resolve().parents[1]/'tools/launch-session.py'
    spec=importlib.util.spec_from_file_location('acl_launch_test',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def valid():return {'pid':12,'launch_id':'new','phase':'READY','ticks':5,'draws':3,'authored_count':100000,'preview_count':100000}


def configure_launch(monkeypatch, profile):
    monkeypatch.setattr(sys, 'argv', [
        'launch-session.py', '--blender', 'blender.exe', '--profile', str(profile),
        '--bootstrap', 'startup.py',
    ])
    (profile / 'config').mkdir()


def test_cli_reuses_responsive_project_process(monkeypatch, tmp_path):
    launcher = api()
    configure_launch(monkeypatch, tmp_path)
    record = {'pid': 12, 'launch_id': 'new', 'identity': 'existing-process'}
    session = tmp_path / 'session.json'
    session.write_text(json.dumps(record), encoding='utf-8')
    ready = tmp_path / 'config' / 'session-12.json'
    ready.write_text(json.dumps(valid()), encoding='utf-8')
    monkeypatch.setattr(launcher, 'process_identity', lambda pid: 'existing-process')
    focused = []

    def focus(pid, focus=False):
        focused.append((pid, focus))
        return True

    monkeypatch.setattr(launcher, 'window_state', focus)
    monkeypatch.setattr(launcher, 'spawn', lambda *args: pytest.fail('launched a duplicate'))
    launcher.main()

    assert focused == [(12, True)]
    assert json.loads(session.read_text(encoding='utf-8')) == record


def test_cli_closes_failed_launch_it_owns(monkeypatch, tmp_path):
    launcher = api()
    configure_launch(monkeypatch, tmp_path)
    cleanup = []
    child = SimpleNamespace(
        pid=12,
        poll=lambda: None,
        kill=lambda: cleanup.append('kill'),
        wait=lambda timeout: cleanup.append(('wait', timeout)),
    )
    monkeypatch.setattr(launcher, 'spawn', lambda *args: child)
    monkeypatch.setattr(launcher, 'process_identity', lambda pid: 'started-process')
    monkeypatch.setattr(launcher, 'await_ready', lambda *args: False)

    with pytest.raises(RuntimeError, match='the failed launch was closed'):
        launcher.main()

    result = json.loads((tmp_path / 'startup-result.json').read_text(encoding='utf-8'))
    assert result['window_responding'] is False
    assert cleanup == ['kill', ('wait', 10)]


def test_previous_launch_cannot_validate_reopened_window():
    m=api();d=valid();d['launch_id']='old'
    assert not m.ready_matches(d,12,'new')
    d=valid();d['pid']=11
    assert not m.ready_matches(d,12,'new')
    assert not m.ready_matches({'pass':True,'particle_count':100000},12,'new')


def test_scene_creation_without_rendered_frames_is_not_ready():
    m=api();d=valid();d['draws']=0
    assert not m.ready_matches(d,12,'new')
    d=valid();d['preview_count']=0
    assert not m.ready_matches(d,12,'new')
    assert m.ready_matches(valid(),12,'new')


def test_bootstrap_ready_alone_cannot_override_unresponsive_window():
    m=api();clock=[0.]
    def alive():return True
    def pause(t):clock[0]+=t
    assert not m.await_ready(lambda:valid(),lambda:False,alive,12,'new',.5,lambda:clock[0],pause)
    assert m.await_ready(lambda:valid(),lambda:True,alive,12,'new',.5,lambda:clock[0],pause)


def test_child_does_not_inherit_controller_output_pipes(tmp_path):
    m=api();captured={}
    def popen(args,**kwargs):
        captured.update(kwargs);assert kwargs['stdout'].fileno()>=0;assert kwargs['stderr'].fileno()>=0
        return SimpleNamespace(pid=123)
    m.spawn(['blender'],{},tmp_path,popen=popen)
    assert captured['close_fds'] is True and captured['stdin']==m.subprocess.DEVNULL
    assert captured['stdout']!=m.subprocess.PIPE and captured['stderr']!=m.subprocess.PIPE
