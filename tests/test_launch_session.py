import importlib.util
from pathlib import Path
from types import SimpleNamespace


def api():
    p=Path(__file__).resolve().parents[1]/'tools/launch-session.py'
    spec=importlib.util.spec_from_file_location('acl_launch_test',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def valid():return {'pid':12,'launch_id':'new','phase':'READY','ticks':5,'draws':3,'authored_count':100000,'preview_count':100000}


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
    m=api();clock=[0.];alive=lambda:True
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
