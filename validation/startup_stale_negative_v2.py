"""Disposable negative control: old success must not validate a new launch."""
import os,json,pathlib
pid=os.getpid()
p=pathlib.Path(os.environ['BLENDER_USER_CONFIG'])/('session-'+str(pid)+'.json')
p.write_text(json.dumps({'pid':pid,'launch_id':'previous-launch','phase':'READY','ticks':99,'draws':99,'authored_count':100000,'preview_count':100000}))
