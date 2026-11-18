"""Public-control GUI probe; run only in an owned isolated MCP session.

Start with begin(). Read bpy.app.driver_namespace['acl_tree_controls_proof'].
No solver pumping, timeline playback, regular preferences, or original-file writes.
"""
import time
import traceback

import bpy
from astro_collision_lab import runtime
from astro_collision_lab.simulation import SimulationConfig


def begin():
    scene = bpy.context.scene
    roots = sorted((obj for obj in scene.objects if obj.acl_body.is_galaxy), key=lambda obj: obj.name)
    if len(roots) != 2:
        raise ValueError('This probe requires the two-galaxy review scene')
    proof = {'phase': 'SETUP', 'errors': [], 'checks': {}, 'observations': [],
             'started': time.time(), 'initial_file': bpy.data.filepath}
    bpy.app.driver_namespace['acl_tree_controls_proof'] = proof
    settings = scene.acl_settings
    bpy.ops.acl.reset()
    settings.tree_storage = 'COMPACT'
    settings.tree_max_depth = 16
    for obj in roots:
        obj.acl_body.particle_count = 128
    phase = 'START_SMALL'
    deadline = time.monotonic() + 120
    observed = None

    def tick():
        nonlocal phase, deadline, observed
        try:
            state = runtime.controller.status()
            if state['error']:
                raise RuntimeError(state['error'])
            if time.monotonic() > deadline:
                raise TimeoutError('Public control probe timed out in ' + phase)
            sim = runtime.controller.simulation
            config = sim.config if sim is not None else None
            if phase == 'START_SMALL':
                proof['checks']['start'] = sorted(bpy.ops.acl.start())
                phase = 'SMALL_16'
            elif phase == 'SMALL_16' and state['running'] and state['frame'] >= 3:
                assert state['active_count'] == 256 and config.tree_max_depth == 16
                proof['observations'].append({'label': 'depth16', 'state': state})
                bpy.ops.acl.pause()
                observed = runtime.controller.status()['frame']
                phase = 'PAUSE'
            elif phase == 'PAUSE':
                assert not state['running'] and state['frame'] == observed
                proof['checks']['pause_holds'] = True
                bpy.ops.acl.step()
                phase = 'STEP'
            elif phase == 'STEP' and state['frame'] == observed + 1:
                assert not state['running']
                proof['checks']['single_step'] = True
                bpy.ops.acl.start()
                settings.tree_max_depth = 9
                phase = 'EDIT_9'
            elif phase == 'EDIT_9' and state['running'] and config.tree_max_depth == 9 and state['frame'] >= 3:
                assert sim.runtime.config.max_depth == 9
                assert sim.runtime.config.stack_capacity == 64
                proof['checks']['depth9_rebuild_resume'] = True
                proof['observations'].append({'label': 'depth9', 'state': state})
                settings.tree_max_depth = 16
                phase = 'EDIT_16'
            elif phase == 'EDIT_16' and state['running'] and config.tree_max_depth == 16 and state['frame'] >= 3:
                assert sim.runtime.config.max_depth == 16
                assert sim.runtime.config.stack_capacity == 113
                proof['checks']['depth16_rebuild_resume'] = True
                try:
                    SimulationConfig(particle_count=800000, tree_max_depth=16,
                                     tree_storage='STRIDED').validate()
                except ValueError as exc:
                    proof['checks']['baseline_800k_rejected'] = str(exc)
                else:
                    raise AssertionError('Reserved baseline unexpectedly admitted 800k at depth16')
                for obj in roots:
                    obj.acl_body.particle_count = 400000
                phase = 'LARGE'
                deadline = time.monotonic() + 180
            elif phase == 'LARGE' and state['running'] and state['active_count'] == 800000 and state['frame'] >= 3:
                assert config.tree_max_depth == 16
                assert sim.runtime.config.max_depth == 16
                assert config.tree_storage == 'COMPACT'
                assert sim.runtime.storage_status()['actual_nodes'] < 9000000
                assert state['viewport']['batches'] and state['viewport']['batches'][0]['strict_interop']
                observed = (state['frame'], state['physical_time'],
                            state['viewport']['batches'][0]['updates'],
                            state['viewport']['batches'][0]['draws'])
                proof['observations'].append({'label': 'large_start', 'state': state})
                phase = 'LARGE_ADVANCE'
            elif phase == 'LARGE_ADVANCE' and state['frame'] >= observed[0] + 10:
                assert state['physical_time'] > observed[1]
                batch = state['viewport']['batches'][0]
                assert batch['updates'] > observed[2] and batch['draws'] > observed[3]
                assert not state['viewport']['last_error'] and not batch['fault']
                proof['observations'].append({'label': 'large_advanced', 'state': state})
                proof['checks']['800k_depth16_live_solver_and_display'] = True
                proof['storage'] = sim.runtime.storage_status()
                proof['phase'] = 'DONE'
                proof['finished'] = time.time()
                return None
            proof['phase'] = phase
            return .4
        except Exception:  # noqa: BLE001 - probe timer records any failure
            proof['errors'].append(traceback.format_exc())
            proof['phase'] = 'FAILED'
            bpy.ops.acl.pause()
            return None

    bpy.app.timers.register(tick, first_interval=1.0)
    return proof
