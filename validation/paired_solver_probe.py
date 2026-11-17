"""Matched main/candidate CUDA solver study, extending solver_probe's real adapter.

SPDX-License-Identifier: LicenseRef-Private-Internal
The study measures completed solver work, not viewport cadence. Every timed trial
is a fresh isolated Blender process under a clean exclusive blender-gate lease.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid


SCHEMA = 1
ROOT = Path(__file__).resolve().parents[1]
MEASURER = Path(__file__).with_name('solver_probe.py')


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_sha(root):
    root = Path(root)
    result = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and (p.suffix == '.py' or p.name == 'blender_manifest.toml'):
            result.append((p.relative_to(root).as_posix(), file_sha(p)))
    if not result:
        raise ValueError('empty source artifact')
    return digest(result)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def make_plan(counts, pairs=8, seed=42):
    if type(pairs) is not int or pairs < 8 or pairs % 2:
        raise ValueError('at least eight even independent pairs required')
    if not counts or len(set(counts)) != len(counts) or any(type(n) is not int or n < 4 for n in counts):
        raise ValueError('unique admitted particle counts required')
    rng = random.Random(seed)
    plan = []
    for phase in ('aa', 'ab'):
        blocks = [(n, pair) for n in counts for pair in range(pairs)]
        rng.shuffle(blocks)
        for count, pair in blocks:
            paired_seed = (seed + count * 131 + pair * 17 + (100003 if phase == 'ab' else 0)) % (2**31)
            for arm in (('a', 'b') if pair % 2 == 0 else ('b', 'a')):
                plan.append(dict(sequence=len(plan), count=count, phase=phase, pair=pair,
                                 arm=arm, variant='baseline' if phase == 'aa' or arm == 'a' else 'candidate',
                                 seed=paired_seed))
    return plan


def p95(values):
    ordered = sorted(values)
    return ordered[math.ceil(.95 * len(ordered)) - 1]


def median_interval(values, confidence=.95):
    ordered = sorted(values)
    n = len(ordered)
    if n < 8:
        raise ValueError('insufficient independent pairs')
    tail = 0
    chosen = None
    for rank in range(1, n // 2 + 1):
        tail += math.comb(n, rank - 1)
        coverage = 1 - 2 * tail / 2**n
        if coverage >= confidence:
            chosen = dict(low=ordered[rank-1], high=ordered[n-rank], coverage=coverage)
        else:
            break
    return dict(chosen, independent_pairs=n, requested_confidence=confidence,
                method='paired-median-order-statistic')


def classify(aa, ab, noise=.05):
    repeatability = median_interval(aa)
    interval = median_interval(ab)
    if repeatability['low'] < 1-noise or repeatability['high'] > 1+noise:
        status = 'NOISY_BASELINE'
    elif interval['low'] > 1:
        status = 'IMPROVEMENT'
    elif interval['high'] < 1:
        status = 'REGRESSION'
    else:
        status = 'INCONCLUSIVE'
    return dict(status=status, paired_median_speedup=statistics.median(ab),
                interval=interval, aa_interval=repeatability,
                practical_noise_envelope=noise)


def validate_record(record, row, *, contract_sha256, source_sha256, frames, warmup, dt):
    def check(condition, reason):
        if not condition:
            raise ValueError(reason)
    check(record.get('plan') == row, 'trial plan mismatch')
    check(record.get('contract_sha256') == contract_sha256, 'contract mismatch')
    check(record.get('source_sha256') == source_sha256, 'source artifact mismatch')
    check(record.get('actual_backend') == 'cuda', 'forbidden backend')
    check(record.get('quality_status') == 'PASS', 'quality failure')
    check(record.get('count') == row['count'] and record.get('active_count') == row['count'], 'count mismatch')
    check(record.get('max_depth') == 16, 'tree depth mismatch')
    check(record.get('leaf_capacity') == 8, 'tree leaf capacity mismatch')
    check(record.get('theta') == .6, 'tree theta mismatch')
    check(record.get('softening') == .03, 'tree softening mismatch')
    check(record.get('precision') == 'float32', 'precision mismatch')
    check(record.get('substeps') == 1, 'substeps mismatch')
    check(record.get('gravitational_constant') == 1., 'gravity constant mismatch')
    check(record.get('physical_dt') == dt, 'physical dt mismatch')
    check(math.isclose(record.get('physical_advance', -1), frames*dt, rel_tol=1e-8, abs_tol=1e-9), 'physical advance mismatch')
    check(record.get('lease_verdict') == 'clean', 'GPU lease not clean')
    before, after = record.get('monitor_before', {}), record.get('monitor_after', {})
    check(bool(before.get('id')) and before.get('id') == after.get('id')
          and before.get('samples', 0) > 0 and after.get('samples', 0) > before['samples'],
          'GPU isolation monitor checkpoints missing')
    for key, count in [('measured_samples_ms', frames), ('warmup_samples_ms', warmup)]:
        values = record.get(key)
        check(isinstance(values, list) and len(values) == count, 'sample count mismatch')
        check(all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in values), 'invalid sample')
    check(bool(record.get('initial_state_sha256')) and bool(record.get('process_nonce')), 'missing state or fresh-process identity')


def validate_pair(a, b, tolerance):
    if a['initial_state_sha256'] != b['initial_state_sha256']:
        raise ValueError('paired initial state mismatch')
    if a['sample_indices'] != b['sample_indices'] or not a['sample_indices']:
        raise ValueError('paired quality sample mismatch')
    errors = {}
    for field in ('positions_sample', 'velocities_sample'):
        lhs, rhs = a[field], b[field]
        if len(lhs) != len(rhs) or len(lhs) != len(a['sample_indices']):
            raise ValueError('paired output shape mismatch')
        if any(len(v) != 3 for v in lhs+rhs):
            raise ValueError('paired vector shape mismatch')
        x = [float(v) for row in lhs for v in row]
        y = [float(v) for row in rhs for v in row]
        if not all(math.isfinite(v) for v in x+y):
            raise ValueError('nonfinite paired output')
        error = math.sqrt(math.fsum((u-v)**2 for u, v in zip(x, y))) / max(math.sqrt(math.fsum(u*u for u in x)), 1e-30)
        if not math.isfinite(error) or error > tolerance:
            raise ValueError(f'paired trajectory quality mismatch: {field} {error}')
        errors[field] = error
    return errors


def prepare(args):
    if args.frames < 20 or args.warmup < 1:
        raise ValueError('at least twenty measured frames and one warmup required')
    output = Path(args.output).resolve()
    if (output / 'contract.json').exists():
        raise ValueError('a frozen study already exists; choose a new evidence directory')
    variants = {}
    for name, artifact, revision in [('baseline', args.baseline, args.baseline_revision),
                                     ('candidate', args.candidate, args.candidate_revision)]:
        artifact = Path(artifact).resolve()
        variants[name] = dict(root=str(artifact), source_id='git:'+revision,
                              source_sha256=source_sha(artifact),
                              storage='STRIDED' if name == 'baseline' else args.storage)
    sys.path.insert(0, str(Path(args.candidate).resolve().parent))
    measure = load_module('acl_paired_measure_plan', MEASURER)
    fixtures = {}
    for count in args.counts:
        config, bodies = measure.probe_scene('GALAXY', count, dt=.02)
        common = dataclasses.asdict(config)
        common.pop('tree_max_depth', None)
        common.pop('tree_storage', None)
        fixtures[str(count)] = dict(config=common, bodies=[dataclasses.asdict(body) for body in bodies])
    contract = dict(schema=SCHEMA, state='frozen', claim_scope='solver', comparison_kind='implementation',
                    statement='Completed CUDA solver work at identical quality; no viewport performance claim',
                    variants=variants, runtime=dict(blender=args.blender, warp_version='1.18.0',
                        dependency_root=str(Path(args.dependency).resolve()), kernel_cache=str(Path(args.cache).resolve())),
                    fixtures=fixtures, fixture_sha256=digest(fixtures),
                    timing=dict(boundary='host perf_counter before sim.step through CUDA device synchronize',
                        frames=args.frames, warmup=args.warmup, dt=.02,
                        isolation_monitor='Wait up to 20 seconds outside timing for a live GPU sample before measurement and a newer sample afterward',
                        cache='persistent JIT disk cache preheated by separate pilots; fresh process and isolated profile every trial',
                        exclusions=['initialization/JIT', 'initial and final validation readbacks', 'existing inter-frame scalar diagnostic reads', 'viewport packing and drawing']),
                    sampling=dict(pairs=args.pairs, seed=args.seed, requested_confidence=.95,
                        aa_relative_tolerance=.05, noise_envelope_note='Predeclared engineering repeatability threshold, not a scientific bound'),
                    quality=dict(max_sample_relative_l2=.001, finite_all_channels=True,
                        exact_initial_gpu_array_hash=True, sample_particles=1024,
                        note='FP32 paired trajectory comparison is an engineering regression oracle, not scientific validation'),
                    verifier=dict(path=str(Path(__file__).resolve()), sha256=file_sha(__file__),
                                  measurer=str(MEASURER), measurer_sha256=file_sha(MEASURER)),
                    plan=make_plan(args.counts, args.pairs, args.seed))
    write_json(output / 'contract.json', contract)
    print(json.dumps(dict(contract=str(output/'contract.json'), sha256=digest(contract), trials=len(contract['plan']))))


def monitor_checkpoint(previous=-1):
    """Wait for the installed gate's real monitor; never manufacture clean receipts."""
    info = json.loads((Path(os.environ['LOCALAPPDATA'])/'blender-gate'/'broker.json').read_text())
    deadline = time.monotonic()+20
    while time.monotonic() < deadline:
        req = urllib.request.Request(f"http://127.0.0.1:{info['port']}/status", data=b'{}',
            headers={'Content-Type': 'application/json', 'X-BGate-Token': info['token']})
        with urllib.request.urlopen(req, timeout=5) as response:
            status = json.load(response)
        lease = next((x for x in status['active'] if x['id'] == os.environ.get('BGATE_LEASE_ID')), None)
        if not lease or lease['cls'] != 'exclusive' or lease.get('contamination'):
            raise ValueError('exclusive GPU lease lost or contaminated')
        if lease['gpu_samples'] > max(previous, 0):
            return dict(id=lease['id'], samples=lease['gpu_samples'], observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        time.sleep(.25)
    raise ValueError('GPU isolation monitor produced no fresh sample within 20 seconds')


def trial(spec_path):
    """Executed only inside the fresh gated background Blender process."""
    import bpy
    spec = json.loads(Path(spec_path).read_text())
    contract = spec['contract']
    row = spec['row']
    variant = contract['variants'][row['variant']]
    result = dict(plan=row, contract_sha256=digest(contract), source_sha256=source_sha(variant['root']),
                  process_nonce=uuid.uuid4().hex, pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    sim = None
    try:
        if result['source_sha256'] != variant['source_sha256']:
            raise ValueError('source artifact changed')
        if file_sha(__file__) != contract['verifier']['sha256'] or file_sha(MEASURER) != contract['verifier']['measurer_sha256']:
            raise ValueError('benchmark verifier changed')
        sys.path.insert(0, contract['runtime']['dependency_root'])
        sys.path.insert(0, str(Path(variant['root']).parent))
        import numpy as np
        import warp as wp
        from astro_collision_lab import simulation
        if wp.__version__ != contract['runtime']['warp_version']:
            raise ValueError('loaded Warp version mismatch')
        if not Path(wp.__file__).resolve().is_relative_to(Path(contract['runtime']['dependency_root']).resolve()):
            raise ValueError('loaded Warp path mismatch')
        if not Path(simulation.__file__).resolve().is_relative_to(Path(variant['root']).resolve()):
            raise ValueError('loaded simulation path mismatch')
        wp.config.kernel_cache_dir = contract['runtime']['kernel_cache']
        fixture = contract['fixtures'][str(row['count'])]
        config = dict(fixture['config'], seed=row['seed'])
        if row['variant'] == 'candidate':
            config.update(tree_max_depth=16, tree_storage=variant['storage'])
        bodies = [simulation.BodySpec(**dict(body, seed=(row['seed']+i) % (2**31)))
                  for i, body in enumerate(fixture['bodies'])]
        sim = simulation.AstroCollisionSimulation(simulation.SimulationConfig(**config))
        initial = {}

        def initialized(owner):
            device = wp.get_device(owner.device_context.device)
            if not device.is_cuda:
                raise ValueError('forbidden CPU fallback')
            hasher = hashlib.sha256()
            channels = owner.display_channels()
            for name in ('positions', 'velocities', 'masses', 'radii', 'active', 'stable_ids'):
                array = channels.get(name, owner.runtime._arrays.get(name))
                if array is not None:
                    data = np.ascontiguousarray(array.numpy())
                    hasher.update(name.encode())
                    hasher.update(str(data.dtype).encode())
                    hasher.update(data.tobytes())
            initial.update(hash=hasher.hexdigest(), backend='cuda', device=str(device), device_name=device.name)
            result.update(actual_backend='cuda', device=str(device), device_name=device.name,
                          warp=wp.__version__, warp_path=wp.__file__,
                          blender=bpy.app.version_string, blender_binary=bpy.app.binary_path,
                          simulation_path=simulation.__file__)
            if row['phase'] == 'pilot':
                from astro_collision_lab._vendor.astro_warp.product.tree import BarnesHutConfig, BarnesHutRuntime
                from astro_collision_lab._vendor.astro_warp.product.sampling import BodySet
                runtime_type = BarnesHutRuntime
                if variant['storage'] == 'COMPACT':
                    from astro_collision_lab.compact_tree import CompactBarnesHutRuntime
                    runtime_type = CompactBarnesHutRuntime
                rng = np.random.default_rng(771)
                points = rng.normal(size=(32, 3)).astype(np.float32)
                masses = np.linspace(.01, .1, 32, dtype=np.float32)
                reference = runtime_type(owner.device_context, BarnesHutConfig(capacity=32, count=32,
                    gravitational_constant=1., dt=.0001, theta=.01, softening=.01, max_depth=16,
                    stack_capacity=113, leaf_capacity=8))
                try:
                    reference.load_bodies(BodySet(positions=points, velocities=np.zeros_like(points),
                        masses=masses, radii=np.zeros(32, np.float32), active=np.ones(32, np.int32),
                        stable_ids=np.arange(32, dtype=np.int64)))
                    delta = points[None, :, :].astype(np.float64) - points[:, None, :]
                    expected = (delta * masses[None, :, None] /
                        (np.sum(delta*delta, axis=2) + .01**2)[:, :, None]**1.5).sum(axis=1)
                    actual = reference._arrays['accelerations'].numpy()
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
                    result['force_oracle'] = dict(passed=True, count=32, unequal_masses=True,
                        maximum_absolute_error=float(np.max(np.abs(actual-expected))))
                finally:
                    reference.close()
            result['monitor_before'] = monitor_checkpoint()

        measure = load_module('acl_paired_measure_trial', contract['verifier']['measurer'])
        timings = measure.measure_simulation(sim, bodies,
            frames=contract['timing']['frames'], warmup=contract['timing']['warmup'],
            synchronize=lambda: wp.synchronize_device(sim.device_context.device), on_initialized=initialized)
        result['monitor_after'] = monitor_checkpoint(result['monitor_before']['samples'])
        channels = sim.display_channels()
        if spec.get('negative'):
            damaged = channels['positions'].numpy().copy()
            damaged[0, 0] = np.nan
            wp.copy(channels['positions'], wp.array(damaged, dtype=channels['positions'].dtype,
                                                   device=sim.device_context.device))
            wp.synchronize_device(sim.device_context.device)
            result['negative_gpu_write'] = True
        positions = channels['positions'].numpy()
        velocities = channels['velocities'].numpy()
        if not np.isfinite(positions).all():
            raise ValueError('nonfinite_positions')
        if not timings['finite'] or not np.isfinite(velocities).all() or timings['solver_guardrails_pass'] is not True:
            raise ValueError('quality guardrails failed')
        if timings['active_count'] != row['count'] or timings['count'] != row['count']:
            raise ValueError('active particle count changed')
        if timings['motion']['moving_particles'] == 0:
            raise ValueError('solver did not advance particle positions')
        indices = np.linspace(0, sim.count-1, min(1024, sim.count), dtype=np.int64)
        arrays = {}
        for name, array in sim.runtime._arrays.items():
            if getattr(array, 'ptr', None) and getattr(array, 'device', None) is not None:
                arrays[int(array.ptr)] = int(array.size) * wp.types.type_size_in_bytes(array.dtype)
        rt = sim.runtime.config
        result.update(actual_backend=initial['backend'], device=initial['device'], device_name=initial['device_name'],
            warp=wp.__version__, warp_path=wp.__file__, blender=bpy.app.version_string,
            blender_binary=bpy.app.binary_path, simulation_path=simulation.__file__,
            initial_state_sha256=initial['hash'], quality_status='PASS',
            count=timings['count'], active_count=timings['active_count'], max_depth=rt.max_depth,
            leaf_capacity=rt.leaf_capacity, theta=rt.theta, softening=rt.softening,
            precision=rt.precision.value, physical_dt=rt.dt, substeps=rt.substeps,
            gravitational_constant=rt.gravitational_constant,
            runtime_class=type(sim.runtime).__name__,
            physical_advance=timings['physical_time_advanced'], measured_samples_ms=timings['frame_ms'],
            warmup_samples_ms=timings['warmup_frame_ms'], owned_device_array_bytes=sum(arrays.values()),
            allocated_node_capacity=sim.runtime._total_nodes,
            storage=sim.runtime.storage_status() if hasattr(sim.runtime, 'storage_status') else {'storage': 'STRIDED'},
            graph_keys=list(sim.runtime._graphs), positions_sample=positions[indices].tolist(),
            velocities_sample=velocities[indices].tolist(), sample_indices=indices.tolist(),
            measurement=timings)
    except Exception as exc:
        result.update(quality_status='REJECTED', quality_failure=str(exc), traceback=traceback.format_exc())
        write_json(spec['result'], result)
        traceback.print_exc()
        if spec.get('negative') and str(exc) == 'nonfinite_positions':
            return
        raise
    finally:
        if sim is not None:
            sim.close()
    result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(spec['result'], result)


def execute_one(root, contract, row, *, name=None, negative=False):
    root = Path(root)
    name = name or f"{row['sequence']:03d}-{row['phase']}-{row['count']}-{row['pair']}-{row['arm']}"
    directory = root / 'trials' / name
    if directory.exists():
        raise ValueError(f'trial already exists: {directory}; never silently overwrite attempts')
    directory.mkdir(parents=True)
    spec = dict(contract=contract, row=row, result=str(directory/'result.json'), negative=negative)
    write_json(directory/'spec.json', spec)
    env = os.environ.copy()
    for key in ('RESOURCES', 'CONFIG', 'EXTENSIONS', 'SCRIPTS', 'DATAFILES'):
        folder = directory/'profile'/key.lower()
        folder.mkdir(parents=True)
        env['BLENDER_USER_'+key] = str(folder)
    env['ACL_KERNEL_CACHE'] = contract['runtime']['kernel_cache']
    env['PYTHONUNBUFFERED'] = '1'
    argv = ['bgate', 'run', '--exclusive', '--owner', 'galaxy-main-benchmark', '--label', name,
            '--blender', contract['runtime']['blender'], '--wait', '600', '--max-seconds', '240',
            '--report', str(directory/'lease.json'), '--', '-b', '--disable-autoexec', '--offline-mode',
            '--python-exit-code', '1', '--python', str(Path(__file__).resolve()), '--', 'trial', str(directory/'spec.json')]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    telemetry = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,pstate,power.limit,clocks.gr,clocks.mem',
                                '--format=csv'], capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    with (directory/'process.log').open('w', encoding='utf-8') as log:
        process = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=900, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if not (directory/'result.json').is_file() or not (directory/'lease.json').is_file():
        raise ValueError(f'missing runtime/lease evidence: {directory}')
    record = json.loads((directory/'result.json').read_text())
    lease = json.loads((directory/'lease.json').read_text())
    lease = lease.get('lease', lease)
    record.update(lease_verdict=lease.get('verdict'), lease=lease, exit_code=process.returncode,
                  process_started_utc=started, process_finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  argv=argv, directory=str(directory),
                  environment=dict(gpu_before_launch=telemetry, os=platform.platform(), cpu=platform.processor()))
    write_json(directory/'result.json', record)
    if process.returncode != 0:
        raise ValueError(f'Blender trial failed ({process.returncode}): {directory}')
    record['evidence_sha256'] = file_sha(directory/'result.json')
    record['lease_sha256'] = file_sha(directory/'lease.json')
    return record


def verify_evidence(root, rec):
    directory = Path(rec['directory']).resolve()
    if not directory.is_relative_to((Path(root)/'trials').resolve()):
        raise ValueError('execution evidence escapes the study root')
    if file_sha(directory/'result.json') != rec['evidence_sha256'] or file_sha(directory/'lease.json') != rec['lease_sha256']:
        raise ValueError('corrupt runtime or lease evidence')
    raw = json.loads((directory/'result.json').read_text())
    if digest(raw) != digest({key: value for key, value in rec.items() if key not in ('evidence_sha256', 'lease_sha256')}):
        raise ValueError('aggregate record differs from executed evidence')
    lease = json.loads((directory/'lease.json').read_text())
    lease = lease.get('lease', lease)
    if lease != rec['lease'] or lease.get('verdict') != 'clean' or rec['exit_code'] != 0:
        raise ValueError('GPU lease or process exit invalid')


def validate_controls(root, contract, controls):
    for variant in ('baseline', 'candidate'):
        rec = controls[variant]
        verify_evidence(root, rec)
        validate_record(rec, rec['plan'], contract_sha256=digest(contract),
            source_sha256=contract['variants'][variant]['source_sha256'],
            frames=contract['timing']['frames'], warmup=contract['timing']['warmup'], dt=.02)
        if rec['plan']['variant'] != variant or rec['plan']['phase'] != 'pilot':
            raise ValueError('pilot variant mismatch')
        if rec.get('force_oracle', {}).get('passed') is not True:
            raise ValueError('direct unequal-mass force oracle missing')
    validate_pair(controls['baseline'], controls['candidate'], contract['quality']['max_sample_relative_l2'])
    rec = controls['negative']
    verify_evidence(root, rec)
    if (rec.get('quality_failure') != 'nonfinite_positions' or rec.get('quality_status') != 'REJECTED'
            or rec.get('actual_backend') != 'cuda' or rec.get('negative_gpu_write') is not True
            or rec.get('contract_sha256') != digest(contract)
            or rec.get('source_sha256') != contract['variants']['candidate']['source_sha256']):
        raise ValueError('negative control failed for an unrelated reason')


def run_study(args):
    root = Path(args.output).resolve()
    contract = json.loads((root/'contract.json').read_text())
    if file_sha(__file__) != contract['verifier']['sha256'] or file_sha(MEASURER) != contract['verifier']['measurer_sha256']:
        raise ValueError('verifier changed after freeze')
    records_path = root/'records.json'
    records = json.loads(records_path.read_text()) if args.resume and records_path.exists() else []
    if records and not args.resume:
        raise ValueError('existing study requires explicit resume')
    if not (root/'controls.json').exists():
        controls = {}
        pilot_count = min(int(n) for n in contract['fixtures'])
        for variant in ('baseline', 'candidate'):
            row = dict(sequence=-1, count=pilot_count, phase='pilot', pair=-1, arm='a', variant=variant, seed=1007)
            rec = execute_one(root, contract, row, name='pilot-'+variant)
            validate_record(rec, row, contract_sha256=digest(contract),
                source_sha256=contract['variants'][variant]['source_sha256'],
                frames=contract['timing']['frames'], warmup=contract['timing']['warmup'], dt=.02)
            controls[variant] = rec
        validate_pair(controls['baseline'], controls['candidate'], contract['quality']['max_sample_relative_l2'])
        bad = dict(sequence=-1, count=pilot_count, phase='negative', pair=-1, arm='a', variant='candidate', seed=1007)
        rec = execute_one(root, contract, bad, name='negative-nonfinite', negative=True)
        if (rec.get('quality_failure') != 'nonfinite_positions' or rec.get('quality_status') != 'REJECTED'
                or rec.get('actual_backend') != 'cuda' or rec.get('negative_gpu_write') is not True):
            raise ValueError('negative control failed for an unrelated reason')
        controls['negative'] = rec
        validate_controls(root, contract, controls)
        write_json(root/'controls.json', controls)
    validate_controls(root, contract, json.loads((root/'controls.json').read_text()))
    for index, row in enumerate(contract['plan']):
        if index < len(records):
            rec = records[index]
        else:
            rec = execute_one(root, contract, row)
            records.append(rec)
            write_json(records_path, records)
        validate_record(rec, row, contract_sha256=digest(contract),
            source_sha256=contract['variants'][row['variant']]['source_sha256'],
            frames=contract['timing']['frames'], warmup=contract['timing']['warmup'], dt=.02)
        print(json.dumps(dict(done=index+1, total=len(contract['plan']), variant=row['variant'],
                              count=row['count'], median_ms=statistics.median(rec['measured_samples_ms']))), flush=True)
        if args.limit and index+1 >= args.limit:
            return
    analyze_study(root)


def analyze_study(root):
    root = Path(root)
    contract = json.loads((root/'contract.json').read_text())
    if file_sha(__file__) != contract['verifier']['sha256'] or file_sha(MEASURER) != contract['verifier']['measurer_sha256']:
        raise ValueError('verifier changed after freeze')
    for variant in contract['variants'].values():
        if source_sha(variant['root']) != variant['source_sha256']:
            raise ValueError('source artifact changed after freeze')
    records = json.loads((root/'records.json').read_text())
    if len(records) != len(contract['plan']):
        raise ValueError('missing trials; partial evidence cannot pass')
    validate_controls(root, contract, json.loads((root/'controls.json').read_text()))
    seen = set()
    pairs = {}
    last_end = None
    environment = None
    for row, rec in zip(contract['plan'], records):
        verify_evidence(root, rec)
        validate_record(rec, row, contract_sha256=digest(contract),
            source_sha256=contract['variants'][row['variant']]['source_sha256'],
            frames=contract['timing']['frames'], warmup=contract['timing']['warmup'], dt=.02)
        if rec['process_nonce'] in seen:
            raise ValueError('fresh process nonce reused')
        seen.add(rec['process_nonce'])
        if last_end is not None and rec['process_started_utc'] < last_end:
            raise ValueError('overlapping or reordered runtime trials')
        last_end = rec['process_finished_utc']
        observed = (rec['blender_binary'], rec['blender'], rec['warp'], rec['warp_path'], rec['device_name'])
        if environment is not None and observed != environment:
            raise ValueError('actual runtime environment changed')
        environment = observed
        if rec['storage']['storage'] != contract['variants'][row['variant']]['storage']:
            raise ValueError('actual tree storage mismatch')
        pairs.setdefault((row['count'], row['phase'], row['pair']), {})[row['arm']] = rec
    quality = []
    ratios = {}
    for (count, phase, pair), arms in pairs.items():
        a, b = arms['a'], arms['b']
        errors = validate_pair(a, b, contract['quality']['max_sample_relative_l2'])
        quality.append(dict(count=count, phase=phase, pair=pair, relative_l2=errors))
        ratios.setdefault((count, phase), []).append(statistics.median(a['measured_samples_ms'])/statistics.median(b['measured_samples_ms']))
    cases = {}
    for count in sorted(int(n) for n in contract['fixtures']):
        rows = [r for r in records if r['plan']['count'] == count and r['plan']['phase'] == 'ab']
        summary = classify(ratios[count, 'aa'], ratios[count, 'ab'])
        summary['trials'] = [dict(variant=r['plan']['variant'], pair=r['plan']['pair'],
            median_ms=statistics.median(r['measured_samples_ms']), p95_ms=p95(r['measured_samples_ms']),
            owned_device_array_bytes=r['owned_device_array_bytes'], nodes=r['allocated_node_capacity']) for r in rows]
        cases[str(count)] = summary
    report = dict(scope='completed CUDA solver; no viewport/presentation claim', comparison_kind='implementation',
                  contract_sha256=digest(contract), cases=cases, quality=quality,
                  all_trials_clean=True, raw_trials=len(records),
                  note='A slowdown is reported as a regression; memory savings do not imply faster playback.')
    write_json(root/'report.json', report)
    print(json.dumps({count: row['status'] for count, row in cases.items()}))
    return report


def main():
    argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    for key in ('baseline', 'candidate', 'baseline-revision', 'candidate-revision', 'dependency', 'cache', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--blender', default='5.2')
    p.add_argument('--storage', choices=['COMPACT', 'STRIDED'], default='COMPACT')
    p.add_argument('--counts', type=int, nargs='+', default=[100000, 400000])
    p.add_argument('--frames', type=int, default=120)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--pairs', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261117)
    p = commands.add_parser('run')
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    p = commands.add_parser('analyze')
    p.add_argument('--output', required=True)
    p = commands.add_parser('trial')
    p.add_argument('spec')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'run':
        run_study(args)
    elif args.command == 'analyze':
        analyze_study(args.output)
    else:
        trial(args.spec)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
