"""Package the measured 0.2.0 candidate with source and retained evidence."""
import argparse,hashlib,json,pathlib,subprocess,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
SHA='49a74dae199957aa8090d2f20eea70689a238a2d3979d456d475f4b44029ef90'
INSTALLER=ROOT/'artifacts/astro_collision_lab-0.2.0-windows-x64.zip'
POSITIVE='live-galaxy-9fadc87f'
NEGATIVE='live-galaxy-37bc4445'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main(bundle=False):
    batch=read(ROOT/'status/qualification-v2-batch.json')
    assert batch['phase']=='finished' and batch['pass'] and len(batch['runs'])==6,'Final batch is not qualified'
    assert hashlib.sha256(INSTALLER.read_bytes()).hexdigest()==SHA,'Candidate differs'
    with zipfile.ZipFile(INSTALLER) as z:
        assert not any(n.lower().endswith(('.whl','.dll','.pyd')) for n in z.namelist())
        manifest=json.loads(z.read('_build.json'))
        for name,digest in manifest['files'].items():
            assert hashlib.sha256((ROOT/'extension/astro_collision_lab'/name).read_bytes()).hexdigest()==digest,name
    positive=read(ROOT/'status/proofs'/POSITIVE/'live_100k.json')
    negative=read(ROOT/'status/proofs'/NEGATIVE/'live_100k.json')
    assert positive['pass'] and not negative['pass']
    assert negative['checks']['Known control payload reached renderer']
    assert not negative['checks']['Float64 authored points render at expected projected position']
    native=read(ROOT/'status/native-authoring-v2.json');assert native['pass'] and len(native['checks'])==25
    unit=(ROOT/'status/unit-final-0.2.0.log').read_text(encoding='utf-16')
    assert '66 passed, 10 subtests passed' in unit
    for family in ('DISK','DWARF','ELLIPTICAL','LENTICULAR'):
        model=read(ROOT/('status/isolation-v2-'+family+'.json'));assert model['pass']
        assert all(g['pass'] and g['active_count']==50000 and g['periods']>=5 for g in model['groups'].values())
    lines=['# Astro Collision Lab 0.2.0 qualification','',
        'Windows RTX 4080 SUPER; Blender 5.1.0 (adfe2921d5f3); certified Warp 1.15.0. Measured using installed OpenGL windows and isolated project profiles. No production Blender preferences changed.','',
        f'Installer SHA-256: `{SHA}`. Size: {INSTALLER.stat().st_size} bytes. No dependency wheels or binaries are bundled.','',
        '| Mode | Trial | Fresh completed FPS | p95 cadence (ms) | p95 step to GL finish (ms) | Frames |','|---|---:|---:|---:|---:|---:|']
    selected=[]
    for item in batch['runs']:
        p=pathlib.Path(item['report']);d=read(p);runner=read(p.parent/'runner.json')
        assert runner['candidate'].lower()==SHA and d['pass'] and all(d['checklist'].values())
        assert d['usable_rows']==2700 and d['end_to_end_complete_fps']>=24
        assert d['p95_draw_interval_ms']<=1000/24 and d['p95_step_to_glfinish_ms']<=1000/24
        lines.append(f"| {item['mode']} | {item['trial']} | {d['end_to_end_complete_fps']:.2f} | {d['p95_draw_interval_ms']:.2f} | {d['p95_step_to_glfinish_ms']:.2f} | {d['usable_rows']} |")
        selected.append(p.parent)
    edits=read(pathlib.Path(batch['edits']['report']));assert edits['pass'] and all(edits['edit_checks'].values())
    selected.extend([pathlib.Path(batch['edits']['report']).parent,ROOT/'status/proofs'/POSITIVE,ROOT/'status/proofs'/NEGATIVE])
    lines+=['','Every measured live state contained exactly 100000 active and submitted particles. Distinct revisions and physical progress were required. Cadence and completed-step timing both passed the 41.67 ms p95 gate. These are measured windows, not a guarantee of every frame meeting budget.','',
        'Live-edit probe: '+json.dumps(edits['edit_checks'],sort_keys=True)+'.','',
        'Unit tests: 66 passed plus 10 subtests. Background native Blender: 25 checks. Real-window native authoring: 14 checks. The disposable broken-format control failed specifically at projected-point placement.','',
        '## Isolated galaxy families','',
        'Each default family passed five reference orbital periods at its actual 50000-particle allocation using the encounter solver and authored settings. Component radius/height, total conservation, finite/bound-state and overflow gates remain enforced for custom inputs. Reports: status/isolation-v2-{DISK,DWARF,ELLIPTICAL,LENTICULAR}.json.','',
        '## Limits and retained failures','',
        'The earlier candidate batch remains in status/qualification-v2-before-preview-format.json, including a failed 50.65 ms galaxy p95 cadence trial. Final results above are a separate batch on the exact final archive. Failed initializer candidates and the framebuffer verifier correction are documented in IMPLEMENTATION_0_2_0.md.','',
        'Engineering isolation checks use sampled energy and spherical binding diagnostics. Scientific convergence, arbitrary custom-model stability, scientific Moon formation, SpaceSim speed parity and Linux GPU certification remain unqualified. The older source-lineage reconciliation remains a separate documented open item.','',
        'Open launch-lab.cmd to use the qualified Windows Blender in the project-owned profile. Start runs the unguided simulation; physical edits rebuild and resume, appearance edits preserve its trajectory.']
    (ROOT/'docs/QUALIFICATION_0_2_0.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not bundle:return
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=[]
    for directory in ('extension','tools','tests','validation','docs'):
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() not in ('.pyc','.whl','.dll','.pyd') and 'wheels' not in p.parts)
    files.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in ('.md','.toml','.txt','.cmd','.ini'))
    files.extend([ROOT/'.gitignore',INSTALLER,INSTALLER.with_suffix('.build.json')])
    for pattern in ('isolation-v2-*.json','native-authoring-v2.json','native-galaxy-snapshot-v2.json','qualification-v2*.json','qualification-v2*.log','unit-final-0.2.0.log','native-authoring-final-0.2.0.log','native-gui-v2-readback-correct.log','native-gui-v2-format-negative-final.log'):
        files.extend((ROOT/'status').glob(pattern))
    old=read(ROOT/'status/qualification-v2-before-preview-format.json')
    selected.extend(pathlib.Path(r['report']).parent for r in old.get('runs',[]) if r.get('report'))
    if old.get('edits',{}).get('report'):selected.append(pathlib.Path(old['edits']['report']).parent)
    for folder in selected:
        files.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.log'))
    out=ROOT/'artifacts/astro_collision_lab-0.2.0-source-and-evidence.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),(1980,1,1,0,0,0));info.external_attr=0o644<<16
            z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('SOURCE_COMMIT.txt',commit+'\nThe pre-existing original live harness working changes are retained in this bundle; see the repository working status.\n')
    print(json.dumps({'source_archive':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'commit':commit}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',action='store_true');main(p.parse_args().bundle)
