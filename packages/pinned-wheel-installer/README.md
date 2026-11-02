# pinned-wheel-installer (`pinned_wheel`)

Install one exact version of a dependency into a private folder, checked by SHA-256, without
pip. Made for apps with embedded Python, such as Blender add-ons, that need a heavy package
(for example NVIDIA Warp) at an exact version but must not bundle the wheel or change the
host's own packages. Generalized from the Astro Collision Lab add-on's Warp installer.
Standard library only.

What it checks:

- The wheel's SHA-256 matches your pin, and PyPI's published hash matches it too.
- No file in the wheel can land outside the target folder.
- After install, the module that actually imports is the pinned version. If something else
  in the process already loaded a different version, it says so instead of mixing them.

## Install

```
pip install "pinned-wheel-installer @ git+https://github.com/trdillman/astro-collision-lab#subdirectory=packages/pinned-wheel-installer"
```

It is one small file (`src/pinned_wheel/__init__.py`), so copying it into an add-on is fine
too.

## Use

```python
import pinned_wheel as pw

WARP = pw.PinnedWheel(
    project="warp-lang", version="1.15.0",
    filename="warp_lang-1.15.0-py3-none-win_amd64.whl",
    sha256="06ef42c3ce522749268056653ea253645eb4a96ca164af74ae5113f0d55a6970",
    module="warp",                      # defaults to the project name before the first "-"
)

pw.install(WARP, deps_dir)              # download from PyPI, verify, unpack
pw.activate(WARP, deps_dir, required=True)   # add to sys.path and check the version
import warp
```

From a shell: `python -m pinned_wheel warp-lang 1.15.0 <filename> <sha256> <folder>`.

### In a Blender add-on

Run the download in a separate background Blender so the UI never freezes, with every
Blender user folder pointed at a scratch directory so the user's own add-ons can't interfere:

```python
import subprocess, tempfile, bpy
argv, env = pw.isolated_blender_command(bpy.app.binary_path, install_script, [deps_dir],
                                        scratch=tempfile.mkdtemp())
process = subprocess.Popen(argv, env=env)   # poll it from bpy.app.timers
```

where `install_script` calls `pw.install(...)` with the folder passed after `--`.

## Test

```
pip install -e "packages/pinned-wheel-installer[test]"
pytest packages/pinned-wheel-installer/tests
```

Tests use a fake package index; nothing is downloaded.
