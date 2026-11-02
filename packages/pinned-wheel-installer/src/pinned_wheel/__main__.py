"""``python -m pinned_wheel PROJECT VERSION FILENAME SHA256 DESTINATION``"""
import argparse

from . import PinnedWheel, install


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pinned_wheel", description=__doc__)
    for name in ("project", "version", "filename", "sha256", "destination"):
        parser.add_argument(name)
    a = parser.parse_args(argv)
    target = install(PinnedWheel(a.project, a.version, a.filename, a.sha256), a.destination)
    print(f"Installed {a.project} {a.version} into {target}")


if __name__ == "__main__":
    main()
