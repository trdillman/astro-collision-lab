import pytest


def pytest_collection_modifyitems(items):
    here = __file__.rsplit("conftest.py", 1)[0]
    for item in items:
        if str(item.path).startswith(here):
            item.add_marker(pytest.mark.validation)
