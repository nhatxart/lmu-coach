import sys
from pathlib import Path

import pytest

# Make `import app.main` work when pytest is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run tests marked slow (e.g. T6 over all nine seeds)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long runs, skipped unless --runslow or -m slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") or "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow: run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
