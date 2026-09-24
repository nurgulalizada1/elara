import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("ELARA_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live API tests: set ELARA_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
