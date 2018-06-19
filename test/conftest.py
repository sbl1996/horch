import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--torch", action="store_true", default=False, help="run tests depend on pytorch"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--torch"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_torch = pytest.mark.skip(reason="need --torch option to run")
    for item in items:
        if "torch" in item.keywords:
            item.add_marker(skip_torch)