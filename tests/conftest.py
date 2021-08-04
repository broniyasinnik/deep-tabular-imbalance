import pytest
from experiment_utils import load_config
from ml_collections import ConfigDict
from pathlib import Path


@pytest.fixture(name='adult')
def fixture_adult():
    root = Path.cwd().parent/'BenchmarkData'/'adult'
    return root


@pytest.fixture(name='config')
def fixture_config():
    conf = load_config('./data/config.yml')
    return conf
