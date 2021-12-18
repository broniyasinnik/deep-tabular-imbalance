import pytest
from src.experiment_utils import load_config
from src.experiment_utils import ExperimentFactory


@pytest.fixture(name='config')
def fixture_config():
    conf = load_config('./experiment/config.yml')
    return conf


@pytest.fixture(name='factory')
def fixture_factory(config):
    factory = ExperimentFactory(config)
    return factory

