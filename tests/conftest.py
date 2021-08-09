import pytest
from experiment_utils import load_config
from experiment_utils import ExperimentFactory


@pytest.fixture(name='config')
def fixture_config():
    conf = load_config('./data/config.yml')
    return conf


@pytest.fixture(name='factory')
def fixture_factory(config):
    factory = ExperimentFactory(config)
    return factory
