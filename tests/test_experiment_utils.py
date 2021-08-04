import pytest
from experiment_utils import ExperimentFactory

def test_prepare_experiment(config):
    factory = ExperimentFactory(config)
    factory.prepare_experiment('downsampling')
    assert False
