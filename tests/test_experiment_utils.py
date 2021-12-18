from src.experiment_utils import ExperimentFactory


def test_prepare_experiment(config):
    factory = ExperimentFactory(config)
    experiment = factory.prepare_experiment('base')
    batch = next(iter(experiment.loaders["train"]))
    pass

