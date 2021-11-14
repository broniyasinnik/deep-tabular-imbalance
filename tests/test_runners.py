import pytest
from runners import MetaClassificationRunner


def test_meta_handle_batch(factory):
    experiment = factory.prepare_experiment(name='meta')
    runner = MetaClassificationRunner()
    runner.train(model=experiment.model,
                 loaders=experiment.loaders,
                 logdir='./experiment/models/logs',
                 num_epochs=experiment.epochs,
                 hparams=experiment.hparams,
                 valid_loader="valid",
                 valid_metric="ap",
                 verbose=False,
                 minimize_valid_metric=False,
                 )

