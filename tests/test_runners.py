import pytest
from runners import MetaClassificationRunner


def test_meta_handle_batch(factory):
    experiment = factory.prepare_meta_experiment_with_smote(name='meta')
    runner = MetaClassificationRunner(dataset=experiment.loaders["train"].dataset, use_kde=False,
                                      use_armijo=True)
    runner.train(model=experiment.model,
                 loaders=experiment.loaders,
                 logdir='./data/logs',
                 num_epochs=experiment.epochs,
                 hparams=experiment.hparams,
                 valid_loader="valid",
                 valid_metric="ap",
                 verbose=False,
                 minimize_valid_metric=False,
                 )
