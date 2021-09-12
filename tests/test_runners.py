import pytest
from runners import MetaClassificationRunner
import torch.nn.functional as F
from torch.autograd import grad
from runners import armijo_step_model


def test_meta_handle_batch(factory):
    experiment = factory.prepare_meta_experiment(name='meta')
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


def test_armijo_step_model(factory):
    experiment = factory.prepare_meta_experiment(name='meta')
    model = experiment.model
    batch = next(iter(experiment.loaders['train']))
    x = batch['features']
    y = batch['target']
    p = model(x)
    loss = F.binary_cross_entropy_with_logits(p, y.reshape_as(p))
    gradients = grad(loss, model.parameters())
    armijo_step_model(model, gradients, 10, x, y)
    assert False
