import pytest
import torch
import numpy as np
from experiment_utils import get_model, get_train_data
from ml_collections import ConfigDict


@torch.no_grad()
def test_network_initial_loss():
    config_model = ConfigDict({
        "input_dim": 14,
        "hiddens": [128, 16]
    })
    train_file = "./data/adult_ir50.tst.npz"
    model = get_model(config_model)
    train_data = get_train_data(train_file=train_file)
    loss = 0.0
    ir = 50.0
    model.classifier[-1].bias.data.fill_(np.log(1 / ir))
    for row in train_data:
        features = row["features"]
        label = row["targets"]
        p = model(features)
        loss += torch.binary_cross_entropy_with_logits(p, label.reshape_as(p))

    loss_expected = -1. / ir * np.log(1. / ir) - (1. - 1. / ir) * np.log(1. - 1. / ir)
    loss = loss / len(train_data)
    assert np.isclose(loss_expected, loss.numpy(), rtol=1e-1)
