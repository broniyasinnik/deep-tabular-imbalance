import pytest
import torch
import torch.nn as nn
from models.networks import TabularModel, EmbeddingLayer
from models.networks import MLP, HyperNetwork
from catalyst.dl import utils
from torch.utils.data import DataLoader
from models.transforms import ScalerTransform, Compose, ToTensor

def test_tabular_model(adult):
    tab_model = TabularModel(adult.embeds, adult.num_continuous,
                             out_sz=128, layers=[64, 128])
    loader = DataLoader(adult, batch_size=10)
    batch = next(iter(loader))
    result = tab_model(batch[0].type(torch.long), batch[1])
    assert True

def test_embedding_layer(adult):
    scaler_transform = ScalerTransform(adult.data, features=adult.continuous_cols)
    to_tensor = ToTensor()
    feat_transform = Compose([scaler_transform, to_tensor])
    adult.transform = feat_transform
    x = adult[0][0].reshape(1, -1)
    layer = EmbeddingLayer(adult.categorical_cols,
                           adult.continuous_cols,
                           adult.embeds)
    embedded_x = layer(x)
    assert True

def test_hyper_network():
    hypernet = HyperNetwork()
    input = torch.randn((2, 55))
    label = torch.ones(1)
    loss = (hypernet(input, label)-label)
    assert True
