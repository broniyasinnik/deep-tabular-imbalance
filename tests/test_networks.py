import pytest
import torch
import torch.nn as nn
import numpy as np
from models.networks import SVDDLayer
from torch.utils.data import DataLoader
from models.transforms import ScalerTransform, Compose, ToTensor


def test_svdd_layer():
    data = np.load('../Adult/ir100/adult_ir100.tra.npz')
    X, y = data["X"], data["y"]
    minority = torch.tensor(X[y==1], dtype=torch.float32).unsqueeze(0)[:, :4, :]
    svdd = SVDDLayer(eps=1e-3)
    out = svdd(minority)
    dist_o = ((minority-out)**2).sum(dim=1).max()
    dist_m = ((minority-minority.mean(dim=1))**2).sum(dim=1).max()
    assert dist_o < dist_m
