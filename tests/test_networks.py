import pytest
import torch
from models.networks import TabularModel
from torch.utils.data import DataLoader


def test_tabular_model(adult):
    tab_model = TabularModel(adult.embeds, adult.num_continuous,
                             out_sz=128, layers=[64, 128])
    loader = DataLoader(adult, batch_size=10)
    batch = next(iter(loader))
    result = tab_model(batch[0].type(torch.long), batch[1])
    assert True
