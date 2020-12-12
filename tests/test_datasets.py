import pytest
from torch.utils.data import DataLoader
from datasets import MoonsDataset, DatasetSSL, DatasetImbalanced
from catalyst import data


def test_moons():
    dataset = MoonsDataset()
    sampler = data.BalanceClassSampler(labels=dataset.get_labels())
    train_loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=20)
    assert True


def test_shuttle_scale_features(shuttle):
    features_to_scale = list(range(shuttle.train_x.shape[1]))
    shuttle.scale_features(features_to_scale, scale_type='standard')
    assert True


def test_dataset_ssl(adult):
    dataset = DatasetSSL(adult)
    dataset.split_to_labeled_unlabeled(100)
    assert True


@pytest.mark.parametrize("ratio", [None, 1, 0.1, 0.7])
def test_dataset_imbalanced(adult, ratio):
    imb_adult = DatasetImbalanced(imbalance_ratio=ratio)(adult)
    assert hasattr(imb_adult, "pos_weight")
    assert hasattr(imb_adult, "neg_weight")
    assert hasattr(imb_adult, "num_minority")
    assert hasattr(imb_adult, "num_majority")
    assert imb_adult.target.sum() == imb_adult.num_minority
    if ratio is not None:
        assert abs(imb_adult.num_minority / imb_adult.num_majority - ratio) < 0.01
