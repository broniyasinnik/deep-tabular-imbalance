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

def test_dataset_imbalanced(adult):
    imb_adult = DatasetImbalanced(num_minority=1000)(adult)
    assert imb_adult.targets.sum() == 1000
