import pytest
import numpy as np
from datasets import TableSyntheticDataset, TableDataset


def test_table_dataset(config):
    dataset = TableDataset.from_npz(config.train_file)
    print(dataset)


def test_synthetic_dataset(config):
    dataset = TableSyntheticDataset(real_data=config.train_file,
                                    synthetic_data=config.synthetic_file)

    real_data = np.load(config.train_file)
    synth_data = np.load(config.synthetic_file)
    print(dataset)
    # The synthetic dataset should have the same length of real data§
    assert len(dataset) == synth_data['X'].shape[0]
    real_targets_sum = 0.
    for i in range(len(dataset)):
        assert np.allclose(dataset[i]['features_z'], synth_data['X'][i])
        real_targets_sum += dataset[i]["targets_x"]
        # assert np.allclose(dataset[i]['features_x'], real_data['X'][i])

    assert np.abs(real_targets_sum / len(dataset) - 0.5) <= 0.01
