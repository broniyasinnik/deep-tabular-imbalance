import pytest
import numpy as np
from datasets import TableDataset, MultiTableDataset


def test_table_dataset(config):
    dataset = TableDataset.from_npz(config.train_file)
    print(dataset)


def test_multi_table_dataset():
    npz_dict = {
        "train1": "./data/adult.tra.npz",
        "train2": "./data/adult.val.npz",
        "train3": "./data/adult.synth.npz",
    }
    feat1, y1 = np.load(npz_dict["train1"]).values()
    feat2, y2 = np.load(npz_dict["train2"]).values()
    feat3, y3 = np.load(npz_dict["train3"]).values()
    dataset = MultiTableDataset.from_npz_files(npz_dict)
    print(dataset)
    print(dataset.ir)
    assert len(dataset) == max(feat1.shape[0],
                               feat2.shape[0],
                               feat3.shape[0])
    for i in range(len(dataset)):
        assert np.allclose(dataset[i]["train1"]["features"].numpy(), feat1[i % feat1.shape[0]])
        assert np.allclose(dataset[i]["train1"]["targets"].numpy(), y1[i % y1.shape[0]])
        assert np.allclose(dataset[i]["train2"]["features"].numpy(), feat2[i % feat2.shape[0]])
        assert np.allclose(dataset[i]["train2"]["targets"].numpy(), y2[i % y2.shape[0]])
        assert np.allclose(dataset[i]["train3"]["features"].numpy(), feat3[i % feat3.shape[0]])
        assert np.allclose(dataset[i]["train3"]["targets"].numpy(), y3[i % feat3.shape[0]])

# def test_synthetic_dataset(config):
#     dataset = TableSyntheticDataset(real_data=config.train_file,
#                                     synthetic_data=config.synthetic_file)
#
#     real_data = np.load(config.train_file)
#     synth_data = np.load(config.synthetic_file)
#     print(dataset)
#     # The synthetic dataset should have the same length of real data§
#     assert len(dataset) == synth_data['X'].shape[0]
#     real_targets_sum = 0.
#     for i in range(len(dataset)):
#         assert np.allclose(dataset[i]['features_z'], synth_data['X'][i])
#         real_targets_sum += dataset[i]["targets_x"]
#         # assert np.allclose(dataset[i]['features_x'], real_data['X'][i])
#
#     assert np.abs(real_targets_sum / len(dataset) - 0.5) <= 0.01
