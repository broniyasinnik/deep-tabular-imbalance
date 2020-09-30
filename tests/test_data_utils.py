import numpy as np
import pytest
from data_utils import minority_data_undersample, label_column
from data_utils import ctgan_syntesize


@pytest.mark.parametrize("ratio", [0.1, 0.01, 0.001])
def test_minority_data_undersample(ratio, adult):
    train, _, meta, _, _ = adult
    y_train, index = label_column(train, meta)
    _, counts = np.unique(y_train, return_counts=True)
    minority_count = np.min(counts)
    majority_count = np.sum(counts) - minority_count
    data_sample = minority_data_undersample(train, meta, ratio)
    expected_size = majority_count + min(minority_count,
                                         np.floor(majority_count * ratio))
    assert data_sample.shape[0] == expected_size


def test_ctgan_syntesize(adult):
    data, _, _, cat, ord = adult
    synesized = ctgan_syntesize(data, cat, ord)
    assert synesized.shape[0] == data.shape[0]