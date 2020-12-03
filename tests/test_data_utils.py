import numpy as np
import pytest
from data_utils import minority_data_undersample, label_column, data_undersample
from data_utils import ctgan_syntesize
from data_utils import CatBoostFeatureMaker


@pytest.mark.parametrize("ratio", [0.1, 0.01, 0.001])
def test_minority_data_undersample(ratio, adult_train_test):
    x_train, y_train, _, _ = adult_train_test
    _, counts = np.unique(y_train, return_counts=True)
    minority_count = np.min(counts)
    majority_count = np.sum(counts) - minority_count
    x_sample, y_sample = minority_data_undersample(x_train, y_train, ratio)
    expected_size = majority_count + min(minority_count,
                                         np.floor(majority_count * ratio))
    assert x_sample.shape[0] == expected_size
    assert y_sample.shape[0] == expected_size


@pytest.mark.table("credit")
def test_ctgan_syntesize(data):
    data, _, _, cat, ord = data
    synesized = ctgan_syntesize(data, cat, ord)
    assert synesized.shape[0] == data.shape[0]


@pytest.mark.table("adult")
@pytest.mark.parametrize("ir", [1, 0.5, 0.1])
def test_data_undersample(ir, data):
    train, _, meta, _, _ = data
    new_train = data_undersample(train, meta, ir)
    y_train, index = label_column(new_train, meta)
    _, counts = np.unique(y_train, return_counts=True)
    minority_count = np.min(counts)
    majority_count = np.sum(counts) - minority_count
    expected_ir = minority_count / majority_count
    assert np.abs(expected_ir - ir) <= 1e-2


@pytest.mark.table("adult")
def test_make_catboost_features(data):
    train, test, meta, cat, ord = data
    fm = CatBoostFeatureMaker(meta, cat, ord)
    fm.fit(train)
    features, labels = fm.transform(train)
    assert features.shape[1] == train.shape[1]-1
