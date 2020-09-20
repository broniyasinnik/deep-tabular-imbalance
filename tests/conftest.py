import pytest
from sdgym import load_dataset
from data_utils import prepare_test_train


@pytest.fixture(name='adult')
def fixture_adult():
    train, test, meta, cat, ord = load_dataset('adult', benchmark=True)
    return train, test, meta, cat, ord


@pytest.fixture(name='adult_train_test')
def fixture_adult_train_test(adult):
    train, test, meta, _, _ = adult
    x_train, y_train, x_test, y_test = prepare_test_train(train, test, meta)
    return x_train, y_train, y_train, y_test


