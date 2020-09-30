import pytest
from sdgym import load_dataset
from sdgym.evaluate import FeatureMaker

@pytest.fixture(name='adult')
def fixture_adult():
    train, test, meta, cat, ord = load_dataset('adult', benchmark=True)
    return train, test, meta, cat, ord


@pytest.fixture(name='adult_train_test')
def fixture_adult_train_test(adult):
    train, test, meta, _, _ = adult
    fm = FeatureMaker(meta)
    x_train, y_train = fm.make_features(train)
    x_test, y_test = fm.make_features(test)
    return x_train, y_train, y_train, y_test


