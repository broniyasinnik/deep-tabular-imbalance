import pytest
from pathlib import Path
from datasets import ShuttleDataset, AdultDataSet
from models.transforms import ToTensor
from sdgym import load_dataset
from sdgym.evaluate import FeatureMaker

@pytest.fixture(name='data')
def fixture_data(request):
    marker = request.node.get_closest_marker("table")
    table_name = marker.args[0]
    train, test, meta, cat, ord = load_dataset(table_name, benchmark=True)
    return train, test, meta, cat, ord


@pytest.fixture(name='train_test')
def fixture_train_test(data):
    train, test, meta, _, _ = data
    fm = FeatureMaker(meta, sample=train.shape[0])
    x_train, y_train = fm.make_features(train)
    x_test, y_test = fm.make_features(test)
    return x_train, y_train, x_test, y_test


@pytest.fixture(name='shuttle')
def fixture_shuttle():
    shuttle = ShuttleDataset()
    features_to_scale = list(range(shuttle.num_features))
    shuttle.scale_features(features_to_scale, scale_type='standard')
    return shuttle

@pytest.fixture(name='adult')
def fixture_adult():
    root = Path.cwd()/'BenchmarkData/adult'
    adult = AdultDataSet(root, train=True,
                         transform=ToTensor(), target_transform=ToTensor())
    return adult
