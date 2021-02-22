import pytest
from pathlib import Path
from datasets import ShuttleDataset, AdultDataSet


@pytest.fixture(name='adult')
def fixture_adult():
    root = Path.cwd().parent/'BenchmarkData'/'adult'
    adult = AdultDataSet(root, train=True)
    return adult
