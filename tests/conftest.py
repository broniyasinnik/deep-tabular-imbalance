import pytest
from pathlib import Path


@pytest.fixture(name='adult')
def fixture_adult():
    root = Path.cwd().parent/'BenchmarkData'/'adult'
    return root
