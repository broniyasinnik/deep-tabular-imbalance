import pytest
from data_utils import TableConfig


def test_table_config():
    config = TableConfig(config_path='data/config.json', label='cardio')
    print(config.columns)


