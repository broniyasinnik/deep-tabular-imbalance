from datasets import TableDataset
from data_utils import TableConfig
from models.transforms import TableColumnsTransform

def test_table_column_transform():
    config = TableConfig('data/config.json')
    columns_t = TableColumnsTransform(config=config)
    dataset = ...


