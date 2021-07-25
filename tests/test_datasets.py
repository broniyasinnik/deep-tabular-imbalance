import pytest
from datasets import SmoteDataset
from datasets import TableDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from catalyst.data.sampler import BalanceClassSampler



def test_smote_dataset():
    train_data = SmoteDataset(data='../Keel1/glass4/glass4.tra.npz',
                          smote_data='../Keel1/glass4/glass4-smt.tra.npz')
    loader = DataLoader(train_data, batch_size=20, shuffle=True)
    # data = ConcatDataset([train1, smote])
    pass

