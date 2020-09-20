import numpy as np
from typing import Dict
from sdgym.evaluate import FeatureMaker
from sdgym.synthesizers import CTGANSynthesizer


def label_column(data: np.array, metadata: Dict):
    columns = metadata['columns']
    for index, cinfo in enumerate(columns):
        col = data[:, index]
        if cinfo['name'] == 'label':
            labels = col.astype(int)
            return labels


def prepare_test_train(train: np.array, test: np.array, meta: Dict):
    fm = FeatureMaker(meta)
    train_x, train_y = fm.make_features(train)
    test_x, test_y = fm.make_features(test)
    return train_x, train_y, test_x, test_y


def minority_data_undersample(data: np.array, meta: np.array, ratio: float):
    y = label_column(data, meta)
    values, counts = np.unique(y, return_counts=True)
    min_idx = np.argmin(counts)
    minority_value = values[min_idx]
    minority_ids, = np.where(y == minority_value)
    majority_ids, = np.where(y != minority_value)
    num_minority_samples = min(minority_ids.size,
                               int(np.floor(majority_ids.size * ratio)))
    new_minority_ids = np.random.choice(minority_ids, num_minority_samples, replace=False)
    ids = np.concatenate([majority_ids, new_minority_ids])
    return data[ids]


def ctgan_syntesize(data, categorical_columns, ordinal_columns):
    synthesizer = CTGANSynthesizer(epochs=1)
    syntetic_data = synthesizer.fit_sample(data, categorical_columns, ordinal_columns)
    return syntetic_data
