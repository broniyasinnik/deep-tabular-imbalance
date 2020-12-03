import numpy as np
import time
import logging
from category_encoders import LeaveOneOutEncoder
from sdgym.constants import CATEGORICAL, CONTINUOUS, ORDINAL
from sdgym.synthesizers import CTGANSynthesizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from typing import Dict




class CatBoostFeatureMaker:

    def __init__(self, metadata, cat_cols, ord_cols, sample=50000):
        self.columns = metadata['columns']
        self.sample = sample
        self.cat_encoder = LeaveOneOutEncoder()
        self.cont_encoder = MinMaxScaler()
        self.label_col = next(filter(lambda i: self.columns[i]['name'] == 'label',
                                     range(len(self.columns))))
        self.cat_cols = cat_cols
        if self.label_col in self.cat_cols:
            self.cat_cols.remove(self.label_col)

        self.ord_cols = ord_cols
        if self.label_col in self.ord_cols:
            self.ord_cols.remove(self.label_col)

        self.cont_cols = list(filter(lambda i: self.columns[i]['type'] == 'continuous',
                                     range(len(self.columns))))

    def fit(self, data):
        data = data.copy()
        np.random.shuffle(data)
        data = data[:self.sample]
        labels = data[:, self.label_col].astype(float)
        data = np.delete(data, self.label_col, axis=1)
        self.cat_encoder.fit(data[:, self.cat_cols], labels)
        self.cont_encoder.fit(data[:, self.cont_cols])

    def transform(self, data):
        data = data.copy()
        labels = data[:, self.label_col].astype(int)
        data[:, self.cat_cols] = self.cat_encoder.transform(data[:, self.cat_cols])
        data[:, self.cont_cols] = self.cont_encoder.transform(data[:, self.cont_cols])
        data = np.delete(data, self.label_col, axis=1)
        return data, labels


def label_column(data: np.array, metadata: Dict):
    columns = metadata['columns']
    for index, cinfo in enumerate(columns):
        col = data[:, index]
        if cinfo['name'] == 'label':
            labels = col.astype(int)
            return labels, index


def minority_data_undersample(X: np.array, y: np.array, ratio: float):
    values, counts = np.unique(y, return_counts=True)
    min_idx = np.argmin(counts)
    minority_value = values[min_idx]
    minority_ids, = np.where(y == minority_value)
    majority_ids, = np.where(y != minority_value)
    num_minority_samples = min(minority_ids.size,
                               int(np.floor(majority_ids.size * ratio)))
    print(f"The number of undersampled minority:{num_minority_samples}")
    new_minority_ids = np.random.choice(minority_ids, num_minority_samples, replace=False)
    ids = np.concatenate([majority_ids, new_minority_ids])
    return X[ids], y[ids]


def data_undersample(data: np.array, meta: np.array, ir_ratio: float, random_state=None):
    assert 0 <= ir_ratio <= 1, "ir should be between [0,1]"
    if random_state:
        np.random.seed(random_state)
    y, index = label_column(data, meta)
    values, counts = np.unique(y, return_counts=True)
    majority_idx = np.argmax(counts)
    majority_value = values[majority_idx]
    majority_ids, = np.where(y == majority_value)
    minority_ids, = np.where(y != majority_value)
    ir = minority_ids.size / majority_ids.size
    if (ir_ratio <= ir):
        num_samples = int(np.floor(minority_ids.size * ir_ratio / ir))
        new_minority_ids = np.random.choice(minority_ids, num_samples, replace=False)
        ids = np.concatenate([majority_ids, new_minority_ids])
    else:
        num_samples = int(np.floor(majority_ids.size * ir / ir_ratio))
        new_majority_ids = np.random.choice(majority_ids, num_samples, replace=False)
        ids = np.concatenate([new_majority_ids, minority_ids])

    np.random.shuffle(ids)
    return data[ids]


def ctgan_syntesize(data, categorical_columns, ordinal_columns):
    synthesizer = CTGANSynthesizer(epochs=300)
    syntetic_data = synthesizer.fit_sample(data, categorical_columns, ordinal_columns)
    return syntetic_data


def t_sne(data, pca_comp=0):
    if pca_comp:
        pca = PCA(n_components=pca_comp)
        pca_result = pca.fit_transform(data)
        data = pca_result
        logging.info('Cumulative explained variation for 50 principal components: {}'.format(
            np.sum(pca.explained_variance_ratio_)))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    logging.info('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    return tsne_results
