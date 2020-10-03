import numpy as np
import time
import logging
from sdgym.evaluate import FeatureMaker
from sdgym.synthesizers import CTGANSynthesizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict


def label_column(data: np.array, metadata: Dict):
    columns = metadata['columns']
    for index, cinfo in enumerate(columns):
        col = data[:, index]
        if cinfo['name'] == 'label':
            labels = col.astype(int)
            return labels, index



def minority_data_undersample(data: np.array, meta: np.array, ratio: float):
    y, index = label_column(data, meta)
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
    return data[ids]

def majority_data_undersample(data: np.array, meta: np.array, ratio: float):
    y, index = label_column(data, meta)
    values, counts = np.unique(y, return_counts=True)
    majority_idx = np.argmax(counts)
    majority_value = values[majority_idx]
    majority_ids, = np.where(y == majority_value)
    minority_ids, = np.where(y != majority_value)
    num_majority_samples = int(np.floor(majority_ids.size * ratio))
    print(f"The number of undersampled majority:{num_majority_samples}")
    new_majority_ids = np.random.choice(majority_ids, num_majority_samples, replace=False)
    ids = np.concatenate([new_majority_ids, minority_ids])
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
