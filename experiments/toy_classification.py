import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rv_discrete, bernoulli
from torch.utils.data import TensorDataset, ConcatDataset
import matplotlib.pyplot as plt

np.random.seed(42)
lims = (-5, 5)

MEANS = np.array(
    [[-1, -4],
     [2, 3],
     [-3, 0],
     ])

COVS = np.array(
    [[[1, 0.8], [0.8, 1]],
     [[1, -0.5], [-0.5, 1]],
     [[1, 0], [0, 1]],
     ])
PROBS = np.array([
    0.2,
    0.5,
    0.3
])

MEANS1 = np.array([
    [0, 0],
    # [-2, -2]
])

COVS1 = np.array([
    [[0.1, 0], [0, 0.4]],
    # [[0.1, 0], [0, 0.1]]
])

PROBS1 = np.array([
    1,
    # 0.5
])


def sample(n, means, covs, probs):
    assert len(means) == len(covs) == len(probs), "number of components mismatch"
    components = len(means)
    comps_dist = rv_discrete(values=(range(components), probs))
    comps = comps_dist.rvs(size=n)
    conds = np.arange(components)[:, None] == comps[None, :]
    arr = np.array([np.random.multivariate_normal(means[c], covs[c], size=n)
                    for c in range(components)])
    return np.select(conds[:, :, None], arr).astype(np.float32)


def vis_histo_data(data):
    """
        Visualizes data as histogram
    """
    hist = np.histogram2d(data[:, 1], data[:, 0], bins=100, range=[lims, lims])
    plt.pcolormesh(hist[1], hist[2], hist[0], alpha=0.5)


def vis_data(majority, minority):
    plt.scatter(majority[:, 0], majority[:, 1], c='b')
    plt.scatter(minority[:, 0], minority[:, 1], c='r', alpha=0.5)
    plt.show()


def get_dataset(majority=1000, minority=20):
    data_majority = sample(majority, means=MEANS, covs=COVS, probs=PROBS)
    data_minority = sample(minority, means=MEANS1, covs=COVS1, probs=PROBS1)
    tensors_majority = TensorDataset(torch.tensor(data_majority), torch.zeros(majority, 1))
    tensors_minority = TensorDataset(torch.tensor(data_minority), torch.ones(minority, 1))
    dataset = ConcatDataset([tensors_majority, tensors_minority])
    return dataset


def get_classifier():
    classifier = nn.Sequential(nn.Linear(2, 64),
                               nn.ReLU(),
                               nn.Linear(64, 128),
                               nn.ReLU(),
                               nn.Linear(128, 2))
    return classifier




