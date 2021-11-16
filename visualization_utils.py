import io
import cv2
import pandas as pd
import torch
import os
import numpy as np
import torch.nn as nn
from data_utils import load_arrays
from torch.utils.data import Dataset, TensorDataset
from absl import logging
from datasets import TableDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict
from torch.utils.tensorboard import SummaryWriter


def visualize_projection(name: str, save_to: str, **kwargs):
    available_projections = ['PCA', 'TSNE']
    if name not in available_projections:
        logging.info(f"Cannot project with {name}, try to use any of {available_projections}")
        return
    if name == 'PCA':
        pca = PCA(**kwargs['params'])
        X_fit, y_fit = load_arrays(kwargs['fit'])
        X_pca = pca.fit_transform(X_fit, y_fit)
        if 'transform' in kwargs:
            X_t, y_t = load_arrays(kwargs['transform'])
            X_t = pca.transform(X_t)
            X_pca, y = np.concatenate([X_pca, X_t]), np.concatenate([y_fit, y_t])
        save_2d_figure(X_pca, y, title=name, save_dir=save_to)
    elif name == 'TSNE':
        X, y = load_arrays(kwargs['fit'])
        tsne = TSNE(**kwargs['params'])
        X = tsne.fit_transform(X, y)
        save_2d_figure(X, y, title=name, save_dir=save_to)


def save_2d_figure(X: np.array, y: np.array, title: str, save_dir: str):
    fig = plt.figure(figsize=(5, 5))
    y_unique = np.unique(y)
    colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
    for this_y, color in zip(y_unique, colors):
        this_X = X[y == this_y]
        plt.scatter(this_X[:, 0], this_X[:, 1], s=20, c=color[np.newaxis, :],
                    alpha=0.5, edgecolor='k',
                    label="Class %s" % this_y.astype(int))
    plt.legend()
    plt.title(title)
    fig.savefig(os.path.join(save_dir, title))
    logging.info(f"Saved figure {title} to {save_dir}")


def projection(data: Dict[str, TableDataset], logdir: str, name: str = 'projection'):
    writer = SummaryWriter(os.path.join(logdir, name))
    features = torch.cat([torch.tensor(tb.data) for tb in data.values()])
    labels = [f'{name}:{int(label)}' for name, tb in data.items() for label in tb.target.tolist()]
    writer.add_embedding(features, metadata=labels)
    writer.close()


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    return img


def vis_histo_data(data):
    """
        Visualizes data as histogram
    """
    lims = (-5, 5)
    hist = np.histogram2d(data[:, 1], data[:, 0], bins=100, range=[lims, lims])
    plt.pcolormesh(hist[1], hist[2], hist[0], alpha=0.5)


def visualize_decision_boundary(dataset: Dataset, model: nn.Module):
    h = 0.25

    if type(dataset) is TensorDataset:
        X = dataset.tensors[0]
        y = dataset.tensors[1]
    else:
        X = dataset.data
        syn_X = dataset.synthetic_data if hasattr(dataset, "synthetic_data") else None
        syn_y = dataset.synthetic_target if hasattr(dataset, "synthetic_target") else None
        y = dataset.target

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    inputs = torch.tensor([list(xrow) for xrow in Xmesh]).float()
    model.eval()
    scores = torch.sigmoid(model(inputs))
    Z = np.array([s.data > 0.5 for s in scores])
    Z = Z.reshape(xx.shape)

    fig = plt.figure(figsize=(10, 10))
    plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    if syn_X is not None:
        plt.scatter(syn_X[:, 0], syn_X[:, 1], c=syn_y, s=40)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    return fig







