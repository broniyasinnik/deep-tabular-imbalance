import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution


class GaussianKDE(Distribution):
    def __init__(self, X, bw):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims),
                                      covariance_matrix=torch.eye(self.dims))

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X
        log_probs = torch.log(
            (self.bw ** (-self.dims) *
             torch.exp(self.mvn.log_prob(
                 (X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n)

        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated

        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        X_chunks = self.X.split(1000)
        Y_chunks = Y.split(1000)

        log_prob = 0

        for x in X_chunks:
            for y in Y_chunks:
                log_prob += self.score_samples(y, x).sum(dim=0)

        return log_prob


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_layers: List[int]):
        super(MLP, self).__init__()
        self.layers = []
        hidden_layers.insert(0, in_features)
        hidden_layers.append(out_features)
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if i != len(hidden_layers) - 1:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm1d(hidden_layers[i]))
        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.mlp(x)


class TabularModel(nn.Module):
    """Basic model for tabular data."""

    def __init__(self, cat_feats: List[int], cont_feats: List[int],
                 emb_szs: List[Tuple[int, int]], out_sz, layers, embed_p=0.):
        super(TabularModel, self).__init__()
        n_cont = len(cont_feats)
        self.cat_feats = cat_feats
        self.cont_feats = cont_feats
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont = n_emb, n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]
        _layers = []
        for i in range(len(sizes) - 1):
            _layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh()]
        _layers.pop(-1)
        self.layers = nn.Sequential(*_layers)

    def forward(self, x):
        x_cat = x[:, self.cat_feats].type(torch.long)
        x_cont = x[:, self.cont_feats]
        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)


class EmbeddingLayer(nn.Module):
    def __init__(self, categorical_features: List[int],
                 continuous_features: List[int],
                 embedding_dims: List[Tuple[int, int]]):
        super(EmbeddingLayer, self).__init__()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.embeddings = nn.ModuleList([nn.Embedding(input_dim, output_dim)
                                         for input_dim, output_dim in embedding_dims])

    def forward(self, x):
        x_cat = x[:, self.categorical_features].type(torch.long)
        x_cont = x[:, self.continuous_features]
        out = torch.cat([embed(x_cat[:, i]) for i, embed, in enumerate(self.embeddings)], dim=1)
        out = torch.cat([out, x_cont], dim=1)
        return out


class Generator(nn.Module):

    def __init__(self, noise_dim: int, hidden_dim: int, out_dim: int):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        layers = [
            nn.Linear(self.noise_dim + 1, hidden_dim),  # First layer includes also the condition dimension
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        ]
        self.generator = nn.Sequential(*layers)

    def forward(self, batch):
        result = self.generator(batch)
        return result

    def sample_latent_vectors(self, batch_size: int, labels=None):
        # Sample random points in the latent space
        latent_vectors = torch.randn(batch_size, self.noise_dim)
        if labels is None:
            p = 1 / 2 * torch.ones((batch_size, 1))
            labels = torch.bernoulli(p)
        assert labels.size().numel() == batch_size
        random_latent_vectors = torch.cat([latent_vectors, labels], dim=1)
        return random_latent_vectors, labels
