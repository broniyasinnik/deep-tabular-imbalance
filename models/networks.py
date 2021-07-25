import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from torch import distributions
from copy import deepcopy
from qpth.qp import QPFunction
from typing import List, Tuple

import torch
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
            (self.bw**(-self.dims) *
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

class CentersDistance(nn.Module):
    def __init__(self, dim=-1):
        super(CentersDistance, self).__init__()
        self._dim = dim

    def forward(self, inputs, centers):
        logits = -torch.sum((centers.unsqueeze(1) - inputs)**2, dim=self._dim)
        # if `keepdim=True` up there, remove unsqueeze here
        return logits

class SVDDLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-6):
        super(SVDDLayer, self).__init__()
        self._dim = dim
        self._eps = eps

    def forward(self, inputs):
        shot = inputs.shape[1]

        kernel_matrices = torch.bmm(inputs, inputs.transpose(1, 2))
        kernel_matrices += self._eps * torch.eye(shot)
        kernel_diags = torch.diagonal(kernel_matrices, dim1=-2, dim2=-1)
        Q = 2 * kernel_matrices
        p = -kernel_diags
        A = torch.ones(1, shot)
        b = torch.ones(1)
        G = -torch.eye(shot)
        h = torch.zeros(shot)
        alphas = QPFunction(verbose=True)(
            Q,
            p,
            G.detach(),
            h.detach(),
            A.detach(),
            b.detach(),
        )

        alphas = alphas.unsqueeze(-1)
        centers = torch.sum(alphas * inputs, dim=self._dim)
        # `keepdim=True` here to avoid unsqueezing in `CentersDistance`, which
        # could be used for vanilla protonet?

        return centers


class PrototypeLayer(nn.Module):
    def __init__(self, dim=1):
        super(PrototypeLayer, self).__init__()
        self._dim = dim
        # TODO: does this work for vanilla prototypical networks?

    def forward(self, inputs):
        prototypes = torch.mean(inputs, dim=self._dim)
        return prototypes

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


class CirclesHyperNetwork(nn.Module):

    def __init__(self, classifier):
        super(CirclesHyperNetwork, self).__init__()
        # H1
        self.hyper_layer1 = nn.Sequential(nn.Linear(2, 64), nn.Tanh())  # 2 x 32
        self.hyper_bias1 = nn.Sequential(nn.Linear(2, 32), nn.Tanh())
        # Layer1
        self.layer1 = classifier[0]  # nn.Linear(2, 32)
        # H2
        self.hyper_layer2 = nn.Sequential(nn.Linear(2, 1024), nn.Tanh())  # 32 x 32
        self.hyper_bias2 = nn.Sequential(nn.Linear(2, 32), nn.Tanh())
        # Layer2
        self.layer2 = classifier[2]  # nn.Linear(32, 32)
        # H3
        self.hyper_layer3 = nn.Sequential(nn.Linear(2, 32), nn.Tanh())  # 32 x 1
        self.hyper_bias3 = nn.Sequential(nn.Linear(2, 1))
        # Layer3
        self.layer3 = classifier[4]  # nn.Linear(32, 1)

        # net activation
        self.activation = nn.ReLU()

        # initialize the values for each network layer
        self.layer1_weights_new = self.layer1.weight.clone().detach()
        self.layer1_bias_new = self.layer1.bias.clone().detach()
        self.layer2_weights_new = self.layer2.weight.clone().detach()
        self.layer2_bias_new = self.layer2.bias.clone().detach()
        self.layer3_weights_new = self.layer3.weight.clone().detach()
        self.layer3_bias_new = self.layer3.bias.clone().detach()

        # initialize synthetic points
        self._generate_synthetic_points()

    def _update_weights(self):
        self.layer1.load_state_dict({'weight': self.layer1_weights_new,
                                     'bias': self.layer1_bias_new})
        self.layer2.load_state_dict({'weight': self.layer2_weights_new,
                                     'bias': self.layer2_bias_new})
        self.layer3.load_state_dict({'weight': self.layer3_weights_new,
                                     'bias': self.layer3_bias_new})

    def _generate_synthetic_points(self, samples=100):
        x_syn = torch.randn((samples, 2), requires_grad=True)
        y_syn = torch.empty((samples, 1)).random_(2)
        self.x_syn = x_syn
        self.y_syn = y_syn

    def weight_regularization_loss(self):
        loss1 = F.mse_loss(self.layer1_weights_new, self.layer1.weight) + \
                F.mse_loss(self.layer1_bias_new, self.layer1.bias)
        loss2 = F.mse_loss(self.layer2_weights_new, self.layer2.weight) + \
                F.mse_loss(self.layer2_bias_new, self.layer2.bias)
        loss3 = F.mse_loss(self.layer3_weights_new, self.layer3.weight) + \
                F.mse_loss(self.layer3_bias_new, self.layer3.bias)
        loss = loss1 + loss2 + loss3
        return loss

    def hyper_forward(self, input: Tensor):
        self.layer1_weights_new = self.hyper_layer1(input).mean(0).view(32, 2)
        self.layer1_bias_new = self.hyper_bias1(input).mean(0).view(32)
        self.layer2_weights_new = self.hyper_layer2(input).mean(0).view(32, 32)
        self.layer2_bias_new = self.hyper_bias2(input).mean(0).view(32)
        self.layer3_weights_new = self.hyper_layer3(input).mean(0).view(1, 32)
        self.layer3_bias_new = self.hyper_bias3(input).mean(0).view(1)

    def forward(self, input):
        layer1_out = F.linear(input, self.layer1_weights_new, self.layer1_bias_new)
        layer1_out = self.activation(layer1_out)

        layer2_out = F.linear(layer1_out, self.layer2_weights_new, self.layer2_bias_new)
        layer2_out = self.activation(layer2_out)

        layer3_out = F.linear(layer2_out, self.layer3_weights_new, self.layer3_bias_new)

        return layer3_out


# class HyperNetwork(nn.Module):
#
#     def __init__(self, classifier):
#         super(HyperNetwork, self).__init__()
#         # H1
#         self.hyper_layer1 = nn.Sequential(nn.Linear(2, 64), nn.Tanh())  # 2 x 32
#         self.hyper_bias1 = nn.Sequential(nn.Linear(2, 32), nn.Tanh())
#         # Layer1
#         self.layer1 = classifier[0]  # nn.Linear(2, 32)
#         # H2
#         self.hyper_layer2 = nn.Sequential(nn.Linear(2, 1024), nn.Tanh())  # 32 x 32
#         self.hyper_bias2 = nn.Sequential(nn.Linear(2, 32), nn.Tanh())
#         # Layer2
#         self.layer2 = classifier[2]  # nn.Linear(32, 32)
#         # H3
#         self.hyper_layer3 = nn.Sequential(nn.Linear(2, 32), nn.Tanh())  # 32 x 1
#         self.hyper_bias3 = nn.Sequential(nn.Linear(2, 1))
#         # Layer3
#         self.layer3 = classifier[4]  # nn.Linear(32, 1)
#
#         # net activation
#         self.activation = nn.ReLU()
#         self.net = classifier
#
#         # Module dictionary
#         self.hypernetwork = nn.ModuleDict(
#             {"hyper_layer1": self.hyper_layer1,
#              "hyper_bias1": self.hyper_bias1,
#              "hyper_layer2": self.hyper_layer2,
#              "hyper_bias2": self.hyper_bias2,
#              "hyper_layer3": self.hyper_layer3,
#              "hyper_bias3": self.hyper_bias3
#              }
#         )
#         # update values for each network layer
#         self.layer1_weights_update = self.layer1.weight.clone().detach()
#         self.layer1_bias_update = self.layer1.bias.clone().detach()
#         self.layer2_weights_update = self.layer2.weight.clone().detach()
#         self.layer2_bias_update = self.layer2.bias.clone().detach()
#         self.layer3_weights_update = self.layer3.weight.clone().detach()
#         self.layer3_bias_update = self.layer3.bias.clone().detach()
#
#     def update_weights(self):
#         self.layer1.load_state_dict({'weight': self.layer1_weights_update,
#                                      'bias': self.layer1_bias_update})
#         self.layer2.load_state_dict({'weight': self.layer2_weights_update,
#                                      'bias': self.layer2_bias_update})
#         self.layer3.load_state_dict({'weight': self.layer3_weights_update,
#                                      'bias': self.layer3_bias_update})
#
#     def forward(self, input, syntetic, lr=1e-3):
#         self.layer1_weights_update = self.layer1.weight + lr * self.hyper_layer1(syntetic).mean(0).view(32, 2)
#         self.layer1_bias_update = self.layer1.bias + lr * self.hyper_bias1(syntetic).mean(0).view(32)
#         self.layer2_weights_update = self.layer2.weight + lr * self.hyper_layer2(syntetic).mean(0).view(32, 32)
#         self.layer2_bias_update = self.layer2.bias + lr * self.hyper_bias2(syntetic).mean(0).view(32)
#         self.layer3_weights_update = self.layer3.weight + lr * self.hyper_layer3(syntetic).mean(0).view(1, 32)
#         self.layer3_bias_update = self.layer3.bias + lr * self.hyper_bias3(syntetic).mean(0).view(1)
#
#         layer1_out = F.linear(input, self.layer1_weights_update, self.layer1_bias_update)
#         layer1_out = self.activation(layer1_out)
#
#         layer2_out = F.linear(layer1_out, self.layer2_weights_update, self.layer2_bias_update)
#         layer2_out = self.activation(layer2_out)
#
#         layer3_out = F.linear(layer2_out, self.layer3_weights_update, self.layer3_bias_update)
#
#         return layer3_out


# class HyperNetwork(nn.Module):
#
#     def __init__(self, classifier):
#         super(HyperNetwork, self).__init__()
#         # H1
#         self.hyper_layer1 = nn.Sequential(nn.Linear(1, 3520), nn.Tanh())  # 55 x 64
#         self.hyper_bias1 = nn.Sequential(nn.Linear(1, 64), nn.Tanh())
#         # Layer1
#         self.layer1 = classifier[0]  # nn.Linear(55, 64)
#         self.batch_norm1 = classifier[2]  # nn.BatchNorm1d(64)
#         # H2
#         self.hyper_layer2 = nn.Sequential(nn.Linear(1, 1024), nn.Tanh())  # 64 x 16
#         self.hyper_bias2 = nn.Sequential(nn.Linear(1, 16), nn.Tanh())
#         # Layer2
#         self.layer2 = classifier[3]  # nn.Linear(64, 16)
#         self.batch_norm2 = classifier[5]  # nn.BatchNorm1d(16)
#         # H3
#         self.hyper_layer3 = nn.Sequential(nn.Linear(1, 16), nn.Tanh())  # 16 x 1
#         self.hyper_bias3 = nn.Sequential(nn.Linear(1, 1))
#         # Layer3
#         self.layer3 = classifier[6]  # nn.Linear(16, 1)
#
#         # net activation
#         self.activation = nn.ReLU()
#         self.net = classifier
#
#         # Module dictionary
#         self.hypernetwork = nn.ModuleDict(
#             {"hyper_layer1": self.hyper_layer1,
#              "hyper_bias1": self.hyper_bias1,
#              "hyper_layer2": self.hyper_layer2,
#              "hyper_bias2": self.hyper_bias2,
#              "hyper_layer3": self.hyper_layer3,
#              "hyper_bias3": self.hyper_bias3
#              }
#         )
#         # update values for each network layer
#         self.layer1_weights_update = self.layer1.weight.clone().detach()
#         self.layer1_bias_update = self.layer1.bias.clone().detach()
#         self.layer2_weights_update = self.layer2.weight.clone().detach()
#         self.layer2_bias_update = self.layer2.bias.clone().detach()
#         self.layer3_weights_update = self.layer3.weight.clone().detach()
#         self.layer3_bias_update = self.layer3.bias.clone().detach()
#
#     def update_weights(self):
#         self.layer1.load_state_dict({'weight': self.layer1_weights_update,
#                                      'bias': self.layer1_bias_update})
#         self.layer2.load_state_dict({'weight': self.layer2_weights_update,
#                                      'bias': self.layer2_bias_update})
#         self.layer3.load_state_dict({'weight': self.layer3_weights_update,
#                                      'bias': self.layer3_bias_update})
#
#     def forward(self, input, label, lr=1e-3):
#         self.layer1_weights_update = self.layer1.weight + lr * self.hyper_layer1(label).view(64, 55)
#         self.layer1_bias_update = self.layer1.bias + lr * self.hyper_bias1(label).view(64)
#         self.layer2_weights_update = self.layer2.weight + lr * self.hyper_layer2(label).view(16, 64)
#         self.layer2_bias_update = self.layer2.bias + lr * self.hyper_bias2(label).view(16)
#         self.layer3_weights_update = self.layer3.weight + lr * self.hyper_layer3(label).view(1, 16)
#         self.layer3_bias_update = self.layer3.bias + lr * self.hyper_bias3(label).view(1)
#
#         layer1_out = F.linear(input, self.layer1_weights_update, self.layer1_bias_update)
#         layer1_out = self.activation(layer1_out)
#         layer1_out = self.batch_norm1(layer1_out)
#
#         layer2_out = F.linear(layer1_out, self.layer2_weights_update, self.layer2_bias_update)
#         layer2_out = self.activation(layer2_out)
#         layer2_out = self.batch_norm2(layer2_out)
#
#         layer3_out = F.linear(layer2_out, self.layer3_weights_update, self.layer3_bias_update)
#
#         return layer3_out


class MetricModel(nn.Module):
    def __init__(self):
        super(MetricModel, self).__init__()
        self.model = nn.Sequential(nn.Linear(512, 40),
                                   nn.Tanh(),
                                   nn.Linear(40, 10), nn.Tanh(),
                                   nn.Linear(10, 2))

    def forward(self, x):
        return self.model(x)


class RealNVP(nn.Module):

    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(),
                                     nn.Linear(256, 2),
                                     nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(),
                                     nn.Linear(256, 2))
        masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
        prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, log_p = self.f(x)
        return self.prior.log_prob(z) + log_p

    def sample(self, batchsize):
        z = self.prior.sample((batchsize, 1))
        log_p = self.prior.log_prob(z)
        x = self.g(z)
        return x
