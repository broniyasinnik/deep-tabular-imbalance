import torch
import torch.nn as nn
import numpy as np
from torch import distributions
from typing import List, Tuple


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_layers: List[int]):
        super(MLP, self).__init__()
        self.layers = []
        hidden_layers.insert(0, in_features)
        hidden_layers.append(out_features)
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if i != len(hidden_layers)-1:
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
