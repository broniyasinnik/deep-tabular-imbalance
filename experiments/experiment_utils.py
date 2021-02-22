from pathlib import Path
import torch.nn as nn
import typing as tp
from models.networks import EmbeddingLayer, Generator, MLP
from datasets import AdultDataSet, DatasetImbalanced
from torch.utils.data import DataLoader
from catalyst.dl import ConfigExperiment
from models.transforms import ToTensor, ScalerTransform, Compose


def get_loaders(stage: str = None):
    root = Path.cwd().parent / 'BenchmarkData' / 'adult'
    # load adult dataset
    adult_train = AdultDataSet(root, train=True,
                               target_transform=ToTensor())
    scaler_transform = ScalerTransform(adult_train.data, features=adult_train.continuous_cols)
    to_tensor = ToTensor()
    feat_transform = Compose([scaler_transform, to_tensor])
    adult_train.transform = feat_transform
    adult_test = AdultDataSet(root, train=False,
                              transform=feat_transform,
                              target_transform=ToTensor())

    # Creating imbalanced adult dataset
    adult_train = DatasetImbalanced(imbalance_ratio=1)(adult_train)
    adult_train, discarded_train = DatasetImbalanced(imbalance_ratio=0.1)(adult_train, return_the_complement=True)
    adult_test = DatasetImbalanced(imbalance_ratio=1)(adult_test)
    adult_test, discarded_test = DatasetImbalanced(imbalance_ratio=0.1)(adult_test, return_the_complement=True)
    loaders = {
        "train": DataLoader(adult_train, batch_size=32, shuffle=True),
        "discarded_train": DataLoader(discarded_train, batch_size=32, shuffle=False),
        "test": DataLoader(adult_test, batch_size=32, shuffle=False),
        "discarded_test": DataLoader(discarded_test, batch_size=32, shuffle=False)
    }

    # create config
    config = {"categorical_cols": adult_train.categorical_cols,
              "continuous_cols": adult_train.continuous_cols,
              "embedding_dims": adult_train.embeds}

    return loaders, config


def get_encoder(config, checkpoint=None):
    encoder = nn.Sequential(EmbeddingLayer(config["categorical_cols"],
                                           config["continuous_cols"],
                                           config["embedding_dims"]), nn.BatchNorm1d(55))
    if checkpoint:
        encoder.load_state_dict(checkpoint)

    return encoder


def get_classifier(checkpoint=None):
    classifier = nn.Sequential(nn.Linear(55, 64), nn.ReLU(), nn.BatchNorm1d(64),
                               nn.Linear(64, 16), nn.ReLU(), nn.BatchNorm1d(16),
                               nn.Linear(16, 1))
    if checkpoint:
        classifier.load_state_dict(checkpoint)

    return classifier


def get_generator(noise_dim=10, hidden_dim=32, out_dim=55, checkpoint=None):
    generator = Generator(noise_dim, hidden_dim, out_dim)

    if checkpoint:
        generator.load_state_dict(checkpoint)

    return generator


def get_discriminator(checkpoint=None):
    hidden_dim = 100
    layers = [
        nn.Linear(55 + 1, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, 1)]

    discriminator = nn.Sequential(*layers)

    if checkpoint:
        discriminator.load_state_dict(checkpoint)

    return discriminator
