import torch
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class ToTensor(object):

    def __call__(self, arr: np.array, dtype=torch.float32):
        if arr.size == 1:
            arr = np.array(arr)
        return torch.from_numpy(arr).type(dtype)


class OneHotTransform:
    def __init__(self, categorical_columns: List[int], categories_sizes: List[int] = None):
        self.categorical_columns = categorical_columns
        self.categories_sizes = categories_sizes
        self.encoder = None
        # categories = [np.arange(cat_size) for cat_size in categories_sizes]
        # self.encoder = OneHotEncoder(categories=categories)

    @classmethod
    def from_data(cls, data: np.array, categorical_columns: List[int]):
        self = cls(categorical_columns)
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.encoder.fit(data[:, categorical_columns])

    def __call__(self, data):
        _data = data.copy()
        cat_feat = _data[self.categorical_columns].astype(np.int)
        one_hot_features = [np.squeeze(np.eye(size)[cat]) for cat, size in zip(cat_feat, self.categories_sizes)]
        if self.encoder:
            cat_feat = self.encoder.transform(cat_feat.reshape(1, -1)).toarray()
        else:
            cat_feat = np.concatenate(one_hot_features)
        _data = np.delete(_data, self.categorical_columns)
        _data = np.concatenate((_data, cat_feat.astype(np.float32).squeeze()))
        return _data


class ScalerTransform:
    SCALERS = {
        'standard': StandardScaler,
        'min_max': MinMaxScaler
    }

    def __init__(self, data: np.array, features: List = None, type: str = 'min_max'):
        self.scaler = self.SCALERS[type]()
        self.features = list(range(data.shape[1]))
        if features is not None:
            self.features = features
        data = data[:, self.features].copy()
        self.scaler.fit(data)

    def __call__(self, data):
        _data = data.copy()
        feat_to_scale = _data[self.features].reshape(1, -1)
        feat_to_scale = self.scaler.transform(feat_to_scale)
        _data[self.features] = feat_to_scale
        return _data


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        """
        Args:
            transforms: list of transforms to compose.
        Example:
            >>> Compose([ToTensor(), ScalerTransform()])
        """
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
