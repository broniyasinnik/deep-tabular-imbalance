import torch
import numpy as np
from sklearn.metrics import average_precision_score


def average_precision_metric(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)
    ap = average_precision_score(y_true, y_pred)
    return ap

