import numpy as np
from sklearn.metrics import precision_recall_curve


def f1_score(y_pred, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1 = np.nanmax(precision * recall * 2 / (precision + recall))
    return f1

