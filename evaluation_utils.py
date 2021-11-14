import os
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from ml_collections import ConfigDict
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

from data_utils import load_arrays
from datasets import TableDataset

COLORS: List[str] = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'black', 'orange', 'blue']


def save_pr_curve(labels: np.array, scores: np.array, logdir: str):
    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1.)
    thresholds = np.concatenate([thresholds, [1.]])
    df = pd.DataFrame(data={"precision": precision,
                            "recall": recall,
                            "thresholds": thresholds})
    assert os.path.exists(logdir), f"The directory {logdir} doesn't exist"
    df.to_csv(os.path.join(logdir, "pr.csv"),
              index=False)


def save_pr_figure(results, logdir):
    plt.figure()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    colors = cycle(COLORS)
    for name in results:
        labels, scores = results.get(name)['labels'], results.get(name)['scores']
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        plt.step(recall, precision, color=next(colors), where='post', label=f"PR of {name} (area = {ap:.2f})")
    plt.legend()
    plt.savefig(os.path.join(logdir, "pr.png"))
    plt.close()


def save_roc_figure(results, logdir):
    plt.figure()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC Curve')
    colors = cycle(COLORS)
    for name in results:
        labels, scores = results.get(name)['labels'], results.get(name)['scores']
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.step(fpr, tpr, where='post', color=next(colors), label=f"ROC of {name} (area = {roc_auc:.2f})")
    plt.legend()
    plt.savefig(os.path.join(logdir, "roc.png"))
    plt.close()


def save_metrics_table(results: Dict[str, Any], save_ap: bool, save_auc: bool, p_at: List, logdir: str):
    metrics = defaultdict(list)
    for name in results:
        labels, scores = results.get(name)['labels'], results.get(name)['scores']
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        roc_auc = roc_auc_score(labels, scores)
        metrics["AP"].append(ap)
        metrics["AUC"].append(roc_auc)
        for r in p_at:
            metrics[f'P@{int(r * 100)}%'].append(precision[recall >= r][-1])

    df = pd.DataFrame(data=metrics, index=results.keys())
    df.to_csv(os.path.join(logdir, 'metrics.csv'), index=True)


def aggregate_results(results: Dict[str, Any], metrics: ConfigDict, logdir: str):
    if 'pr' in metrics:
        save_pr_figure(results, logdir)
    if 'roc' in metrics:
        save_roc_figure(results, logdir)
    save_ap = 'ap' in metrics
    save_auc = 'auc' in metrics
    p = next(filter(lambda x: x.startswith('p@'), metrics), [])
    if p:
        p = list(map(lambda x: float(x.strip()), p.partition('@')[2].split(',')))

    save_metrics_table(results, save_ap=save_ap, save_auc=save_auc, p_at=p, logdir=logdir)


def save_predictions(labels: np.array, scores: np.array, save_to: str):
    df = pd.DataFrame(data={"labels": labels,
                            "scores": scores}, columns=['labels', 'scores'])
    df.to_csv(save_to, index=False, header=True)


@torch.no_grad()
def evaluate_model_predictions(model, data: TableDataset, save_dir: Optional[str] = None):
    labels = []
    scores = []
    loader = DataLoader(data, shuffle=False)
    for batch in loader:
        x, y = batch['features'], batch['targets']
        y_hat = model(x)
        labels.append(y.numpy())
        scores.append(torch.sigmoid(y_hat).numpy())
    labels = np.concatenate(labels).squeeze()
    scores = np.concatenate(scores).squeeze()
    if save_dir:
        save_to = Path(save_dir)/f"predictions_{data.name}.csv"
        save_predictions(labels, scores, save_to)
    return labels, scores


def get_low_confidence_predictions(data: str, predictions: str, label: float = 1.0, ratio=0.5, save: Optional[str] = None):
    assert os.path.exists(predictions), "The prediction file doesn't exist"
    features, targets = load_arrays(data)
    df_pred = pd.read_csv(predictions)
    df_pred_lbl = df_pred[df_pred["labels"] == label]
    num_to_take = int(df_pred_lbl.shape[0] * ratio)
    index = df_pred_lbl.sort_values('scores').head(num_to_take).index
    result = dict(features=features[index],
                  targets=targets[index],
                  scores=df_pred.loc[index]['scores'])
    if save:
        np.savez(save, X=features, y=targets)
    return result
