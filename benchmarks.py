import sdgym
import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
from data_utils import majority_data_undersample

def f1_vs_majority_ratio(x_train, y_train, x_test, y_test, model, ratios):
    d = defaultdict(list)
    for ratio in ratios:
        values, counts = np.unique(y_train, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_value = values[majority_idx]
        majority_ids, = np.where(y_train == majority_value)
        minority_ids, = np.where(y_train != majority_value)
        num_majority_samples = int(np.floor(majority_ids.size * ratio))
        print(f"The number of undersampled majority:{num_majority_samples}")
        new_majority_ids = np.random.choice(majority_ids, num_majority_samples, replace=False)
        ids = np.concatenate([new_majority_ids, minority_ids])
        np.random.shuffle(ids)
        x = x_train[ids]
        y = y_train[ids]
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            model.fit(x, y)
            pred = model.predict(x_test)
        macro_f1 = f1_score(y_test, pred, average='macro')
        d['ratio'].append(ratio)
        d['f1'].append(macro_f1)
    return pd.DataFrame(d)

def sdgym_binary_classification_benchmark(x_train, y_train, x_test, y_test):
    classifiers = sdgym.evaluate._MODELS['binary_classification']
    performance = []
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        logging.info('Evaluating using binary classifier %s', model_repr)
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            model.fit(x_train, y_train)
            # keep probabilities of positive class
            pred_proba = model.predict_proba(x_test)[:, 1]
            pred = model.predict(x_test)

        precision, recall, thresholds = precision_recall_curve(y_test, pred_proba)
        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average='macro')
        micro_f1 = f1_score(y_test, pred, average='micro')

        performance.append(
            {
                "name": model_repr,
                "precision": precision,
                "recall": recall,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1
            }
        )

    return performance
