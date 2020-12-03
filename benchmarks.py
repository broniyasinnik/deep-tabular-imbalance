import pickle
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
from sklearn.metrics import average_precision_score


def performance(saved_model: str, x_test, y_test):
    model = pickle.load(open(saved_model, 'rb'))
    pred_proba = model.predict_proba(x_test)[:, 1]
    pred = model.predict(x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, pred_proba)
    acc = accuracy_score(y_test, pred)
    ap = average_precision_score(y_test, pred_proba)
    # f1 = f1_score(y_test, pred)
    f1 = np.max(precision * recall * 2 / (precision + recall))
    performance = {
        "pred": pred,
        "pred_proba": pred_proba,
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
        "AP": ap,
        "f1": f1,
        "thresholds": thresholds
    }
    return performance


def binary_classification_benchmark(x_train, y_train, x_test, y_test, model):
    unique_labels = np.unique(y_train)
    if len(unique_labels) == 1:
        pred = [unique_labels[0]] * len(x_test)
    else:

        # model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
        model.fit(x_train, y_train)
        # keep probabilities of positive class
        pred_proba = model.predict_proba(x_test)[:, 1]
        pred = model.predict(x_test)

    precision, recall, thresholds = precision_recall_curve(y_test, pred_proba)
    acc = accuracy_score(y_test, pred)
    ap = average_precision_score(y_test, pred_proba)
    f1 = f1_score(y_test, pred)

    performance = {
        "pred": pred,
        "pred_proba": pred_proba,
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
        "AP": ap,
        "f1": f1,
        "thresholds": thresholds
    }

    return performance


class ClassificationBenchMark:

    def __init__(self,
                 xgboost=False,
                 logistic_reg=False,
                 mlp=False,
                 xgboost_params=None,
                 logistic_reg_params=None,
                 mlp_params=None):

        if xgboost_params is None:
            xgboost_params = {
                'use_label_encoder': False
            }
        if mlp_params is None:
            mlp_params = {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'batch_size': 100,
                'max_iter': 100,
            }
        if logistic_reg_params is None:
            logistic_reg_params = {}

        self.classifiers = []
        if xgboost:
            self.xgboost = xgb.XGBClassifier(**xgboost_params)
            self.classifiers.append(self.xgboost)
        if logistic_reg:
            self.logistic_reg = LogisticRegression(**logistic_reg_params)
            self.classifiers.append(self.logistic_reg)
        if mlp:
            self.mlp = MLPClassifier(**mlp_params)
            self.classifiers.append(self.mlp)

    def fit(self, train_x, train_y):
        for cls in self.classifiers:
            print(f"Fitting {type(cls).__name__}")
            cls.fit(train_x, train_y)

    def predict(self, test_x):
        pred_proba = {}
        for cls in self.classifiers:
            pred_proba[f'{type(cls).__name__}'] = cls.predict_proba(test_x)
        return pred_proba

    def performance_auc(self, test_x, test_y):
        auc_dict = {}
        for cls in self.classifiers:
            y_hat = cls.predict(test_x)
            score = accuracy_score(test_y, y_hat)
            auc_dict[f'{type(cls).__name__}'] = score
        return auc_dict

    def performance_precision_recall(self, test_x, test_y):
        prec_recall_dict = {}
        for cls in self.classifiers:
            pred_proba = cls.predict_proba(test_x)[:, 1]
            precision, recall, thresholds = precision_recall_curve(test_y, pred_proba)
            prec_recall_dict[f'{type(cls).__name__}'] = {
                "precision": precision,
                "recall": recall
            }
        return prec_recall_dict
