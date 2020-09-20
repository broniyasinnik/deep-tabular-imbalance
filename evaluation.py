import sdgym
import numpy as np
import logging
from sklearn.metrics import precision_recall_curve


def binary_classification(x_train, y_train, x_test, y_test):
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
            pred = model.predict_proba(x_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, pred)

        performance.append(
            {
                "name": model_repr,
                "precision": precision,
                "recall": recall
            }
        )

    return performance
