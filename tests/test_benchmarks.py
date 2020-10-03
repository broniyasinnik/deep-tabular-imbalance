import numpy as np
from benchmarks import f1_vs_majority_ratio
from sklearn.neural_network import MLPClassifier

def test_f1_vs_majority_ratio(adult_train_test):
    x_train, y_train, x_test, y_test = adult_train_test
    model = MLPClassifier(max_iter=2)
    ratios = np.linspace(1, 0.5, 5)
    df = f1_vs_majority_ratio(x_train, y_train, x_test, y_test, model, ratios)
    assert df.shape == (5, 2)

