import pytest
from evaluation import binary_classification


def test_binary_classification(adult_train_test):
    x_train, y_train , x_test, y_test = adult_train_test
    performance = binary_classification(x_train, y_train, x_test, y_test)
    print(performance)
