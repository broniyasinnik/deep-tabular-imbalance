import pytest
from benchmarks import ClassificationBenchMark
from benchmarks import binary_classification_benchmark
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from datasets import DatasetSSL


@pytest.mark.table("census")
def test_binary_classification_benchmark(data_train_test):
    train_x, train_y, test_x, test_y = data_train_test
    model = LinearSVC(random_state=17)
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    performance = binary_classification_benchmark(train_x, train_y, test_x, test_y, calibrated)
    proba = performance['pred_proba']
    assert True


def test_benchmark(adult):
    dataset = DatasetSSL(adult)
    dataset.split_to_labeled_unlabeled(num_labeled=300, sample_classes='balanced')
    benchmark = ClassificationBenchMark(xgboost=True,
                                        # logistic_reg=True,
                                        mlp=True)
    benchmark.fit(dataset.train_x_labeled, dataset.train_y_labeled)
    auc = benchmark.perfomance_auc(dataset.test_x, dataset.test_y)
    print(auc)
    assert True