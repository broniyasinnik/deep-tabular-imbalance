from models.transforms import OneHotTransform


def test_one_hot_transform(adult):
    one_hot = OneHotTransform(adult.categorical_cols, adult.categories_sizes)
    result = one_hot(adult.data[0])
    assert True
