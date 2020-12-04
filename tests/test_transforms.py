from models.transforms import OneHotTransform


def test_one_hot_transform(adult):
    one_hot = OneHotTransform(adult.data, cat_features=adult.categorical_cols)
    print(one_hot(adult.data[0]))
    assert True
