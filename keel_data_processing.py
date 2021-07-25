import glob
import json
import os
from dataclasses import make_dataclass
from pathlib import Path
from collections import defaultdict
import yaml
from imblearn.over_sampling import SMOTE
from category_encoders.target_encoder import TargetEncoder
from ml_collections import ConfigDict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Callable
import re
import numpy as np
import pandas as pd


def get_all_datasets(self):
    train_files = glob.glob(f'{self.root}/**/*tra.dat', recursive=True)
    test_files = glob.glob(f'{self.root}/**/*tst.dat', recursive=True)
    train_test_dic = {}
    f_names = []
    for tra_file in train_files:
        path = Path(tra_file)
        f_name = (path.name.rsplit('tra.dat')[0])
        f_names.append(f_name)
        for tst_file in test_files:
            if tst_file.endswith(f"{f_name}tst.dat"):
                train_test_dic[f_name] = (tra_file, tst_file)
                break
    assert len(train_test_dic.keys()) == len(f_names)
    return train_test_dic


def num_config_lines(data_file):
    with open(data_file) as f:
        config_count = 0
        for line in f:
            if line.startswith('@'):
                config_count += 1
        return config_count


def keel_table_config(dat_file: str) -> ConfigDict:
    relation_pat = re.compile('^@relation (?P<table_name>\S+)$')
    attribute_real = re.compile("^@attribute (?P<attribute>\S+) real \[.*\]$")
    attribute_integer = re.compile("^@attribute (?P<attribute>\S+) integer \[.*\]$")
    attribute_category = re.compile("^@attribute (?P<attribute>\S+) \{.*\}$")
    inputs = re.compile("^@inputs (?P<inputs>(\S+?\, )+\S+)$")
    outputs = re.compile("^@outputs (?P<output>\S+)$")
    config = ConfigDict()
    config.real = []
    config.categorical = []
    config.integer = []
    with open(dat_file) as f:
        for line in f:
            if m := re.match(relation_pat, line):
                config.relation = m.group('table_name')
            elif m := re.match(attribute_real, line):
                config.real.append(m.group('attribute'))
            elif m := re.match(attribute_integer, line):
                config.integer.append(m.group('attribute'))
            elif m := re.match(attribute_category, line):
                config.categorical.append(m.group('attribute'))
            elif m := re.match(inputs, line):
                lst = m.group('inputs')
                config.features = list(map(str.strip, lst.split(',')))
            elif m := re.match(outputs, line):
                output = m.group('output')
                config.output = output
            elif line == '@data':
                break
    skip_lines = num_config_lines(data_file=dat_file)
    config.data = pd.read_csv(dat_file, skiprows=skip_lines, names=config.features+[config.output])
    if config.output in config.categorical:
        config.categorical.remove(config.output)

    return config


def process_keel_dataset(path: str, smote=True):
    dataset_name = os.path.basename(path)
    train_file = glob.glob(f'{path}/*tra.dat')[0]
    test_file = glob.glob(f'{path}/*tst.dat')[0]
    skip_lines_train = num_config_lines(train_file)
    skip_lines_test = num_config_lines(test_file)
    config = keel_table_config(train_file)
    output_col = config.output
    features = config.features
    column_names = features + [output_col]
    train_df = pd.read_csv(train_file, skiprows=skip_lines_train, names=column_names)
    test_df = pd.read_csv(test_file, skiprows=skip_lines_test, names=column_names)
    train_df[output_col] = train_df[output_col].apply(str.strip).map({'positive': 1, 'negative': 0})
    test_df[output_col] = test_df[output_col].apply(str.strip).map({'positive': 1, 'negative': 0})
    X_train, y_train = train_df[features], train_df[output_col]
    X_test, y_test = test_df[features], test_df[output_col]
    if (y_train == 1).sum() < (y_train == 0).sum():
        y_train.map({0: 1, 1: 0})
        y_test.map({0: 1, 1: 0})

    real_transformer = StandardScaler()
    integer_transformer = MinMaxScaler()
    categorical_transformer = TargetEncoder()
    preprocessor = ColumnTransformer(
        transformers=[
            ('real', real_transformer, config.real),
            ('int', integer_transformer, config.integer)
            ('cat', categorical_transformer, config.categorical)])

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_test = preprocessor.transform(X_test)
    if smote:
        smt = SMOTE(random_state=42)
        X_smt, y_smt = smt.fit_resample(X_train, y_train)
        X_smt = X_smt[X_train.shape[0]:]
        y_smt = y_smt[X_train.shape[0]:]

    np.savez(f'{path}/{dataset_name}.tra.npz', X=X_train, y=y_train)
    np.savez(f'{path}/{dataset_name}.tst.npz', X=X_test, y=y_test)
    if smote:
        np.savez(f'{path}/{dataset_name}.smt.npz', X=X_smt, y=y_smt)


# def process_keel_smote_data(path: str):
#     f_name = os.path.basename(path)
#     smote_file = glob.glob(f'{path}/*smt.dat')[0]
#     train_file = glob.glob(f'{path}/*tra.dat')[0]
#
#     num_skip = num_config_lines(smote_file)
#     df_smote = pd.read_csv(smote_file, skiprows=num_skip, header=None)
#     df_train = pd.read_csv(train_file, skiprows=num_skip, header=None)
#
#     df_smote[df_smote.columns[-1]] = df_smote[df_smote.columns[-1]].apply(str.strip)
#     df_smote[df_smote.columns[-1]] = df_smote[df_smote.columns[-1]].map({'positive': 1, 'negative': 0})
#     df_train[df_train.columns[-1]] = df_train[df_train.columns[-1]].apply(str.strip)
#     df_train[df_train.columns[-1]] = df_train[df_train.columns[-1]].map({'positive': 1, 'negative': 0})
#
#     X_train, y_train = df_train[df_train.columns[:-1]], df_train[df_train.columns[-1]]
#     X_smote, y_smote = df_smote[df_smote.columns[:-1]], df_smote[df_smote.columns[-1]]
#     if (y_train == 0).sum() < (y_train == 1).sum():
#         print(f"Dataset {f_name} has more positives than negatives")
#         y_train.map({0: 1, 1: 0})
#         y_smote.map({0: 1, 1: 0})
#
#     X_train = pd.get_dummies(X_train)
#     X_smote = pd.get_dummies(X_smote)
#     X_smote = X_smote[X_train.shape[0]:]
#     y_smote = y_smote[y_train.shape[0]:]
#     assert (y_smote == 1).all(), "All labels in smote should be 1"
#     assert (y_train == 0).sum() == ((y_train == 1).sum() + (y_smote == 1).sum()), "Sizes of train and smote don't match"
#
#     scaler = StandardScaler()
#     scaler.fit(X_train)
#     X_smote = scaler.transform(X_smote)
#     np.savez(f'{path}/{f_name}.smt.npz', X=X_smote, y=y_smote)
#     # np.savez(f'{path}/{f_name}-smt.tst.npz', X=X_test, y=y_test)


def apply_to_all_datasets(root: str, func: Callable):
    for d in os.listdir(root):
        print("Processing ", d)
        func(os.path.join(root, d))


if __name__ == '__main__':
    root = './Keel1/winequality-red-4/winequality-red-4-5-1tra.dat'
    config = keel_table_config(root)
    # config = process_keel_dataset(root)
    # apply_to_all_datasets(root, process_keel_smote_data)
    # df = process_keel_results(root)
    # d = get_ir_in_configs(root)
    # with open('all.json', 'w') as f:
    #     json.dump(all_files, f, indent=4)
    # # process_keel_smote_data(root)
    # for d_name in os.listdir(root):
    #     print(f"Processing {d_name}")
    #     # process_keel_smote_data(os.path.join(root, d_name))
    #     # print(f"Finished processing {d_name}")
