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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Callable, Tuple
import re
import numpy as np
import pandas as pd


def split_trian_valid_test(dat_file: str):
    assert os.path.exists(dat_file), "The dat file doesn't exists"
    file_name = os.path.basename(dat_file).rpartition('.dat')[0]
    dir_name = os.path.dirname(dat_file)
    data, conf = read_dat_file(dat_file)
    X, y = data.drop(conf.output, axis=1), data[conf.output]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)
    train_data = pd.concat([X_train, y_train], axis=1)
    valid_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv(f'{dir_name}/{file_name}.tra.csv', index=False)
    valid_data.to_csv(f'{dir_name}/{file_name}.val.csv', index=False)
    test_data.to_csv(f'{dir_name}/{file_name}.tst.csv', index=False)
    with open(f'{dir_name}/metadata.yaml', 'w') as file:
        yaml.dump(conf.to_dict(), stream=file)
    return conf


def get_all_data_files(root: str):
    data_files = glob.glob(f'{root}/**/*.dat', recursive=True)
    for dat_file in data_files:
        yield dat_file


def num_config_lines(data_file):
    with open(data_file) as f:
        config_count = 0
        for line in f:
            if line.startswith('@'):
                config_count += 1
        return config_count


def read_dat_file(dat_file: str) -> Tuple[pd.DataFrame, ConfigDict]:
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
    data = pd.read_csv(dat_file, skiprows=skip_lines, names=config.features+[config.output])
    if config.output in config.categorical:
        config.categorical.remove(config.output)

    return data, config


def process_keel_dataset(data_dir: str, smote=True):
    dataset_name = os.path.basename(data_dir)
    train_file = glob.glob(f'{data_dir}/*.tra.csv')[0]
    valid_file = glob.glob(f'{data_dir}/*.val.csv')[0]
    test_file = glob.glob(f'{data_dir}/*.tst.csv')[0]
    with open(f'{data_dir}/metadata.yaml') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
        config = ConfigDict(d)
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    test_df = pd.read_csv(test_file)
    train_df[config.output] = train_df[config.output].apply(str.strip).map({'positive': 1, 'negative': 0})
    valid_df[config.output] = valid_df[config.output].apply(str.strip).map({'positive': 1, 'negative': 0})
    test_df[config.output] = test_df[config.output].apply(str.strip).map({'positive': 1, 'negative': 0})
    X_train, y_train = train_df[config.features], train_df[config.output]
    X_val, y_val = valid_df[config.features], valid_df[config.output]
    X_test, y_test = test_df[config.features], test_df[config.output]
    if (y_train == 1).sum() < (y_train == 0).sum():
        y_train.map({0: 1, 1: 0})
        y_test.map({0: 1, 1: 0})

    real_transformer = StandardScaler()
    integer_transformer = MinMaxScaler()
    categorical_transformer = TargetEncoder()
    preprocessor = ColumnTransformer(
        transformers=[
            ('real', real_transformer, config.real),
            ('int', integer_transformer, config.integer),
            ('cat', categorical_transformer, config.categorical)])

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    if smote:
        smt = SMOTE(random_state=42)
        X_smt, y_smt = smt.fit_resample(X_train, y_train)
        X_smt = X_smt[X_train.shape[0]:]
        y_smt = y_smt[X_train.shape[0]:]
    with open(f'{data_dir}/{dataset_name}.tra.npz', 'wb') as f:
        np.savez(f, X=X_train, y=y_train)
    with open(f'{data_dir}/{dataset_name}.val.npz', 'wb') as f:
        np.savez(f, X=X_val, y=y_val)
    with open(f'{data_dir}/{dataset_name}.tst.npz', 'wb') as f:
        np.savez(f, X=X_test, y=y_test)
    if smote:
        with open(f'{data_dir}/{dataset_name}.smt.npz', 'wb') as f:
            np.savez(f, X=X_smt, y=y_smt)


def apply_to_all_datasets(root: str, func: Callable):
    for d in os.listdir(root):
        func(os.path.join(root, d))



if __name__ == '__main__':
    for dir in os.listdir('./Keel1'):

        print("Processing ", dir, " ...")
        split_trian_valid_test(os.path.join('./Keel1', dir, f"{dir}.dat"))
        process_keel_dataset(os.path.join('./Keel1', dir))
        print("Finished ", dir)

    # conf = split_trian_valid_test('./Keel1/flare-F/flare-F.dat')
    # process_keel_dataset('./Keel1/flare-F')
    # root = "./Keel1/glass-0-1-6_vs_2/glass-0-1-6_vs_2-5-1tra.dat"
    # config = keel_table_config(root)
    # root = './Keel1/'
    # apply_to_all_datasets(root, split_trian_valid_test)
