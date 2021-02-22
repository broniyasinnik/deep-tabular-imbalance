import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
with open('./BenchmarkData/credit/credit.json') as f:
    config = json.load(f)

names = [col.get('name') for col in config]
df = pd.read_csv('BenchmarkData/raw/creditcard_kaggle.csv', index_col=0, header=None, names=names)
labels = pd.read_csv('BenchmarkData/raw/creditcard_kaggle_labels.csv', index_col=0, header=None, names=['labels'])
normal_ids = np.where(labels == 0)[0]
fraud_ids = np.where(labels == 1)[0]
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.25, random_state=42, stratify=True)
