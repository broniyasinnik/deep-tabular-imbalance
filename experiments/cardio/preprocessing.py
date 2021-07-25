import pandas as pd
import numpy as np
from data_utils import TableConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


data_path = "data/full/cardio_full.csv"
config_path = "data/full/cardio.json"
df = pd.read_csv(data_path,
                 delimiter=';',
                 index_col='id')

df_train, df_test, df_y_train, df_y_test = train_test_split(df, df['cardio'],
                                                            test_size=0.33,
                                                            random_state=42,
                                                            stratify=df['cardio'])

df_train.to_csv('./data/cardio_train.csv', index=False)
df_test.to_csv('./data/cardio_test.csv', index=False)

scaler = MinMaxScaler()
onehot_encoder = OneHotEncoder(drop='if_binary', sparse=False)
config = TableConfig(config_path, label='cardio')
y = df['cardio'].to_numpy()
df.drop(columns='cardio', inplace=True)
X_cont = scaler.fit_transform(df[config.continuous_cols])
X_cat = onehot_encoder.fit_transform(df[config.categorical_cols])
X = np.concatenate([X_cont, X_cat], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

np.savez('data/full/cardio_train.npz', X=X_train, y=y_train)
np.savez('data/full/cardio_test.npz', X=X_test, y=y_test)



