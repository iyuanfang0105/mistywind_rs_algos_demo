import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder

pd.set_option('display.max_columns', None)
# from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Dropout, SpatialDropout1D, Activation, concatenate
# from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.layers.advanced_activations import ReLU, PReLU, LeakyReLU, ELU
# from tensorflow.keras.layers.normalization import BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import plot_model

class Dataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.COLUMNS = [
            "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
            "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss",
            "hours_per_week", "native_country", "income_bracket"
        ]

        self.DROP_COLUMNS = ["fnlwgt", "education_num"]

        self.CATEGORICAL_COLUMNS = [
            "workclass", "education", "marital_status", "occupation", "relationship",
            "race", "gender", "native_country"
        ]

        self.CONTINUOUS_COLUMNS = [
            "age", "capital_gain", "capital_loss", "hours_per_week"
        ]

        self.LABEL_COLUMN = "label"

    def load_data(self):
        train_data = self.read_data_file(os.path.join(self.data_dir, 'train.d'))
        test_data = self.read_data_file(os.path.join(self.data_dir, 'test.d'))
        print('====>>>> dataset info')
        print('train shape: {}, test shape: {}'.format(train_data.shape, test_data.shape))
        print('data sample:\n {}'.format(train_data.sample(n=1)))
        return train_data, test_data

    def read_data_file(self, file_name):
        data = pd.read_csv(os.path.join(self.data_dir, file_name), names=self.COLUMNS)
        # select used columns
        data = data.drop(self.DROP_COLUMNS, axis=1)
        data = data.dropna(how='any', axis=0)
        data['label'] = data['income_bracket'].apply(lambda x:'>50K' in x).astype(np.uint8)
        data = data.drop(self.LABEL_COLUMN)
        return data

# def preprocessing():
#     train_data = pd.read_csv('../data/wide_and_deep/adult.data', names=COLUMNS)
#     train_data.dropna(how='any', axis=0)
#
#     test_data = pd.read_csv('../data/wide_and_deep/adult.test', skiprows=1, names=COLUMNS)
#     test_data.dropna(how='any', axis=0)
#
#     all_data = pd.concat([train_data, test_data])
#
#     # label
#     all_data[LABEL_COLUMN] = all_data['income_bracket'].apply(lambda x: ">50K" in x).astype(int)
#     all_data.pop('income_bracket')
#
#     y = all_data[LABEL_COLUMN].values
#     all_data.pop(LABEL_COLUMN)
#
#     for c in CATEGORICAL_COLUMNS:
#         le = LabelEncoder()
#         all_data[c] = le.fit_transform(all_data[c])
#     train_size = len(train_data)
#     x_train = all_data.iloc[:train_size]
#     y_train = y[:train_size]
#     x_test = all_data.iloc[train_size:]
#     y_test = y[train_size:]
#     x_train_categ = np.array(x_train[CATEGORICAL_COLUMNS])  # 训练集中的类别数据，转成numpy类型
#     x_test_categ = np.array(x_test[CATEGORICAL_COLUMNS])  # 测试集中的类别类别数据
#     x_train_conti = np.array(x_train[CONTINUOUS_COLUMNS], dtype='float64')  # 训练集中的连续数据
#     x_test_conti = np.array(x_test[CONTINUOUS_COLUMNS], dtype='float64')  # 测试集中的连续数据
#     scaler = StandardScaler()
#     x_train_conti = scaler.fit_transform(x_train_conti)  # 连续训练数据标准化
#     x_test_conti = scaler.transform(x_test_conti)
#     return x_train, y_train, x_test, y_test, x_train_categ, x_test_categ, x_train_conti, x_test_conti, all_data


if __name__ == '__main__':
    dataset = Dataset(data_dir='../data/')
    dataset.load_data()