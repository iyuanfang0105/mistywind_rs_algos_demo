import os
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)


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
        data = data.drop('income_bracket', axis=1)
        return data


if __name__ == '__main__':
    dataset = Dataset(data_dir='../data/')
    dataset.load_data()