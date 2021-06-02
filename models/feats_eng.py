import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder


class FeatEng(object):
    def __init__(self):
        self.label_encoders = {}
        print()

    def label_encoding(self, dataset, categ_names=[]):
        for c in categ_names:
            le = LabelEncoder()
            dataset[c] = le.fit_transform(dataset[c])
            self.label_encoders[c] = le
        return dataset

    def one_hot_feature(self, dataset, categ_names=[]):
        for col in categ_names:
            dataset = pd.concat([dataset.drop([col], axis=1), pd.get_dummies(
                dataset[col], prefix=col)], axis=1)
        return dataset


if __name__ == '__main__':
    from models.data_preprocessing import Dataset
    dataset = Dataset(data_dir='../data/')
    train, test = dataset.load_data()

    feat_eng = FeatEng()
    train_1 = feat_eng.label_encoding(train, categ_names=dataset.CATEGORICAL_COLUMNS)
    print()
