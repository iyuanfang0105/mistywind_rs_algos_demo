import pandas as pd
from models.data_preprocessing import Dataset
from models.feats_eng import FeatEng
from models.deep_and_wide import DeepWide
from models.logistic_regression import LogisticRegression


class Demo(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dataset = Dataset(data_dir=self.data_dir)

    def deep_and_wide_demo(self, model_type='deep'):
        train, test = self.dataset.load_data()

        feat_eng = FeatEng()
        all_data = pd.concat([train, test])
        all_data = feat_eng.label_encoding(all_data, categ_names=self.dataset.CATEGORICAL_COLUMNS)
        train_1 = all_data.iloc[:train.shape[0]]
        test_1 = all_data.iloc[train.shape[0]:]

        deep_and_wide_model = DeepWide(model_type=model_type, train_df=train_1, test_df=test_1,
                                   categ_names=self.dataset.CATEGORICAL_COLUMNS,
                                   conti_names=self.dataset.CONTINUOUS_COLUMNS)

        deep_and_wide_model.create_model()
        print(deep_and_wide_model.model.summary())
        deep_and_wide_model.train_model(epochs=10, optimizer='adam', batch_size=64)

    def lr_demo(self):
        train, test = self.dataset.load_data()
        feat_eng = FeatEng()
        all_data = pd.concat([train, test])
        all_data = feat_eng.one_hot_feature(all_data, categ_names=self.dataset.CATEGORICAL_COLUMNS)
        train_1 = all_data.iloc[:train.shape[0]]
        test_1 = all_data.iloc[train.shape[0]:]

        y_train = train_1['label']
        X_train = train_1.drop(['label'], axis=1)

        y_test = test_1['label']
        X_test = test_1.drop(['label'], axis=1)

        lr = LogisticRegression(input_dim=X_train.shape[1], class_num=2)
        lr.creat_model()
        lr.train_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=64)

        return 0


if __name__ == '__main__':
    demo = Demo(data_dir='../data/')
    # demo.deep_and_wide_demo(model_type='deep')
    # demo.deep_and_wide_demo(model_type='wide_and_deep')
    demo.lr_demo()

