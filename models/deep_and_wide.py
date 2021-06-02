import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Activation, concatenate, BatchNormalization
from tensorflow.keras.layers import ReLU, PReLU, LeakyReLU, ELU
from sklearn.preprocessing import PolynomialFeatures


class DeepWide(object):
    def __init__(self, model_type, train_df, test_df, categ_names=[], conti_names=[]):
        self.model_type = model_type
        self.model = None
        self.train_df = train_df
        self.test_df = test_df
        self.all_data = pd.concat([train_df, test_df])
        self.categ_names = categ_names
        self.conti_names = conti_names

        # deep component
        self.categ_inputs = None
        self.conti_input = None
        self.deep_component_outlayer = None

        # wide component
        self.train_categ_poly2 = None
        self.test_categ_poly2 = None
        self.wide_component_outlayer = None

    def deep_component(self):
        # embedding for CATEGORICAL features
        categ_inputs = []
        categ_embeds = []
        for i in range(len(self.categ_names)):
            input_i = Input(shape=(1,), dtype='int32')
            dim = len(np.unique(self.all_data[self.categ_names[i]]))
            embed_dim = int(np.ceil(dim ** 0.25))  # decomposing 4 times
            embed_i = Embedding(dim, embed_dim, input_length=1)(input_i)
            flatten_i = Flatten()(embed_i)
            categ_inputs.append(input_i)
            categ_embeds.append(flatten_i)

        # embedding for CONTINUOUS features
        conti_input = Input(shape=(len(self.conti_names),))
        conti_dense = Dense(256, use_bias=False)(conti_input)

        # concate embedding of CATEGORICAL and CONTINUOUS
        concat_embeds = concatenate([conti_dense] + categ_embeds)
        concat_embeds = Activation('relu')(concat_embeds)
        bn_concat = BatchNormalization()(concat_embeds)
        # full connect
        fc1 = Dense(512, use_bias=False)(bn_concat)
        ac1 = ReLU()(fc1)
        bn1 = BatchNormalization()(ac1)
        # fc2 = Dense(256, use_bias=False)(bn1)
        # ac2 = ReLU()(fc2)
        # bn2 = BatchNormalization()(ac2)
        fc3 = Dense(128)(bn1)
        ac3 = ReLU()(fc3)

        self.categ_inputs = categ_inputs
        self.conti_input = conti_input
        self.deep_component_outlayer = ac3

    def poly2(self):
        poly2 = PolynomialFeatures(degree=2, interaction_only=True)
        self.train_categ_poly2 = poly2.fit_transform(self.train_df[self.conti_names])
        self.test_categ_poly2 = poly2.fit_transform(self.test_df[self.conti_names])

    def wide_component(self):
        self.poly2()
        dim = self.train_categ_poly2.shape[1]
        self.wide_component_outlayer = Input(shape=(dim,))

    def create_model(self):
        self.deep_component()
        self.wide_component()
        if self.model_type == 'wide_and_deep':
            out_layer = concatenate([self.deep_component_outlayer, self.wide_component_outlayer])
            inputs = [self.conti_input] + self.categ_inputs + [self.wide_component_outlayer]
            print(inputs)
        elif self.model_type == 'deep':
            out_layer = self.deep_component_outlayer
            inputs = [self.conti_input] + self.categ_inputs
        else:
            print('wrong mode_type')
            return

        output = Dense(1, activation='sigmoid')(out_layer)
        self.model = Model(inputs=inputs, outputs=output)

    def train_model(self, epochs=15, optimizer='adam', batch_size=128):
        if not self.model:
            print('You have to create model first')
            return

        if self.model_type == 'wide_and_deep':
            X_train = [self.train_df[self.conti_names]] +\
                [self.train_df[c] for c in self.categ_names] +\
                [self.train_categ_poly2]
            y_train = self.train_df['label']

            X_test = [self.test_df[self.conti_names]] +\
                [self.test_df[c] for c in self.categ_names] +\
                [self.test_categ_poly2]
            y_test = self.test_df['label']

        elif self.model_type == 'deep':
            X_train = [self.train_df[self.conti_names]] +\
                [self.train_df[c] for c in self.categ_names]
            y_train = self.train_df['label']

            X_test = [self.test_df[self.conti_names]] +\
                [self.test_df[c] for c in self.categ_names]
            y_test = self.test_df['label']
        else:
            print('wrong mode')
            return

        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    from models.data_preprocessing import Dataset
    from models.feats_eng import FeatEng
    dataset = Dataset(data_dir='../data/')
    train, test = dataset.load_data()

    feat_eng = FeatEng()
    train_1 = feat_eng.label_encoding(train, categ_names=dataset.CATEGORICAL_COLUMNS)
    test_1 = feat_eng.label_encoding(test, categ_names=dataset.CATEGORICAL_COLUMNS)

    deep_and_wide_model = DeepWide(model_type='deep', train_df=train_1, test_df=test_1,
                                   categ_names=dataset.CATEGORICAL_COLUMNS,
                                   conti_names=dataset.CONTINUOUS_COLUMNS)

    deep_and_wide_model.create_model()
    print(deep_and_wide_model.model.summary())
    deep_and_wide_model.train_model(epochs=10, optimizer='adam', batch_size=64)