from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Softmax
from tensorflow.keras.utils import to_categorical


class LogisticRegression(object):
    def __init__(self, input_dim, class_num):
        self.model = None
        self.input_dim = input_dim
        self.class_num = class_num

    def creat_model(self):
        self.model = Sequential()
        self.model.add(Dense(self.class_num, activation='softmax', input_dim=self.input_dim))

    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
        y_train = to_categorical(y_train, self.class_num)
        y_test = to_categorical(y_test, self.class_num)
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
