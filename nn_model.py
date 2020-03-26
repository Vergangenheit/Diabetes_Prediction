import tensorflow
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split


def cls_weights(targets: np.array):
    class_weights = class_weight.compute_class_weight('balanced', np.array([0, 1]), targets)

    return class_weights


def split(X: np.array, y: np.array):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0, test_size=0.2, shuffle=True)

    return X_train, X_valid, y_train, y_valid


def model(X_train: np.array):
    inputs = tensorflow.keras.layers.Input(shape=(X_train.shape[1],))
    dense = tensorflow.keras.layers.Dense(128, activation='relu')(inputs)
    drops = tensorflow.keras.layers.Dropout(0.1)(dense)
    outputs = tensorflow.keras.layers.Dense(1, activation='sigmoid')(drops)
    m = tensorflow.keras.models.Model(inputs=inputs, outputs=outputs)

    return m


def train_nn(model: tensorflow.keras.models.Model, class_weights: np.array, X_train, X_valid, y_train, y_valid):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=2, validation_data=(X_valid, y_valid),
              class_weight=dict(enumerate(class_weights)))
