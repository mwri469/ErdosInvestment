import pandas as pd
import torch

import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Input

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import pickle as pkl
from lstm_pipe import *
from globals import *
from preprocess import *

def main():
    (X_train, y_train, X_val, y_val, X_oos, y_oos), scaler = preprocess_data()

    model = build_models(X_train)

    train, val, oos = data_to_tensors(X_train, y_train, X_val, y_val, X_oos, y_oos)

    models = []

    for i in range(NUM_MODELS):
        models.append()

def data_to_tensors(X_train, y_train, X_val, y_val, X_oos, y_oos):
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
    oos_data = tf.data.Dataset.from_tensor_slices((X_oos, y_oos)).batch(256).prefetch(tf.data.bexperimental.AUTOTUNE)
    return train_data, val_data, oos_data

def build_models(X):
    model = Sequential([
        Input((PAST, X.shape[2]))
        LSTM(32),
        LSTM(16)
        LSTM(8),
        Dense(FUTURE)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    return model

def train(train, test, model, i, epochs=30):
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    vbs =0
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
                f'./model_weights/{i}_model_checkpoint.weights.h5',
                monitor='val_loss',
                save_weights_only=True,
                save_best_only=True
            )
    history = model.fit(
                train,
                epochs=epochs,
                validation_data=test,
                callbacks=[es_callback, modelckpt_callback],
                verbose=vbs
            )

if __name__ == '__main__':
    main()
