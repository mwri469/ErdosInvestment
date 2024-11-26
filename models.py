import numpy as np
import pandas as pd
import pickle as pkl
import torch
from tqdm import tqdm

import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Input

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from lstm_pipe import *
from globals import *
from preprocess import *

def main():
    (X_train, y_train, X_val, y_val, X_oos, y_oos), scaler = preprocess_data()

    # Replace NaN values in X's
    X_train, X_val, X_oos = map(replace_NaNs, (X_train, X_val, X_oos))

    train, val, oos = data_to_tensors(X_train, y_train, X_val, y_val, X_oos, y_oos)

    models = []
    
    # Train multiple models and collect 
    print('\nTraining models in ensemble. . .')
    for i in tqdm(range(NUM_MODELS)):
        model = build_models(X_train)  # Build a new model instance
        trained_model, _ = train_model(train, val, model, i)  # Train the model
        models.append(trained_model)  # Store the trained model

    print('\nEvaluating ensemble predictions. . .')
    y_hat_oos = evaluate_ensemble(models, oos, y_oos)

    mse = mean_squared_error(y_hat_oos, y_oos)
    print(f'MSE : {mse}')

def data_to_tensors(X_train, y_train, X_val, y_val, X_oos, y_oos):
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
    oos_data = tf.data.Dataset.from_tensor_slices((X_oos, y_oos)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
    return train_data, val_data, oos_data

def evaluate_ensemble(models: list, oos_data, y_oos: np.array) -> np.ndarray:

    # Collect predictions from all models
    predictions = []
    
    for model in models:
        # Predict on out-of-sample data
        preds = model.predict(oos_data, verbose=0)
        predictions.append(preds)

    # Calculate ensemble mean prediction
    y_hat_oos = np.mean(predictions, axis=0)  # Average across models
    
    return y_hat_oos

def build_models(X):
    model = Sequential([
        # Input(),
        LSTM(32, input_shape=(PAST, X.shape[2]), return_sequences=True),
        LSTM(16, return_sequences=True),
        LSTM(8),
        Dense(FUTURE)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    return model

def train_model(train, test, model, i, epochs=30):
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
    
    return model, history

if __name__ == '__main__':
    main()
