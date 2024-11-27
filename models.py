import numpy as np
import pandas as pd
import pickle as pkl
import torch
from tqdm import tqdm
import random

import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, Dropout, BatchNormalization

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from lstm_pipe import *
from globals import *
from preprocess import *
from sequential_models import *

def main():
    (X_train, y_train, X_val, y_val, X_oos, y_oos), scaler = preprocess_data()

    try:
        with open('data/datasets.pickle', 'wb') as f:
            pkl.dump((X_train, y_train, X_val, y_val, X_oos, y_oos), f)
    except:
        pass
    
    # Replace NaN values in X's
    X_train, X_val, X_oos = map(replace_NaNs, (X_train, X_val, X_oos))

    train, val, oos = data_to_tensors(X_train, y_train, X_val, y_val, X_oos, y_oos)

    models = []
    
    # Train multiple models and collect 
    print('\nTraining models in ensemble. . .')
    pipeline = model_pipeline()
    pipeline.train_ensemble(train, val, X_val)
    pipeline.save_models()

    print('\nEvaluating ensemble predictions. . .')
    y_hat_oos = evaluate_ensemble(models, oos)

    mse = mean_squared_error(y_hat_oos, y_oos)
    print(f'\nMSE : {mse}')

class model_pipeline:
    def __init__(self, config=None):
        if config is not None:
            print('\nLoading configuration. . .')
            self.config = config
        else:
            print('\nUsing default configuration. . .')
            self.config = {
                'Num models': NUM_MODELS,
                'Optimizer': keras.optimizers.Adam,
                'Model choices': [build_simple_model, build_medium_model, build_complex_model, light_dropout,
                                  heavy_dropout, complex_bidirectional]
            }
        self.models = []

    def compile_model(self, model, learning_rate):
        """
        Compiles the given model with specified learning rate.
        """
        optimizer = self.config['Optimizer'](learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def train_ensemble(self, train, val, X):
        """
        Trains multiple models with varying hyperparameters and stores them.
        """
        print('\nTraining models in ensemble. . .')
        for i in tqdm(range(self.config['Num models'])):
            learning_rate = random.uniform(0.0001, 0.01)  # Random learning rate
            epochs = random.randint(10, 50)  # Random number of epochs
            model_choice = random.choice(self.config['Model choices'])
            
            model = model_choice(X)  # Randomly select a model architecture
            model = self.compile_model(model, learning_rate)
            trained_model, _ = train_model(train, val, model, i, epochs)
            self.models.append(trained_model)

    def save_models(self, path='model_weights/ensemble_models.pkl'):
        with open(path, 'wb') as f:
            pkl.dump(self.models, f)

    def load_models(self, path='model_weights/ensemble_models.pkl'):
        with open(path, 'rb') as f:
            self.models = pkl.load(f)

def data_to_tensors(X_train: np.ndarray, y_train:np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_oos: np.ndarray, y_oos: np.ndarray) -> tf.data.Dataset:
    """
    Converts pre-processed X->y arrays to tensors for NN training

    Parameters:
    -----------
        X_train, X_val, X_oos: np.array-like
            ... X datasets to train model on. Expected shape is (Num. obs., Num. timesteps, Num. features)
        y_train, y_val, y_oos: np.array-like
            ... y datasets to compare predictions. Expected shape is (Num. obs, 1)

    Returns:
    --------
        train_data, val_data, oos_data: tf.data.Dataset
            ... Built tensors to train in NN's
    """
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
    oos_data = tf.data.Dataset.from_tensor_slices((X_oos, y_oos)).batch(256).prefetch(tf.data.experimental.AUTOTUNE)
    return train_data, val_data, oos_data

def evaluate_ensemble(models: list, oos_data: tf.data.Dataset) -> np.ndarray:
    """
    From an ensemble model, evaluate on some out-of-sample data.

    Parameters:
    -----------
        models: list-like
        ... List of created ensemble models. Pretrained.
        oos_data:
        ... TODO: eval if X_oos would work instead
    """

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
