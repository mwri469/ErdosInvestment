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

def build_simple_model(X):
    model = Sequential([
        LSTM(16, return_sequences=True, input_shape=(PAST, X.shape[2])),
        LSTM(8),
        Dense(FUTURE)
    ])
    return model

def build_medium_model(X):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(PAST, X.shape[2])),
        LSTM(16, return_sequences=True),
        LSTM(8),
        Dense(FUTURE)
    ])
    return model

def build_complex_model(X):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(PAST, X.shape[2])),
        LSTM(32, return_sequences=True),
        LSTM(16),
        Dense(FUTURE)
    ])
    return model

def light_dropout(X):
    dropout = 0.1
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(PAST, X.shape[2])),
        Dropout(dropout),
        LSTM(32, return_sequences=True),
        Dropout(dropout),
        LSTM(16),
        Dropout(dropout),
        Dense(FUTURE)
    ])
    return model

def heavy_dropout(X):
    dropout = 0.3
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(PAST, X.shape[2])),
        Dropout(dropout),
        LSTM(32, return_sequences=True),
        Dropout(dropout),
        LSTM(16),
        Dropout(dropout-0.2),
        Dense(FUTURE)
    ])
    return model

def complex_bidirectional(X):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(PAST, X.shape[2]))),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32, return_sequences=True)),
        Bidirectional(LSTM(16)),
        Dense(FUTURE)
    ])
    return model