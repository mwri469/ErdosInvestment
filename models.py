import pandas as pd
import torch

import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import pickle as pkl
from lstm_pipe import *
from globals import *
from preprocess import *

def main():
    (train_data, val_data, oos_data), scaler = preprocess_data()

def build_models():
    pass

if __name__ == '__main__':
    main()
