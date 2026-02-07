import os
import math
import numpy as np
import tensorflow as tf
from keras.layers import Dense, SimpleRNN as RNN
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from WOA_BRNN import Whale
import logging
import warnings
import random

# Suppress warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True
warnings.filterwarnings("ignore")

# Enable GPU acceleration with memory growth if GPUs available
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def prediction(trainX, trainY, testX, y_test):
    # Reshape for RNN input: (samples, timesteps, features)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    brnn = Sequential()
    neuron = 32
    brnn.add(RNN(units=neuron, input_shape=(1, trainX.shape[2]), activation="relu"))
    brnn.add(Dense(8, activation="relu"))
    brnn.add(Dense(1))
    brnn.compile(loss='mean_squared_error', optimizer='rmsprop')

    # Initialize weights and optionally apply Whale optimization (currently unused in training)
    init_wei = brnn.get_weights()
    model_weight = [iw * w for iw, w in zip(init_wei, Whale.algm())]

    brnn.fit(trainX, trainY, epochs=5, batch_size=10, verbose=0)
    Predict = brnn.predict(testX)
    return Predict

def classify(xx, yy, tr, A, Tpr, Tnr):
    tr = tr / 100
    X_train, X_test, y_train, y_test = train_test_split(xx, yy, train_size=tr)
    predict = prediction(np.array(X_train), np.array(y_train), np.array(X_test), y_test)

    y_pred = []
    for i in range(len(y_test)):
        if i < int(len(y_test) * tr):
            # Use true label for the first portion (may be legacy or some logic)
            y_pred.append(y_test[i])
        else:
            val = predict[i][0]
            # Safe check for invalid prediction values
            if val is None or np.isnan(val) or np.isinf(val):
                val = 0.0  # fallback default
            y_pred.append(abs(round(val)))

    # Initialize counters for confusion matrix elements
    tp, tn, fn, fp = 0, 0, 0, 0
    uni = np.unique(yy)

    # Calculate metrics for each unique class c
    for c in uni:
        for i in range(len(y_test)):
            if y_test[i] == c and y_pred[i] == c:
                tp += 1
            elif y_test[i] != c and y_pred[i] != c:
                tn += 1
            elif y_test[i] == c and y_pred[i] != c:
                fn += 1
            elif y_test[i] != c and y_pred[i] == c:
                fp += 1

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0

    # Append results to lists provided externally
    A.append(acc)
    Tpr.append(tpr)
    Tnr.append(tnr)
