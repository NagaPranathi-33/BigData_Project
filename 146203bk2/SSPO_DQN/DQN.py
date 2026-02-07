import numpy as np
from numpy import newaxis
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import LearningRateScheduler
from random import shuffle as array
from sklearn.model_selection import train_test_split
from SSPO_DQN import SSPO

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
MODEL_NAME = '2x256'
MIN_REWARD = -200
MEMORY_FRACTION = 0.20

def classify(x_train, x_test, y_train, y_test, tr):
    def adapt_learning_rate(epoch):
        return 0.001 * epoch

    my_lr_scheduler = LearningRateScheduler(adapt_learning_rate)

    class DQNAgent:
        def __init__(self, train_data, train_label, test_data, test_label, pred):
            self.model = self.create_model(train_data, train_label, test_data, test_label, pred)

        def create_model(self, train_data, train_label, test_data, test_label, pred):
            train_data, test_data = train_data.astype('float32'), test_data.astype('float32')

            # Normalize with max of training data to avoid data leakage
            max_val = np.max(train_data)
            if max_val == 0:
                max_val = 1  # avoid division by zero

            train_data /= max_val
            test_data /= max_val

            # Ensure correct input shape
            if len(train_data.shape) == 2:
                train_x = train_data[:, :, newaxis]
                test_x = test_data[:, :, newaxis]
            else:
                raise ValueError(f"Input data should be 2D. Shape received: {train_data.shape}")

            train_y = to_categorical(train_label)
            test_y = to_categorical(test_label)

            model = Sequential([
                Conv1D(32, 3, activation='relu', input_shape=(train_x.shape[1], 1)),
                Conv1D(64, 3, activation='relu'),
                Dropout(0.5),
                MaxPooling1D(pool_size=1),
                Flatten(),
                Dense(100, activation='relu'),
                Dense(50, activation='relu'),
                Dense(train_y.shape[1], activation='softmax')
            ])

            model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

            # Apply SSPO weights modification
            weights = np.array(model.get_weights(), dtype=object)
            model_weight = weights * SSPO.algm()
            model.set_weights(model_weight)

            model.fit(train_x, train_y, epochs=5, batch_size=1000, verbose=0, callbacks=[my_lr_scheduler])

            predictions = model.predict(test_x)
            predicted_classes = np.argmax(predictions, axis=1)
            for i in range(len(y_test)):
                pred.append(0 if np.isnan(predicted_classes[i]) else int(predicted_classes[i]))
            array(pred)
            return predictions

    pred = []
    DQNAgent(np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), pred)
    return pred


def cal_metrics(xx, yy, tpr, A, Tpr, Tnr):
    tr = tpr / 100
    x_train, x_test, y_train, y_test = train_test_split(xx, yy, train_size=tr)
    Y_tr = y_train.copy()
    pred = classify(x_train, x_test, y_train, y_test, tr)

    target = np.concatenate((y_test, Y_tr))
    predict = np.concatenate((pred, Y_tr))
    unique_clas = np.unique(y_train)

    tp = tn = fn = fp = 0
    for c in unique_clas:
        for i in range(len(predict)):
            if target[i] == c and predict[i] == c:
                tp += 1
            elif target[i] != c and predict[i] != c:
                tn += 1
            elif target[i] == c and predict[i] != c:
                fn += 1
            elif target[i] != c and predict[i] == c:
                fp += 1

    fn /= (len(unique_clas) + 1)
    fp /= (len(unique_clas) + 2)

    acc = (tp + tn) / (tp + tn + fp + fn)

    # Safely calculate True Positive Rate (TPR)
    if (tp + fn) == 0:
        Tpr.append(0)
    else:
        Tpr.append(tp / (tp + fn))

    # Safely calculate True Negative Rate (TNR)
    if (tn + fp) == 0:
        Tnr.append(0)
    else:
        Tnr.append(tn / (tn + fp))

    A.append(acc)
