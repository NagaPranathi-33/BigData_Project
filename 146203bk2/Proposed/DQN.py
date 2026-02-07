import numpy as np, random
from numpy import newaxis
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import tensorflow as tf
import math
from Proposed import DRN
from Proposed import RFQN
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
MODEL_NAME = '2x256'
MIN_REWARD = -200
MEMORY_FRACTION = 0.20


def classify(Data, Label, dts, tr, A, Tpr, Tnr):
    o1 = DRN.classify(dts, tr)
    o3 = RFQN.process(Data, Label, o1)

    x_train, x_test, y_train, y_test = train_test_split(o3, Label, train_size=tr, random_state=42)

    class Blob:
        def __init__(self, size):
            self.size = size
            self.x = np.random.randint(0, size)
            self.y = np.random.randint(0, size)

        def __sub__(self, other):
            return (self.x - other.x, self.y - other.y)

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y

        def action(self, choice):
            movements = [(1, 1), (-1, -1), (-1, 1), (1, -1), (1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
            dx, dy = movements[choice]
            self.move(dx, dy)

        def move(self, x=0, y=0):
            self.x = np.clip(self.x + x, 0, self.size - 1)
            self.y = np.clip(self.y + y, 0, self.size - 1)

    class BlobEnv:
        SIZE = 10
        RETURN_IMAGES = True
        MOVE_PENALTY = 1
        ENEMY_PENALTY = 300
        FOOD_REWARD = 25
        OBSERVATION_SPACE_VALUES = np.shape(x_train)
        ACTION_SPACE_SIZE = 9
        PLAYER_N = 1
        FOOD_N = 2
        ENEMY_N = 3
        d = {1: (255, 175, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

        def reset(self):
            self.player = Blob(self.SIZE)
            self.food = Blob(self.SIZE)
            while self.food == self.player:
                self.food = Blob(self.SIZE)
            self.enemy = Blob(self.SIZE)
            while self.enemy == self.player or self.enemy == self.food:
                self.enemy = Blob(self.SIZE)
            self.episode_step = 0
            return np.array(self.get_image() if self.RETURN_IMAGES else (self.player - self.food) + (self.player - self.enemy))

        def step(self, action):
            self.episode_step += 1
            self.player.action(action)
            new_observation = np.array(self.get_image() if self.RETURN_IMAGES else (self.player - self.food) + (self.player - self.enemy))
            if self.player == self.enemy:
                reward = -self.ENEMY_PENALTY
            elif self.player == self.food:
                reward = self.FOOD_REWARD
            else:
                reward = -self.MOVE_PENALTY
            done = reward in [self.FOOD_REWARD, -self.ENEMY_PENALTY] or self.episode_step >= 200
            return new_observation, reward, done

        def get_image(self):
            # Dummy grayscale image for simulation
            return np.zeros((self.SIZE, self.SIZE))

    env = BlobEnv()

    class ModifiedTensorBoard(TensorBoard):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.step = 1
            self.writer = tf.summary.create_file_writer(self.log_dir)

        def set_model(self, model): pass
        def on_epoch_end(self, epoch, logs=None): self.update_stats(**logs)
        def on_batch_end(self, batch, logs=None): pass
        def on_train_end(self, _): pass
        def update_stats(self, **stats): self._write_logs(stats, self.step)

    class DQNAgent:
        def __init__(self, train_data, train_label, test_data, test_label, pred):
            self.model = self.create_model(train_data, train_label, test_data, test_label, pred)

        def create_model(self, train_data, train_label, test_data, test_label, pred):
            train_data, test_data = train_data.astype('float32') / np.max(train_data), test_data.astype('float32') / np.max(test_data)
            train_x = train_data[:, :, newaxis]
            train_y = to_categorical(train_label)
            test_x = test_data[:, :, newaxis]

            model = Sequential([
                Conv1D(32, kernel_size=3, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])),
                Conv1D(64, kernel_size=3, activation='relu'),
                Dropout(0.5),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(100, activation='relu'),
                Dense(50, activation='relu'),
                Dense(train_y.shape[1], activation='softmax')
            ])
            model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
            model.fit(train_x, train_y, epochs=1, batch_size=16, verbose=0)
            preds = model.predict(test_x)
            pred.extend(preds)
            return model

        def get_qs(self, state):
            return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    pred = []
    agent = DQNAgent(np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), pred)

    predict = list(y_train)
    for i in range(len(pred)):
        if i == 0:
            predict.append(np.argmax(pred[i]))
        else:
            diff = np.abs(pred[i] - pred[i - 1])
            predict.append(np.argmax(diff))

    target = np.concatenate((y_train, y_test), axis=0)
    predict = np.array(predict[:len(target)])
    target = np.array(target)

    tp = tn = fp = fn = 0
    for i in range(len(target)):
        actual = target[i]
        predicted = predict[i]
        if actual == predicted:
            tp += 1
        else:
            fp += 1
            fn += 1

    acc = tp / len(target)
    tpr = tp / (tp + fn) if (tp + fn) else 0
    tnr = tn / (tn + fp) if (tn + fp) else 0

    A.append(acc)
    Tpr.append(tpr)
    Tnr.append(tnr)
