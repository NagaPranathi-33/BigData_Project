import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

def classify(dts, tr):
    #base_path = "/content/drive/MyDrive/Bha Praject-2(jaanu)/Bha Project/146203bk2/Main/Processed"
    base_path = "/content/drive/MyDrive/Bha Praject-2(jaanu)/Bha Project/146203bk2/Main/Processed"
    if dts == 'Adult':
        data_file = 'Adult_trans.csv'
        label_file = 'Adult_label.csv'
    else:
        data_file = 'Credit_Approval_trans.csv'
        label_file = 'Credit_Approval_label.csv'
    
    data_path = os.path.join(base_path, data_file)
    label_path = os.path.join(base_path, label_file)
    
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing file(s): {data_path if not os.path.exists(data_path) else ''} {label_path if not os.path.exists(label_path) else ''}")
    
    data = pd.read_csv(data_path, header=None).values
    label = pd.read_csv(label_path, header=None).values
    
    x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=tr)
    
    # ---------- Handle non-image data (tabular with fewer features) ----------
    input_dim = x_train.shape[1]
    
    # If data has 3072 features, reshape into image and use ResNet (not in this case)
    if input_dim == 3072:
        print("[DRN] Image data detected. Using ResNet.")
        x_train = np.reshape(x_train, (len(x_train), 32, 32, 3)).astype('float32') / 255
        x_test = np.reshape(x_test, (len(x_test), 32, 32, 3)).astype('float32') / 255

        from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        def lr_schedule(epoch):
            lr = 0.1
            if epoch > 180:
                lr *= 0.5e-3
            elif epoch > 160:
                lr *= 1e-3
            elif epoch > 120:
                lr *= 1e-2
            elif epoch > 80:
                lr *= 1e-1
            return lr

        def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
            x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation:
                x = Activation(activation)(x)
            return x

        def resnet_v1(input_shape, depth, num_classes):
            num_filters = 16
            num_res_blocks = (depth - 2) // 6
            inputs = Input(shape=input_shape)
            x = resnet_layer(inputs)
            for stack in range(3):
                for res_block in range(num_res_blocks):
                    strides = 1 if not (stack > 0 and res_block == 0) else 2
                    y = resnet_layer(x, num_filters, strides=strides)
                    y = resnet_layer(y, num_filters, activation=None)
                    if stack > 0 and res_block == 0:
                        x = resnet_layer(x, num_filters, kernel_size=1, strides=strides,
                                         activation=None, batch_normalization=False)
                    x = tf.keras.layers.add([x, y])
                    x = Activation('relu')(x)
                num_filters *= 2
            x = AveragePooling2D(pool_size=8)(x)
            y = Flatten()(x)
            outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
            return Model(inputs, outputs)

        num_classes = len(np.unique(y_train))
        y_train = tf.keras.utils.to_categorical(y_train.astype('int'), num_classes)
        y_test = tf.keras.utils.to_categorical(y_test.astype('int'), num_classes)

        model = resnet_v1(x_train.shape[1:], depth=20, num_classes=num_classes)
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule(0)),
                      metrics=['accuracy'])

        datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
        datagen.fit(x_train)

        model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=1, verbose=0)
        pred = model.predict(x_test)
        return pred[:, 0]

    else:
        print(f"[DRN] Tabular data with {input_dim} features detected. Using dense model.")
        # Tabular input model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
        pred = model.predict(x_test)
        return pred.reshape(-1, 1)
