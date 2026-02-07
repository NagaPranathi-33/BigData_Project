import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def process(Data, Label, o1):
    # ---------------------- Check Input Dimensions ------------------------
    if Data.ndim != 2:
        raise ValueError(f"Data should be 2D, got shape: {Data.shape}")
    if Label.ndim == 1:
        Label = Label.reshape(-1, 1)
    elif Label.ndim != 2 or Label.shape[1] != 1:
        raise ValueError(f"Label should be a column vector, got shape: {Label.shape}")

    # ---------------------- Weight Calculation using RMSprop ------------------------
    opt = tf.keras.optimizers.RMSprop()  # ✅ Fixed legacy optimizer
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(Data.shape[1],)),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer=opt, loss='mse')
    model.fit(Data, Label, epochs=5, verbose=0)
    weights = model.get_weights()[0]  # Shape: (features, 2)

    # ---------------------- Linear Combination using Weights ------------------------
    # Fix: Use matrix multiplication properly (sum of products)
    y1 = np.sum(Data @ weights[:, 0])

    # ---------------------- Fusion with DRN Output (o1) -----------------------------
    alpha = 2
    o2 = alpha * y1 + 0.5 * alpha * np.sum(o1)

    # ---------------------- Feature Augmentation ------------------------------------
    o2_expanded = np.full((len(Data), 1), o2)
    fused_data = np.concatenate((Data, o2_expanded), axis=1)

    # ---------------------- Check and Impute NaN/Infinite Values --------------------
    if np.any(np.isnan(fused_data)) or np.any(np.isinf(fused_data)):
        print("Warning: NaN or infinite values detected in fused_data. Imputing with mean values...")
        imputer = SimpleImputer(strategy='mean')
        fused_data = imputer.fit_transform(fused_data)

    if np.any(np.isnan(Label)) or np.any(np.isinf(Label)):
        print("Warning: NaN or infinite values detected in Label. Imputing with most frequent values...")
        imputer_label = SimpleImputer(strategy='most_frequent')
        Label = imputer_label.fit_transform(Label)

    # ---------------------- Linear Regression Prediction ---------------------------
    reg = LinearRegression().fit(fused_data, Label)
    prediction = reg.predict(fused_data).astype(int).reshape(-1, 1)

    # ---------------------- Final Feature Set --------------------------------------
    final_output = np.concatenate((fused_data, prediction), axis=1)

    return final_output
