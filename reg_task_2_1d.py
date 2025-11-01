import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     BatchNormalization, Dropout,
                                     Flatten, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

print("Loading dataset ...")
images = np.load("images.npy")
labels = np.load("labels.npy")     # shape (N,2) → [hour, minute]

valid = (
    (labels[:,0] >= 0) & (labels[:,0] < 24) &
    (labels[:,1] >= 0) & (labels[:,1] < 60)
)
images, labels = images[valid], labels[valid]

hours = labels[:,0] % 12
minutes = labels[:,1]
angles = 2 * np.pi * (hours + minutes/60.0) / 12.0     # radians in [0,2π)
y_sin = np.sin(angles)
y_cos = np.cos(angles)
targets = np.stack([y_sin, y_cos], axis=1).astype("float32")

images = images.astype("float32") / 255.0
if images.ndim == 3: images = images[..., np.newaxis]

X_train, X_temp, y_train, y_temp = train_test_split(
    images, targets, test_size=0.2, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

def radians_from_sincos(y):
    """Convert sin/cos vectors → radians."""
    return np.arctan2(y[:,0], y[:,1])

def common_sense_error(y_true, y_pred):
    """Compute circular mean absolute error in minutes."""
    true_angle = np.arctan2(y_true[:,0], y_true[:,1])
    pred_angle = np.arctan2(y_pred[:,0], y_pred[:,1])
    diff = np.abs(true_angle - pred_angle) % (2*np.pi)
    diff = np.minimum(diff, 2*np.pi - diff)
    return np.mean(diff * 12*60 / (2*np.pi))   # convert radians→minutes

def plot_training(h):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(h.history["loss"], label="train")
    plt.plot(h.history["val_loss"], label="val")
    plt.title("Loss (MSE)"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(h.history["mae"], label="train")
    plt.plot(h.history["val_mae"], label="val")
    plt.title("Mean Abs Error"); plt.legend()
    plt.tight_layout(); plt.show()

def build_sincos_cnn(input_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(inp)
    x = MaxPooling2D((2,2))(x); x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2))(x); x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2))(x); x = BatchNormalization()(x)

    x = Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2))(x); x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(2, activation="linear", name="sincos_output")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="mse",
        metrics=["mae"]
    )
    return model

model = build_sincos_cnn(X_train.shape[1:])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=128,
    callbacks=callbacks,
    verbose=2
)

plot_training(history)

preds = model.predict(X_test)
mae_lin = np.mean(np.abs(preds - y_test))
common_err = common_sense_error(y_test, preds)

print(f"Linear MAE (sin/cos): {mae_lin:.4f}")
print(f"Common-sense MAE: {common_err:.2f} minutes")

model.save("tell_time_regression_sincos.h5")
