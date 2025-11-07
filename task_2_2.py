import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     BatchNormalization, Dropout,
                                     Flatten, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import math

# Initializing the required parameters (Episodes, Mini-batching, Learning rate, Random state)

RANDOM_STATE = 42
BATCH_SIZE = 64
INITIAL_LR = 5e-5
EPOCHS = 250

# Loading and processing the data (Converting time to sin/cosine encoding,
# Normalizing pixel values)

print("Loading dataset...")
images = np.load("images.npy")
labels = np.load("labels.npy")

images = images.astype("float32") / 255.0
if images.ndim == 3: images = images[..., np.newaxis]

hours = labels[:, 0] % 12
minutes = labels[:, 1]
angles = 2 * np.pi * (hours + minutes / 60.0) / 12.0  # radians in [0,2π)
y_sin = np.sin(angles)
y_cos = np.cos(angles)
targets = np.stack([y_sin, y_cos], axis=1).astype("float32")

# Train/Validation/Test Split of the data (80/10/10)

X_train, X_temp, y_train, y_temp = train_test_split(
    images, targets, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, shuffle=True
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Computing the circular mean absolute error in minutes

def common_sense_error(y_true, y_pred):
    true_angle = np.arctan2(y_true[:, 0], y_true[:, 1])
    pred_angle = np.arctan2(y_pred[:, 0], y_pred[:, 1])
    diff = np.abs(true_angle - pred_angle)
    diff = np.minimum(diff, 2 * np.pi - diff)
    return np.mean(diff * 12 * 60 / (2 * np.pi))

# Visulazing the loss and the mean absolute error (MAE)

def plot_training(h):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(h.history["loss"], label="train")
    plt.plot(h.history["val_loss"], label="val")
    plt.title("MSE (Loss)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(h.history["mae"], label="train")
    plt.plot(h.history["val_mae"], label="val")
    plt.title("Mean Abs Error")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# CNN Architecture with augmenting (slight rotations, small translations, zoom-in/out)
# images randomly during the training to prevent overfitting.

def build_sincos_cnn(input_shape):
    inp = Input(shape=input_shape)

    # 1. Data Augmentation Layers (Crucial for generalization on visual tasks)
    x = tf.keras.layers.RandomRotation(
        factor=0.08,  # Max ~30 degrees rotation
        fill_mode='constant',
        interpolation='bilinear'
    )(inp)
    x = tf.keras.layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1,
        fill_mode='constant',
        interpolation='bilinear'
    )(x)
    x = tf.keras.layers.RandomZoom(
        height_factor=0.15,
        fill_mode='constant',
        interpolation='bilinear'
    )(x)

    x = Conv2D(32, (5, 5), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)

    out = Dense(2, activation="linear", name="sincos_output")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss="mse",
        metrics=["mae"]
    )
    return model

# Training and Results

model = build_sincos_cnn(X_train.shape[1:])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)

plot_training(history)

preds = model.predict(X_test)
mae_lin = np.mean(np.abs(preds - y_test))
common_err = common_sense_error(y_test, preds)

print("\nEvaluation")
print(f"Linear MAE (sin/cos): {mae_lin:.4f}")
print(f"Common-sense MAE: {common_err:.2f} minutes")

model.save("tell_time_optimized_sincos_task2_final.h5")
