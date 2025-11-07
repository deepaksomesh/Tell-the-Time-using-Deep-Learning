import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     BatchNormalization, Dropout,
                                     Flatten, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initializing the required parameters (Episodes, Mini-batching, Learning rate)

epochs = 100
batch_size = 128
learning_rate = 5e-5

# Loading and processing the data (Converting labels to total minutes, Normalizing pixel values)

print("Loading dataset ...")
images = np.load("images.npy")
labels = np.load("labels.npy")

valid = (
    (labels[:,0] >= 0) & (labels[:,0] < 24) &
    (labels[:,1] >= 0) & (labels[:,1] < 60)
)
images, labels = images[valid], labels[valid]

hours = labels[:,0] % 12
minutes = labels[:,1]
targets = hours + minutes/60.0
targets = targets.astype("float32")

images = images.astype("float32") / 255.0
if images.ndim == 3:
    images = images[..., np.newaxis]

# Train/Validation/Test Split of the data (80/10/10)

X_train, X_temp, y_train, y_temp = train_test_split(
    images, targets, test_size=0.2, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Common Sense Error - Mean circular difference (minimizing around the clock).

def common_sense_error(y_true, y_pred):
    diff = np.abs(y_true - y_pred) % 12.0
    diff = np.minimum(diff, 12.0 - diff)
    return np.mean(diff * 60.0)

# Visualizing the loss MSE and MAE metrics

def plot_training(h):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(h.history["loss"], label="train")
    plt.plot(h.history["val_loss"], label="val")
    plt.title("Loss (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(h.history["mae"], label="train")
    plt.plot(h.history["val_mae"], label="val")
    plt.title("Mean Absolute Error")
    plt.xlabel("Epochs")
    plt.ylabel("MAE (hours)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#A regression CNN architecture with Adam optimizer, MSE loss, and MAE metrics

def build_regression_cnn(input_shape):
    inp = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation="relu", padding="same")(inp)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation="linear", name="time_regression")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model

# Training the model with early stopping and reduce learning rate on plateau
# and Results in MAE and Common Sense Error.

model = build_regression_cnn(X_train.shape[1:])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=2
)

plot_training(history)

preds = model.predict(X_test).reshape(-1)
mae_hours = np.mean(np.abs(preds - y_test))
common_err = common_sense_error(y_test, preds)

print(f"MAE (hours):  {mae_hours:.4f}")
print(f"Common-sense MAE: {common_err:.2f} minutes")
