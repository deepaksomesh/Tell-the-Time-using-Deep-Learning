import numpy as np, tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     BatchNormalization, Dropout,
                                     Flatten, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initializing the required parameters (Episodes, Mini-batching, Learning rate)

epochs = 150
batch_size = 128
learning_rate = 2e-5

# Loading and processing the data (Converting labels to total minutes, Normalizing pixel values)

print("Loading dataset ...")

images = np.load("images.npy")
labels = np.load("labels.npy")

images = images.astype("float32")/255.0
if images.ndim==3:
    images = images[...,np.newaxis]

# for Hours in range [0,12), minutes in [0,60)
hours = (labels[:,0] % 12).astype("float32")
minutes = labels[:,1].astype("float32")

# Train/Validation/Test Split of the data (80/10/10)

X_train,X_temp,yh_train,yh_temp,ym_train,ym_temp = train_test_split(
    images,hours,minutes,test_size=0.2,random_state=42,shuffle=True)
X_val,X_test,yh_val,yh_test,ym_val,ym_test = train_test_split(
    X_temp,yh_temp,ym_temp,test_size=0.5,random_state=42,shuffle=True)

# A Custom Loss Function for circular hours,
# it handles circular regression, returns mean squared difference

def circular_mse_hours(y_true, y_pred):
    diff = tf.abs(y_true - y_pred) % 12.0
    diff = tf.minimum(diff, 12.0 - diff)
    return tf.reduce_mean(tf.square(diff))

# Common Sense Error - Mean circular difference (minimizing around the clock).

def common_sense_error(yh_t, yh_p, ym_t, ym_p):
    true_total = (yh_t * 60 + ym_t) % 720
    pred_total = (yh_p * 60 + ym_p) % 720
    diff = np.abs(true_total - pred_total)
    diff = np.minimum(diff, 720 - diff)
    return np.mean(diff)

# Visualizing the Total loss and Hour Loss (Circular MSE)

def plot_training_a(h):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(h.history["loss"], label="Train Total Loss (Weighted)")
    plt.plot(h.history["val_loss"], label="Val Total Loss (Weighted)")
    plt.title("Total Weighted Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(h.history["hour_output_loss"], label="Train Hour Loss (Circular MSE)")
    plt.plot(h.history["val_hour_output_loss"], label="Val Hour Loss (Circular MSE)")
    plt.title("Hour Loss (Circular MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Visualizing the Hour MAE and Minute MAE (Mean Absolute Error)

def plot_training_b(h):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(h.history["hour_output_mae"], label="Train Hour MAE (hours)")
    plt.plot(h.history["val_hour_output_mae"], label="Val Hour MAE (hours)")
    plt.title("Hour MAE (Mean Absolute Error)")
    plt.xlabel("Epochs")
    plt.ylabel("MAE (hours)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(h.history["minute_output_mae"], label="Train Minute MAE (minutes)")
    plt.plot(h.history["val_minute_output_mae"], label="Val Minute MAE (minutes)")
    plt.title("Minute MAE (Mean Absolute Error)")
    plt.xlabel("Epochs")
    plt.ylabel("MAE (minutes)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# A multi-head regression CNN with hour and minute outputs
# with Adam optimizer and weighted multi-output losses

def build_multihead(input_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(32,3,activation="relu",padding="same")(inp)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(64,3,activation="relu",padding="same")(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128,3,activation="relu",padding="same")(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(256,3,activation="relu",padding="same")(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(512,activation="relu")(x)
    x = Dropout(0.4)(x)
    out_h = Dense(1,activation="linear",name="hour_output")(x)
    out_m = Dense(1,activation="linear",name="minute_output")(x)

    model = Model(inp,[out_h,out_m])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"hour_output": circular_mse_hours, "minute_output": "mse"},
        loss_weights={"hour_output": 2.0, "minute_output": 1.0},
        metrics={"hour_output": "mae", "minute_output": "mae"})
    return model

# Training the model with early stopping and reduce learning rate on plateau
# and Results in MAE and Common Sense Error.

model = build_multihead(X_train.shape[1:])
callbacks = [
    EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=6,min_lr=1e-6)
]

history = model.fit(
    X_train, {"hour_output": yh_train, "minute_output": ym_train},
    validation_data=(X_val, {"hour_output": yh_val, "minute_output": ym_val}),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=2
)


plot_training_a(history)
plot_training_b(history)

yh_pred, ym_pred = model.predict(X_test)
yh_pred, ym_pred = yh_pred.flatten(), ym_pred.flatten()

mae_h = np.mean(np.abs(yh_pred - yh_test))
mae_m = np.mean(np.abs(ym_pred - ym_test))
common_err = common_sense_error(yh_test,yh_pred,ym_test,ym_pred)

print("\nEvaluation")
print(f"Hour MAE (hours):   {mae_h:.3f}")
print(f"Minute MAE (minutes): {mae_m:.2f}")
print(f"Common-sense MAE:   {common_err:.2f} minutes")
