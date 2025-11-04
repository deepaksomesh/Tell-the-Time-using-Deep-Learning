import numpy as np, tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     BatchNormalization, Dropout,
                                     Flatten, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

images = np.load("images.npy")
labels = np.load("labels.npy")

valid = (
        (labels[:,0]>=0)&(labels[:,0]<24)&
        (labels[:,1]>=0)&(labels[:,1]<60)
)
images, labels = images[valid], labels[valid]
images = images.astype("float32")/255.0
if images.ndim==3: images = images[...,np.newaxis]

# Hour in range [0,12), minute in [0,60)
hours = (labels[:,0] % 12).astype("float32")
minutes = labels[:,1].astype("float32")

X_train,X_temp,yh_train,yh_temp,ym_train,ym_temp = train_test_split(
    images,hours,minutes,test_size=0.2,random_state=42,shuffle=True)
X_val,X_test,yh_val,yh_test,ym_val,ym_test = train_test_split(
    X_temp,yh_temp,ym_temp,test_size=0.5,random_state=42,shuffle=True)

def circular_mse_hours(y_true, y_pred):
    diff = tf.abs(y_true - y_pred) % 12.0
    diff = tf.minimum(diff, 12.0 - diff)
    return tf.reduce_mean(tf.square(diff))

def plot_training(h):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(h.history["loss"], label="train")
    plt.plot(h.history["val_loss"], label="val")
    plt.title("Loss (MSE)"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(h.history["mae"], label="train")
    plt.plot(h.history["val_mae"], label="val")
    plt.title("MAE (hours)"); plt.legend()
    plt.tight_layout(); plt.show()

def build_multihead(input_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(32,3,activation="relu",padding="same")(inp)
    x = MaxPooling2D(2)(x); x = BatchNormalization()(x)
    x = Conv2D(64,3,activation="relu",padding="same")(x)
    x = MaxPooling2D(2)(x); x = BatchNormalization()(x)
    x = Conv2D(128,3,activation="relu",padding="same")(x)
    x = MaxPooling2D(2)(x); x = BatchNormalization()(x)
    x = Conv2D(256,3,activation="relu",padding="same")(x)
    x = MaxPooling2D(2)(x); x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512,activation="relu")(x)
    x = Dropout(0.4)(x)
    out_h = Dense(1,activation="linear",name="hour_output")(x)
    out_m = Dense(1,activation="linear",name="minute_output")(x)
    model = Model(inp,[out_h,out_m])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss={"hour_output": circular_mse_hours, "minute_output": "mse"},
        loss_weights={"hour_output": 2.0, "minute_output": 1.0},
        metrics={"hour_output": "mae", "minute_output": "mae"})
    return model

model = build_multihead(X_train.shape[1:])
callbacks = [
    EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=6,min_lr=1e-6)
]

history = model.fit(
    X_train, {"hour_output": yh_train, "minute_output": ym_train},
    validation_data=(X_val, {"hour_output": yh_val, "minute_output": ym_val}),
    epochs=150, batch_size=128, callbacks=callbacks, verbose=2)

yh_pred, ym_pred = model.predict(X_test)
yh_pred, ym_pred = yh_pred.flatten(), ym_pred.flatten()

def common_sense_error(yh_t, yh_p, ym_t, ym_p):
    true_total = (yh_t * 60 + ym_t) % 720
    pred_total = (yh_p * 60 + ym_p) % 720
    diff = np.abs(true_total - pred_total)
    diff = np.minimum(diff, 720 - diff)
    return np.mean(diff)

mae_h = np.mean(np.abs(yh_pred - yh_test))
mae_m = np.mean(np.abs(ym_pred - ym_test))
common_err = common_sense_error(yh_test,yh_pred,ym_test,ym_pred)

print("\nEvaluation")
print(f"Hour MAE (hours):   {mae_h:.3f}")
print(f"Minute MAE (minutes): {mae_m:.2f}")
print(f"Common-sense MAE:   {common_err:.2f} minutes")
