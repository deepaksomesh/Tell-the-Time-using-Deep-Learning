import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                     Dropout, Flatten, Dense, Input)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

n_classes_list = [12, 24, 720]   # 12 (1-hr bins), 24 (30-min bins), 720 (1-min bins)
epochs = 60
batch_size = 128
learning_rate = 1e-4

print("Loading data...")
images = np.load("images.npy")
labels = np.load("labels.npy")   # shape (N, 2): [hour, minute]

# Convert labels to total minutes [0,720)
hours = labels[:, 0] % 12
minutes = labels[:, 1]
total_minutes = (hours * 60 + minutes).astype(int)

# Normalize pixel values
images = images.astype("float32") / 255.0

# Add channel dimension if missing
if len(images.shape) == 3:
    images = images[..., np.newaxis]  # (N, H, W, 1)

# Train/val/test split (80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(
    images, total_minutes, test_size=0.2, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

def minutes_to_class(minutes, n_classes):
    """
    Convert total minutes to class index given number of classes.
    Bins are equal intervals around the clock (cyclic).
    """
    bin_size = 720 / n_classes
    classes = np.floor(minutes / bin_size).astype(int)
    classes = np.clip(classes, 0, n_classes - 1)
    return classes

def class_to_center_minute(class_idx, n_classes):
    """Map class index back to central minute of its interval."""
    bin_size = 720 / n_classes
    return ((class_idx + 0.5) * bin_size) % 720

def common_sense_error(y_true_minutes, y_pred_minutes):
    """Mean circular difference (minimizing around the clock)."""
    diff = np.abs(y_true_minutes - y_pred_minutes) % 720
    diff = np.minimum(diff, 720 - diff)
    return np.mean(diff)

def plot_training(h, n_classes):
    """Visualize loss and accuracy."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(h.history["loss"], label="train")
    plt.plot(h.history["val_loss"], label="val")
    plt.title(f"Loss ({n_classes} classes)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(h.history["accuracy"], label="train")
    plt.plot(h.history["val_accuracy"], label="val")
    plt.title(f"Accuracy ({n_classes} classes)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def build_base_cnn(input_shape, n_classes):
    inp = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation="softmax", name="classification")(x)

    model = Model(inp, out)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model

results = {}

for n_classes in n_classes_list:
    print(f"\nTraining for {n_classes} classes\n")

    # Convert to discrete class indices
    y_train_cls = minutes_to_class(y_train, n_classes)
    y_val_cls = minutes_to_class(y_val, n_classes)
    y_test_cls = minutes_to_class(y_test, n_classes)

    # Build and train CNN
    model = build_base_cnn(X_train.shape[1:], n_classes)
    model.summary()

    history = model.fit(
        X_train, y_train_cls,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_cls),
        verbose=2
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test_cls, verbose=0)
    preds = np.argmax(model.predict(X_test), axis=1)
    pred_minutes = class_to_center_minute(preds, n_classes)
    mean_err = common_sense_error(y_test, pred_minutes)

    print(f"\nAccuracy ({n_classes} classes): {test_acc:.4f}")
    print(f"Mean common-sense error: {mean_err:.2f} minutes")

    results[n_classes] = {
        "accuracy": float(test_acc),
        "common_sense_error_min": float(mean_err)
    }

    plot_training(history, n_classes)

for n, res in results.items():
    print(f"{n:>4} classes → acc: {res['accuracy']:.3f}, "
          f"common-sense err: {res['common_sense_error_min']:.2f} min")
