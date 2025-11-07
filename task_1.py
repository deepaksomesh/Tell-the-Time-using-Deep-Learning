import os, json, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers, initializers, optimizers
from tensorflow.keras.layers import (Input, Dense, Flatten, Dropout,
                                     Conv2D, MaxPooling2D, BatchNormalization)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        CSVLogger, ReduceLROnPlateau)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
set_seed(42)

for p in ["results", "results/models", "results/logs", "results/summaries"]:
    os.makedirs(p, exist_ok=True)

def get_regularizer(cfg):
    if cfg.get("regularizer") == "l1":
        return regularizers.l1(cfg.get("reg_value", 1e-4))
    elif cfg.get("regularizer") == "l2":
        return regularizers.l2(cfg.get("reg_value", 1e-4))
    return None

def get_initializer(cfg):
    name = cfg.get("initializer")
    return initializers.get(name) if name else None

def get_optimizer(cfg):
    opt = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("learning_rate", None)
    if opt == "adam":
        return optimizers.Adam(learning_rate=lr or 1e-3)
    if opt == "sgd":
        return optimizers.SGD(learning_rate=lr or 1e-2, momentum=0.9)
    if opt == "rmsprop":
        return optimizers.RMSprop(learning_rate=lr or 1e-3)
    return optimizers.Adam()

def callbacks_for(run_id):
    return [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ModelCheckpoint(f"results/models/{run_id}.h5",
                        monitor="val_accuracy", save_best_only=True),
        CSVLogger(f"results/logs/{run_id}.csv"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6)
    ]

def summarize_run(run_id, dataset, cfg, hist, test_acc):
    best_val_acc = max(hist.history.get("val_accuracy", [0.0]))
    summary = {
        "run_id": run_id,
        "dataset": dataset,
        "config": cfg,
        "best_val_accuracy": float(best_val_acc),
        "final_test_accuracy": float(test_acc)
    }
    with open(f"results/summaries/{run_id}.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary

def make_run_id(model_type, cfg):
    """Generate descriptive run_id from config"""
    act = cfg.get("activation", "relu")
    opt = cfg.get("optimizer", "adam")
    init = cfg.get("initializer", "glorot_uniform")
    if model_type == "mlp":
        arch = "x".join(map(str, cfg.get("layers", [])))
    else:
        arch = "x".join(map(str, cfg.get("filters", [])))
    reg = cfg.get("regularizer", "none")
    return f"{model_type}_{act}_{opt}_{init}_{reg}_{arch}"

# Fashion MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0

x_train_flat = x_train.reshape((x_train.shape[0], -1))
x_test_flat = x_test.reshape((x_test.shape[0], -1))

x_train_flat, x_val_flat, y_train_flat, y_val_flat = train_test_split(
    x_train_flat, y_train, test_size=0.1, random_state=42, stratify=y_train)
x_train_cnn, x_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

x_train_cnn = x_train_cnn[..., np.newaxis]
x_val_cnn = x_val_cnn[..., np.newaxis]
x_test_cnn = x_test[..., np.newaxis]


def build_mlp(cfg):
    model = Sequential()
    init = get_initializer(cfg)
    model.add(Input(shape=(784,)))
    for units in cfg["layers"]:
        model.add(Dense(units,
                        activation=cfg.get("activation", "relu"),
                        kernel_regularizer=get_regularizer(cfg),
                        kernel_initializer=init))
        if cfg.get("dropout"):
            model.add(Dropout(cfg["dropout"]))
    model.add(Dense(10, activation="softmax", kernel_initializer=init))
    model.compile(optimizer=get_optimizer(cfg),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def build_cnn(cfg, input_shape=(28,28,1)):
    init = get_initializer(cfg)
    model = Sequential()
    model.add(Input(shape=input_shape))
    for f in cfg["filters"]:
        model.add(Conv2D(f, (cfg["kernel"], cfg["kernel"]),
                         activation=cfg.get("activation", "relu"),
                         padding="same",
                         kernel_regularizer=get_regularizer(cfg),
                         kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2))
    if cfg.get("dropout"):
        model.add(Dropout(cfg["dropout"]))
    model.add(Flatten())
    model.add(Dense(128, activation=cfg.get("activation", "relu"),
                    kernel_initializer=init))
    model.add(Dense(10, activation="softmax", kernel_initializer=init))
    model.compile(optimizer=get_optimizer(cfg),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

base_mlp_configs = [
    {"layers":[256,128], "activation":"relu","regularizer":None,"dropout":0.2},
    {"layers":[512,256,128], "activation":"relu","regularizer":"l2","reg_value":1e-4,"dropout":0.3},
    {"layers":[512,256], "activation":"elu","regularizer":"l1","reg_value":1e-5,"dropout":0.3},
    {"layers":[512,256,128], "activation":"relu","regularizer":None,"dropout":0.4}
]

base_cnn_configs = [
    {"filters":[32,64], "kernel":3, "activation":"relu","regularizer":None,"dropout":0.3},
    {"filters":[32,64,128], "kernel":3, "activation":"relu","regularizer":"l2","reg_value":1e-4,"dropout":0.4},
    {"filters":[32,64,128], "kernel":5, "activation":"relu","regularizer":None,"dropout":0.4},
    {"filters":[64,128], "kernel":3, "activation":"elu","regularizer":None,"dropout":0.3}
]

optimizers_to_try = ["adam", "sgd", "rmsprop"]
initializers_to_try = ["glorot_uniform", "he_normal"]

mlp_configs, cnn_configs = [], []
for base in base_mlp_configs:
    for opt in optimizers_to_try:
        for init in initializers_to_try:
            cfg = dict(base)
            cfg.update({"optimizer": opt, "initializer": init})
            mlp_configs.append(cfg)
for base in base_cnn_configs:
    for opt in optimizers_to_try:
        for init in initializers_to_try:
            cfg = dict(base)
            cfg.update({"optimizer": opt, "initializer": init})
            cnn_configs.append(cfg)

summaries = []

for cfg in mlp_configs:
    run_id = make_run_id("mlp", cfg)
    print(f"\nTraining {run_id} ...")
    model = build_mlp(cfg)
    hist = model.fit(x_train_flat, y_train_flat,
                     validation_data=(x_val_flat, y_val_flat),
                     epochs=30, batch_size=128,
                     callbacks=callbacks_for(run_id), verbose=2)
    _, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
    summaries.append(summarize_run(run_id, "fmnist", cfg, hist, test_acc))

for cfg in cnn_configs:
    run_id = make_run_id("cnn", cfg)
    print(f"\nTraining {run_id} ...")
    model = build_cnn(cfg, input_shape=(28,28,1))
    hist = model.fit(x_train_cnn, y_train_cnn,
                     validation_data=(x_val_cnn, y_val_cnn),
                     epochs=30, batch_size=128,
                     callbacks=callbacks_for(run_id), verbose=2)
    _, test_acc = model.evaluate(x_test_cnn, y_test, verbose=0)
    summaries.append(summarize_run(run_id, "fmnist", cfg, hist, test_acc))

pd.DataFrame(summaries).to_csv("results/fmnist_summary.csv", index=False)


df = pd.DataFrame(summaries)
df_sorted = df.sort_values("best_val_accuracy", ascending=False).reset_index(drop=True)


plt.figure(figsize=(12, 5))
plt.plot(range(len(df_sorted)), df_sorted["best_val_accuracy"], marker="o", linestyle="-")
plt.xticks(range(len(df_sorted)), df_sorted["run_id"], rotation=90, fontsize=7)
plt.ylabel("Validation Accuracy")
plt.xlabel("Model Configurations (shortened)")
plt.title("Fashion-MNIST: Validation Accuracy Across All Configurations")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/fmnist_all_configs_lineplot.png", dpi=150)
plt.close()

# Save top-10 JSON summary (still useful)
top10 = df_sorted.head(10)
top3 = top10.head(3).to_dict(orient="records")
with open("results/top3_fmnist.json", "w") as f:
    json.dump(top3, f, indent=2)

# transfer learning on CIFAR10 dataset
(x_train_c, y_train_c), (x_test_c, y_test_c) = tf.keras.datasets.cifar10.load_data()
y_train_c, y_test_c = y_train_c.flatten(), y_test_c.flatten()
x_train_c, x_test_c = x_train_c.astype("float32") / 255.0, x_test_c.astype("float32") / 255.0

x_train_c, x_val_c, y_train_c, y_val_c = train_test_split(
    x_train_c, y_train_c, test_size=0.1, random_state=42, stratify=y_train_c)

retrain_results = []

print("\n=== Retraining top-3 Fashion-MNIST models from scratch on CIFAR-10 ===")

for info in top3:
    run_id, cfg = info["run_id"], info["config"]
    print(f"\nTraining {run_id} on CIFAR-10 from scratch...")

    if run_id.startswith("mlp"):
        # Flatten CIFAR-10 for MLP input
        x_train_c_flat = x_train_c.reshape((x_train_c.shape[0], -1))
        x_val_c_flat   = x_val_c.reshape((x_val_c.shape[0], -1))
        x_test_c_flat  = x_test_c.reshape((x_test_c.shape[0], -1))
        model = build_mlp(cfg)
        hist = model.fit(x_train_c_flat, y_train_c,
                         validation_data=(x_val_c_flat, y_val_c),
                         epochs=40, batch_size=128, verbose=2,
                         callbacks=callbacks_for(run_id + "_cifar"))
        _, test_acc = model.evaluate(x_test_c_flat, y_test_c, verbose=0)
        retrain_results.append({
            "run_id": run_id,
            "dataset": "cifar10",
            "config": cfg,
            "test_accuracy": float(test_acc)
        })

    else:
        model = build_cnn(cfg, input_shape=(32, 32, 3))
        hist = model.fit(x_train_c, y_train_c,
                         validation_data=(x_val_c, y_val_c),
                         epochs=40, batch_size=128, verbose=2,
                         callbacks=callbacks_for(run_id + "_cifar"))
        _, test_acc = model.evaluate(x_test_c, y_test_c, verbose=0)
        retrain_results.append({
            "run_id": run_id,
            "dataset": "cifar10",
            "config": cfg,
            "test_accuracy": float(test_acc)
        })

# Save CIFAR-10 results
pd.DataFrame(retrain_results).to_csv("results/cifar10_summary.csv", index=False)

# Comparison plot (FMNIST vs CIFAR-10)
merged = pd.merge(df, pd.DataFrame(retrain_results), on="run_id", how="inner")
if not merged.empty:
    plt.figure(figsize=(10, 5))
    plt.bar(merged["run_id"], merged["final_test_accuracy"], label="FMNIST")
    plt.bar(merged["run_id"], merged["test_accuracy"], alpha=0.7, label="CIFAR-10")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.title("Top-3 Model Performance: Fashion-MNIST vs CIFAR-10 (No Transfer Learning)")
    plt.tight_layout()
    plt.savefig("results/fmnist_cifar10_comparison.png", dpi=150)
    plt.close()

print("\nDone retraining top-3 models on CIFAR-10!")
print("\n Done mate!!!!!!")
