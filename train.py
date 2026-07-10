"""Train the CNN+LSTM phishing URL detection model.

Architecture (unchanged from the original thesis):
    Conv1D(64, kernel=2, relu) -> MaxPool1D(2) -> LSTM(100)
    -> Flatten -> Dense(64) -> Dense(512) -> Dense(64) -> Dense(2)

Fixes applied relative to the original notebook:
  * input_shape corrected to (55, 1) to match the actual feature count
  * Final activation softmax instead of sigmoid so the two outputs sum
    to one and form a proper categorical distribution
  * StandardScaler fit on the training split only, persisted alongside
    the model so inference reproduces the same preprocessing
  * validation_split + EarlyStopping + ReduceLROnPlateau so training
    no longer runs blind for 100 epochs
  * Stratified train/test split to preserve the 50/50 class balance

Run:
    python train.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

SEED = 42
HERE = Path(__file__).resolve().parent
DATA_DEFAULT = HERE / "dataset_phishing.csv"
FEATURE_COLUMNS_DEFAULT = HERE / "feature_columns.txt"


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_feature_columns(path: Path) -> list[str]:
    """Load the canonical 55-feature column order.

    The original dataset has 87 columns: url + 85 feature columns + status.
    Of those 85, the committed .npy files only carry 55 (the rest were
    dropped during dataset preparation). We use exactly those 55 columns,
    in the same order as the saved arrays, so the trained model is
    compatible with the existing .npy artifacts.
    """
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    # Fall back to inferring from X_train.npy
    x = np.load(HERE / "X_train.npy")
    df = pd.read_csv(DATA_DEFAULT)
    feature_cols = [c for c in df.columns if c not in ("url", "status")]
    if len(feature_cols) != x.shape[1]:
        raise RuntimeError(
            f"Dataset has {len(feature_cols)} feature columns but X_train.npy "
            f"has {x.shape[1]} columns. Cannot infer feature order."
        )
    path.write_text("\n".join(feature_cols) + "\n")
    return feature_cols


def load_dataset(path: Path, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    y = df["status"].map({"legitimate": 0, "phishing": 1}).astype(np.int32).to_numpy()
    X = df[feature_cols].to_numpy(dtype=np.float64)
    return X, y


def build_model(n_features: int) -> keras.Model:
    """Original CNN+LSTM architecture, with corrected input shape and head."""
    model = keras.Sequential([
        keras.layers.Conv1D(
            filters=64,
            input_shape=(n_features, 1),
            kernel_size=2,
            activation="relu",
            kernel_initializer=keras.initializers.GlorotUniform(SEED),
        ),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.LSTM(100),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Youden's J on the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def evaluate(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> dict:
    probs = model.predict(X_test, verbose=0)
    p_phish = probs[:, 1]
    preds = (p_phish >= threshold).astype(np.int32)
    fpr, tpr, _ = roc_curve(y_test, p_phish)
    precision, recall, _ = precision_recall_curve(y_test, p_phish)
    report = classification_report(
        y_test, preds,
        target_names=["legitimate", "phishing"],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, preds).tolist()
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc_roc": float(roc_auc_score(y_test, p_phish)),
        "average_precision": float(average_precision_score(y_test, p_phish)),
        "brier_score": float(brier_score_loss(y_test, p_phish)),
        "f1": float(f1_score(y_test, preds)),
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
    }


def write_sha256sums(out_dir: Path, names: list[str]) -> None:
    lines = []
    for name in names:
        p = out_dir / name
        if not p.exists():
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def write_training_metadata(
    out_dir: Path,
    model: keras.Model,
    history: keras.callbacks.History,
    metrics: dict,
    n_features: int,
    args: argparse.Namespace,
    n_rows: int,
) -> None:
    meta = {
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "seed": SEED,
        "model_architecture": {
            "type": "CNN+LSTM hybrid",
            "total_trainable_params": int(model.count_params()),
            "layers": [
                {"type": "Conv1D", "filters": 64, "kernel_size": 2, "activation": "relu"},
                {"type": "MaxPooling1D", "pool_size": 2},
                {"type": "LSTM", "units": 100},
                {"type": "Flatten"},
                {"type": "Dense", "units": 64, "activation": "relu"},
                {"type": "Dense", "units": 512, "activation": "relu"},
                {"type": "Dense", "units": 64, "activation": "relu"},
                {"type": "Dense", "units": 2, "activation": "softmax"},
            ],
            "loss": "sparse_categorical_crossentropy",
            "optimizer": "Adam",
            "initial_learning_rate": 1e-3,
        },
        "feature_count": int(n_features),
        "dataset_rows": int(n_rows),
        "split": {"test_size": 0.2, "stratify": True, "random_state": SEED},
        "training": {
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "validation_split": args.val_split,
            "early_stopping_patience": args.patience,
            "reduce_lr_on_plateau": True,
            "epochs_run": len(history.history["loss"]),
            "feature_scaling": "StandardScaler fit on X_train only",
        },
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "auc_roc": metrics["auc_roc"],
            "average_precision": metrics["average_precision"],
            "brier_score": metrics["brier_score"],
            "f1": metrics["f1"],
            "threshold": metrics["threshold"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    set_seed(SEED)
    feature_cols = load_feature_columns(FEATURE_COLUMNS_DEFAULT)
    print(f"[train] features: {len(feature_cols)}")

    print(f"[train] loading dataset from {args.data}")
    X, y = load_dataset(args.data, feature_cols)
    print(f"[train] X shape={X.shape}, y shape={y.shape}, "
          f"positive_rate={y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float64)
    X_test_s = scaler.transform(X_test).astype(np.float64)

    # CNN+LSTM expects a channel dimension: (N, features, 1)
    X_train_3d = X_train_s[..., np.newaxis]
    X_test_3d = X_test_s[..., np.newaxis]

    model = build_model(X_train_s.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1,
        ),
    ]

    history = model.fit(
        X_train_3d, y_train,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Threshold tuning on a held-out tail of the training set
    # (the validation_split portion that the model did not fit on).
    n = len(X_train_3d)
    val_slice = slice(max(0, n - 1000), n)
    val_probs = model.predict(X_train_3d[val_slice], verbose=0)[:, 1]
    threshold = find_optimal_threshold(y_train[val_slice], val_probs)
    print(f"[train] chosen threshold = {threshold:.4f}")

    metrics = evaluate(model, X_test_3d, y_test, threshold)
    print("[train] test metrics:")
    print(json.dumps(
        {k: v for k, v in metrics.items()
         if k not in ("roc_curve", "pr_curve", "classification_report")},
        indent=2,
    ))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "my_model.keras"
    scaler_path = out_dir / "scaler.pkl"
    metrics_path = out_dir / "metrics.json"
    history_path = out_dir / "history.json"
    feature_cols_path = out_dir / "feature_columns.txt"

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(history_path, "w") as f:
        json.dump(
            {k: [float(x) for x in v] for k, v in history.history.items()},
            f, indent=2,
        )
    feature_cols_path.write_text("\n".join(feature_cols) + "\n")

    write_sha256sums(
        out_dir,
        ["my_model.keras", "scaler.pkl", "metrics.json", "history.json",
         "feature_columns.txt", "dataset_phishing.csv"],
    )
    write_training_metadata(
        out_dir, model, history, metrics, len(feature_cols), args, X.shape[0],
    )

    print(f"[train] saved artifacts to {out_dir}:")
    for p in (model_path, scaler_path, metrics_path, history_path,
              feature_cols_path, out_dir / "SHA256SUMS",
              out_dir / "training_metadata.json"):
        print(f"  - {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
