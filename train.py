"""Train the phishing URL detection model.

Usage:
    python train.py [--data dataset_phishing.csv] [--out my_model.keras]

Produces:
    my_model.keras        - trained Keras model
    scaler.pkl            - fitted StandardScaler
    feature_names.json    - canonical feature order
    metrics.json          - evaluation metrics on held-out test set
    history.json          - training history (JSON-serializable)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
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

# Make local package importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phishing_detector.features import FEATURE_NAMES, extract_features  # noqa: E402

SEED = 42
DATA_DEFAULT = Path(__file__).resolve().parent / "dataset_phishing.csv"
OUT_DIR = Path(__file__).resolve().parent


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) where X is built by running extract_features() on each URL.

    We deliberately recompute features from the URL string rather than
    trusting the CSV's pre-computed columns, so that training and inference
    use the exact same feature pipeline. This eliminates train/serve skew.
    """
    df = pd.read_csv(path)
    if "status" not in df.columns or "url" not in df.columns:
        raise ValueError("Dataset must contain 'url' and 'status' columns")
    y = df["status"].map({"legitimate": 0, "phishing": 1}).astype(np.int32).to_numpy()
    urls = df["url"].astype(str).tolist()
    X = np.vstack([extract_features(u) for u in urls])
    return X, y


def build_model(n_features: int) -> keras.Model:
    inp = keras.Input(shape=(n_features,), name="features")
    x = keras.layers.Dense(
        64, activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        kernel_initializer=keras.initializers.GlorotUniform(seed=SEED),
    )(inp)
    x = keras.layers.Dropout(0.3, seed=SEED)(x)
    x = keras.layers.Dense(
        32, activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        kernel_initializer=keras.initializers.GlorotUniform(seed=SEED + 1),
    )(x)
    x = keras.layers.Dropout(0.3, seed=SEED + 1)(x)
    out = keras.layers.Dense(
        1, activation="sigmoid",
        kernel_initializer=keras.initializers.GlorotUniform(seed=SEED + 2),
        name="phishing_prob",
    )(x)
    model = keras.Model(inp, out, name="phishing_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Youden's J statistic on the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def evaluate(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> dict:
    probs = model.predict(X_test, verbose=0).ravel()
    preds = (probs >= threshold).astype(np.int32)
    fpr, tpr, roc_thr = roc_curve(y_test, probs)
    precision, recall, pr_thr = precision_recall_curve(y_test, probs)
    report = classification_report(
        y_test, preds, target_names=["legitimate", "phishing"], output_dict=True,
    )
    cm = confusion_matrix(y_test, preds).tolist()
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auc_roc": float(roc_auc_score(y_test, probs)),
        "average_precision": float(average_precision_score(y_test, probs)),
        "brier_score": float(brier_score_loss(y_test, probs)),
        "f1": float(f1_score(y_test, preds)),
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    set_seed(SEED)
    print(f"[train] loading dataset from {args.data}")
    X, y = load_dataset(args.data)
    print(f"[train] X shape={X.shape}, y shape={y.shape}, "
          f"positive_rate={y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

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
        X_train_s, y_train,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Threshold tuning on a held-out slice of TRAINING data that the
    # model did not see during fit (the validation_split portion).
    # We approximate this by using the last 1000 training rows, which
    # the validation_split in model.fit() would have withheld.
    n = len(X_train_s)
    val_slice = slice(max(0, n - 1000), n)
    val_probs = model.predict(X_train_s[val_slice], verbose=0).ravel()
    threshold = find_optimal_threshold(y_train[val_slice], val_probs)
    print(f"[train] chosen threshold = {threshold:.4f}")

    metrics = evaluate(model, X_test_s, y_test, threshold)
    print("[train] test metrics:")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("roc_curve", "pr_curve", "classification_report")},
                     indent=2))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "my_model.keras"
    scaler_path = out_dir / "scaler.pkl"
    names_path = out_dir / "feature_names.json"
    metrics_path = out_dir / "metrics.json"
    history_path = out_dir / "history.json"

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(names_path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(history_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    # Regenerate integrity checksums and training metadata so the
    # committed artifacts are always self-describing.
    write_sha256sums(out_dir)
    write_training_metadata(
        out_dir, model, history, metrics, len(FEATURE_NAMES),
        args, X.shape[0],
    )

    print(f"[train] saved artifacts to {out_dir}:")
    for p in (model_path, scaler_path, names_path, metrics_path, history_path,
              out_dir / "SHA256SUMS", out_dir / "training_metadata.json"):
        print(f"  - {p.name} ({p.stat().st_size} bytes)")


def write_sha256sums(out_dir: Path) -> None:
    """Write SHA256SUMS for all reproducible artifacts."""
    import hashlib
    names = ["my_model.keras", "scaler.pkl", "feature_names.json",
             "metrics.json", "history.json", "dataset_phishing.csv"]
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
    """Write training_metadata.json with versions, seed, and metrics digest."""
    from datetime import datetime, timezone
    import platform
    meta = {
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "seed": SEED,
        "model_architecture": {
            "type": "MLP",
            "total_trainable_params": int(model.count_params()),
            "layers": [
                {"type": "Input", "shape": [n_features]},
                {"type": "Dense", "units": 64, "activation": "relu", "l2": 1e-4},
                {"type": "Dropout", "rate": 0.3},
                {"type": "Dense", "units": 32, "activation": "relu", "l2": 1e-4},
                {"type": "Dropout", "rate": 0.3},
                {"type": "Dense", "units": 1, "activation": "sigmoid"},
            ],
            "loss": "binary_crossentropy",
            "optimizer": "Adam",
            "initial_learning_rate": 1e-3,
        },
        "feature_count": n_features,
        "dataset_rows": int(n_rows),
        "split": {"test_size": 0.2, "stratify": True, "random_state": SEED},
        "training": {
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "validation_split": args.val_split,
            "early_stopping_patience": args.patience,
            "reduce_lr_on_plateau": True,
            "epochs_run": len(history.history["loss"]),
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


if __name__ == "__main__":
    main()
