"""
Lab 16 - Benchmark LightGBM on Credit Card Fraud Detection dataset (CPU flow).

Steps:
1. Load dataset, split train/test.
2. Train an LGBMClassifier for fraud detection.
3. Measure data-load time and training time.
4. Evaluate on the test set: AUC-ROC, Accuracy, F1, Precision, Recall.
5. Measure inference latency (1 row) and throughput (1000 rows).
6. Dump all results to benchmark_result.json.
"""

import json
import time
import platform
import multiprocessing

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

DATA_PATH = "/home/ubuntu/ml-benchmark/creditcard.csv"
RESULT_PATH = "/home/ubuntu/ml-benchmark/benchmark_result.json"
RANDOM_STATE = 42


def main():
    results = {}

    # --- 1 & 3a. Load dataset, measure load time ---
    t0 = time.perf_counter()
    df = pd.read_csv(DATA_PATH)
    load_time = time.perf_counter() - t0
    print(f"[Load] {df.shape[0]} rows x {df.shape[1]} cols in {load_time:.3f}s")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[Split] train={X_train.shape[0]} test={X_test.shape[0]} "
          f"fraud_rate_train={y_train.mean():.5f} fraud_rate_test={y_test.mean():.5f}")

    # --- 2 & 3b. Train LGBMClassifier, measure training time ---
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    train_time = time.perf_counter() - t0
    best_iteration = model.best_iteration_
    print(f"[Train] {train_time:.3f}s, best_iteration={best_iteration}")

    # --- 4. Evaluate on test set ---
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"[Eval] AUC-ROC={auc:.5f} Accuracy={acc:.5f} F1={f1:.5f} "
          f"Precision={precision:.5f} Recall={recall:.5f}")

    # --- 5a. Inference latency: predict 1 row ---
    single_row = X_test.iloc[[0]]
    # warmup
    for _ in range(10):
        model.predict(single_row)

    n_repeats = 200
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(single_row)
    latency_total = time.perf_counter() - t0
    latency_ms = (latency_total / n_repeats) * 1000
    print(f"[Latency] avg over {n_repeats} runs: {latency_ms:.3f} ms/row")

    # --- 5b. Inference throughput: predict 1000 rows ---
    batch = X_test.iloc[:1000] if len(X_test) >= 1000 else X_test
    # warmup
    model.predict(batch)

    t0 = time.perf_counter()
    model.predict(batch)
    throughput_time = time.perf_counter() - t0
    throughput_rows_per_sec = len(batch) / throughput_time
    print(f"[Throughput] {len(batch)} rows in {throughput_time*1000:.3f} ms "
          f"=> {throughput_rows_per_sec:.1f} rows/sec")

    # --- 6. Dump results ---
    results = {
        "environment": {
            "instance_type": "t3.medium (CPU)",
            "python_version": platform.python_version(),
            "lightgbm_version": lgb.__version__,
            "cpu_count": multiprocessing.cpu_count(),
            "platform": platform.platform(),
        },
        "dataset": {
            "name": "mlg-ulb/creditcardfraud",
            "total_rows": int(df.shape[0]),
            "total_cols": int(df.shape[1]),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "fraud_rate_total": float(y.mean()),
        },
        "timing": {
            "load_data_time_sec": round(load_time, 4),
            "training_time_sec": round(train_time, 4),
            "best_iteration": int(best_iteration),
        },
        "metrics": {
            "auc_roc": round(float(auc), 5),
            "accuracy": round(float(acc), 5),
            "f1_score": round(float(f1), 5),
            "precision": round(float(precision), 5),
            "recall": round(float(recall), 5),
        },
        "inference": {
            "latency_ms_per_row": round(latency_ms, 4),
            "throughput_rows_per_sec": round(throughput_rows_per_sec, 1),
            "throughput_batch_size": int(len(batch)),
            "throughput_total_time_ms": round(throughput_time * 1000, 4),
        },
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {RESULT_PATH}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
