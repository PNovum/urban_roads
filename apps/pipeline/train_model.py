from __future__ import annotations

import os
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
import psycopg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_db_dsn() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url[len("postgresql+psycopg://") :]
    return url


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "urban_roads")
    mlflow.set_experiment(experiment_name)

    dsn = get_db_dsn()
    with psycopg.connect(dsn) as conn:
        df = pd.read_sql("SELECT * FROM mart.no_links", conn)

    y = (df["trip_cnt"].fillna(0.0).to_numpy() > 0.0).astype(int)

    drop_cols = {"no_start", "no_end", "name", "trip_cnt"}
    X = df[[c for c in df.columns if c not in drop_cols]].copy()

    X["euclid_dist"] = np.sqrt((X["x_start"] - X["x_end"]) ** 2 + (X["y_start"] - X["y_end"]) ** 2)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1)),
        ]
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        pos_rate = float(y.mean())
        roc_auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None
        pr_auc = float(average_precision_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None
        f1 = float(f1_score(y_test, pred, zero_division=0))
        precision = float(precision_score(y_test, pred, zero_division=0))
        recall = float(recall_score(y_test, pred, zero_division=0))

        mlflow.log_param("model", "logreg_balanced")
        mlflow.log_param("threshold", 0.5)
        mlflow.log_param("features", X.shape[1])
        mlflow.log_metric("pos_rate", pos_rate)
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)
        if pr_auc is not None:
            mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        mlflow.sklearn.log_model(model, artifact_path="model")

        run_ts = datetime.now(timezone.utc)

        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mart.ur_model_runs
                      (run_ts, model_name, run_id, train_rows, test_rows, pos_rate, roc_auc, pr_auc, f1, precision, recall)
                    VALUES
                      (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_ts,
                        "logreg_balanced",
                        run.info.run_id,
                        int(X_train.shape[0]),
                        int(X_test.shape[0]),
                        pos_rate,
                        roc_auc,
                        pr_auc,
                        f1,
                        precision,
                        recall,
                    ),
                )
            conn.commit()


if __name__ == "__main__":
    main()
