from __future__ import annotations

import os
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import psycopg
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def get_db_dsn() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url[len("postgresql+psycopg://") :]
    return url


FEATURE_COLS = (
    [f"feature_{i}_start" for i in range(1, 23)]
    + [f"feature_{i}_end" for i in range(1, 23)]
    + ["x_start", "y_start", "x_end", "y_end"]
)


def rows_to_xy(rows: list[tuple], col_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    X = np.empty((len(rows), len(FEATURE_COLS) + 1), dtype=np.float32)
    y = np.empty((len(rows),), dtype=np.int8)

    ix_xs = FEATURE_COLS.index("x_start")
    ix_ys = FEATURE_COLS.index("y_start")
    ix_xe = FEATURE_COLS.index("x_end")
    ix_ye = FEATURE_COLS.index("y_end")

    for r_i, r in enumerate(rows):
        for j, c in enumerate(FEATURE_COLS):
            v = r[col_idx[c]]
            X[r_i, j] = 0.0 if v is None else float(v)

        xs = X[r_i, ix_xs]
        ys = X[r_i, ix_ys]
        xe = X[r_i, ix_xe]
        ye = X[r_i, ix_ye]
        X[r_i, len(FEATURE_COLS)] = float(np.sqrt((xs - xe) ** 2 + (ys - ye) ** 2))

        trip = r[col_idx["trip_cnt"]]
        y[r_i] = 1 if (trip is not None and float(trip) > 0.0) else 0

    return X, y


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "urban_roads")
    mlflow.set_experiment(experiment_name)

    batch_size = int(os.getenv("TRAIN_BATCH_SIZE", "50000"))
    holdout_size = int(os.getenv("TRAIN_HOLDOUT_ROWS", "200000"))

    dsn = get_db_dsn()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM mart.no_links")
            total_rows = int(cur.fetchone()[0])

    holdout_query = f"""
        SELECT {", ".join(FEATURE_COLS)}, trip_cnt
        FROM mart.no_links
        ORDER BY no_start DESC, no_end DESC
        LIMIT {holdout_size}
    """

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(holdout_query)
            cols = [d.name for d in cur.description]
            col_idx = {c: i for i, c in enumerate(cols)}
            holdout_rows = cur.fetchall()

    X_hold, y_hold = rows_to_xy(holdout_rows, col_idx)

    pos = float((y_hold == 1).sum())
    neg = float((y_hold == 0).sum())
    w0 = (pos + neg) / (2.0 * max(neg, 1.0))
    w1 = (pos + neg) / (2.0 * max(pos, 1.0))

    scaler = StandardScaler(with_mean=True, with_std=True)
    clf = SGDClassifier(loss="log_loss", max_iter=1, tol=None)

    classes = np.array([0, 1], dtype=np.int8)
    seen = 0

    train_query = f"""
        SELECT {", ".join(FEATURE_COLS)}, trip_cnt
        FROM mart.no_links
        ORDER BY no_start, no_end
    """

    with mlflow.start_run() as run:
        mlflow.log_param("model", "sgd_log_loss_streaming_sample_weight")
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("holdout_size", holdout_size)
        mlflow.log_param("total_rows", total_rows)
        mlflow.log_metric("holdout_pos_rate", float(y_hold.mean()))
        mlflow.log_metric("w0", float(w0))
        mlflow.log_metric("w1", float(w1))

        with psycopg.connect(dsn) as conn:
            with conn.cursor(name="stream_no_links") as cur:
                cur.itersize = batch_size
                cur.execute(train_query)

                cols = [d.name for d in cur.description]
                col_idx = {c: i for i, c in enumerate(cols)}

                first_batch = True
                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break

                    Xb, yb = rows_to_xy(rows, col_idx)
                    sw = np.where(yb == 1, w1, w0).astype(np.float32)

                    scaler.partial_fit(Xb)
                    Xb_s = scaler.transform(Xb)

                    if first_batch:
                        clf.partial_fit(Xb_s, yb, classes=classes, sample_weight=sw)
                        first_batch = False
                    else:
                        clf.partial_fit(Xb_s, yb, sample_weight=sw)

                    seen += len(rows)

        Xh_s = scaler.transform(X_hold)
        proba = clf.predict_proba(Xh_s)[:, 1]
        pred = (proba >= 0.5).astype(int)

        roc_auc = float(roc_auc_score(y_hold, proba)) if len(np.unique(y_hold)) > 1 else None
        pr_auc = float(average_precision_score(y_hold, proba)) if len(np.unique(y_hold)) > 1 else None
        f1 = float(f1_score(y_hold, pred, zero_division=0))
        precision = float(precision_score(y_hold, pred, zero_division=0))
        recall = float(recall_score(y_hold, pred, zero_division=0))

        mlflow.log_metric("seen_train_rows", int(seen))
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)
        if pr_auc is not None:
            mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        os.makedirs("/tmp/ur", exist_ok=True)
        path = "/tmp/ur/model.joblib"
        joblib.dump(
            {"scaler": scaler, "clf": clf, "features": FEATURE_COLS + ["euclid_dist"]},
            path,
        )
        mlflow.log_artifact(path, artifact_path="model")

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
                        "sgd_streaming",
                        run.info.run_id,
                        int(seen),
                        int(len(y_hold)),
                        float(y_hold.mean()),
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
