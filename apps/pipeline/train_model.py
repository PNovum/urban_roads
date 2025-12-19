from __future__ import annotations

import os
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import psycopg
from sklearn.linear_model import SGDClassifier
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
        tv = 0.0 if trip is None else float(trip)
        y[r_i] = 1 if tv >= 1.0 else 0

    return X, y


def mape_bin(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.astype(np.float32)
    yp = y_pred.astype(np.float32)
    denom = np.maximum(yt, 1.0)
    return float(np.mean(np.abs(yt - yp) / denom))


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_m = 1e9
    for t in np.linspace(0.05, 0.95, 19):
        pred = (proba >= t).astype(np.int8)
        m = mape_bin(y_true, pred)
        if m < best_m:
            best_m = m
            best_t = float(t)
    return best_t, float(best_m)


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
        mlflow.log_param("model", "sgd_log_loss_streaming")
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("holdout_size", holdout_size)
        mlflow.log_param("total_rows", total_rows)
        mlflow.log_metric("holdout_pos_rate", float(y_hold.mean()))
        mlflow.log_metric("w0", float(w0))
        mlflow.log_metric("w1", float(w1))

        with psycopg.connect(dsn) as conn:
            with conn.cursor(name="stream_no_links_train") as cur:
                cur.itersize = batch_size
                cur.execute(train_query)

                cols = [d.name for d in cur.description]
                col_idx = {c: i for i, c in enumerate(cols)}

                first = True
                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break

                    Xb, yb = rows_to_xy(rows, col_idx)
                    sw = np.where(yb == 1, w1, w0).astype(np.float32)

                    scaler.partial_fit(Xb)
                    Xb_s = scaler.transform(Xb)

                    if first:
                        clf.partial_fit(Xb_s, yb, classes=classes, sample_weight=sw)
                        first = False
                    else:
                        clf.partial_fit(Xb_s, yb, sample_weight=sw)

                    seen += len(rows)

        Xh_s = scaler.transform(X_hold)
        proba = clf.predict_proba(Xh_s)[:, 1].astype(np.float32)

        thr, holdout_mape = best_threshold(y_hold, proba)

        mlflow.log_metric("seen_train_rows", int(seen))
        mlflow.log_param("threshold", float(thr))
        mlflow.log_metric("holdout_mape", float(holdout_mape))

        os.makedirs("/tmp/ur", exist_ok=True)
        path = "/tmp/ur/model.joblib"
        joblib.dump(
            {"scaler": scaler, "clf": clf, "threshold": float(thr), "features": FEATURE_COLS + ["euclid_dist"]},
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
                      (%s, %s, %s, %s, %s, %s, NULL, NULL, NULL, NULL, NULL)
                    """,
                    (
                        run_ts,
                        "sgd_bin_trip_cnt",
                        run.info.run_id,
                        int(seen),
                        int(len(y_hold)),
                        float(holdout_mape),
                    ),
                )
            conn.commit()


if __name__ == "__main__":
    main()
