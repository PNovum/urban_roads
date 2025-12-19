from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import psycopg


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


def load_latest_model_from_mlflow() -> dict:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    exp_name = os.getenv("MLFLOW_EXPERIMENT", "urban_roads")
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {exp_name}")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string='attributes.status = "FINISHED"',
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No FINISHED runs found in MLflow experiment")

    run_id = runs[0].info.run_id
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    model_path = os.path.join(local_dir, "model.joblib")
    return {"run_id": run_id, "payload": joblib.load(model_path)}


def fetch_dist_cent(conn) -> pd.DataFrame:
    q = """
    SELECT
      no, name,
      feature_1, feature_2, feature_3, feature_4, feature_5,
      feature_6, feature_7, feature_8, feature_9, feature_10,
      feature_11, feature_12, feature_13, feature_14, feature_15,
      feature_16, feature_17, feature_18, feature_19, feature_20,
      feature_21, feature_22,
      x, y
    FROM raw.dist_cent
    """
    with conn.cursor() as cur:
        cur.execute(q)
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def build_X(df: pd.DataFrame) -> np.ndarray:
    xs = df["x_start"].to_numpy(dtype=np.float32)
    ys = df["y_start"].to_numpy(dtype=np.float32)
    xe = df["x_end"].to_numpy(dtype=np.float32)
    ye = df["y_end"].to_numpy(dtype=np.float32)
    dist = np.sqrt((xs - xe) ** 2 + (ys - ye) ** 2).astype(np.float32)

    X = np.empty((len(df), len(FEATURE_COLS) + 1), dtype=np.float32)
    for j, c in enumerate(FEATURE_COLS):
        X[:, j] = df[c].to_numpy(dtype=np.float32)
    X[:, len(FEATURE_COLS)] = dist
    return X


def main() -> None:
    dsn = get_db_dsn()
    data_dir = Path(os.getenv("DATA_DIR", "/data"))
    taxi_path = data_dir / "taxi.csv"

    taxi = pd.read_csv(taxi_path)
    taxi["no_start"] = pd.to_numeric(taxi.get("no_start"), errors="coerce")
    taxi["no_end"] = pd.to_numeric(taxi.get("no_end"), errors="coerce")
    taxi = taxi.dropna(subset=["no_start", "no_end"])
    taxi["no_start"] = taxi["no_start"].astype("int64")
    taxi["no_end"] = taxi["no_end"].astype("int64")

    with psycopg.connect(dsn) as conn:
        dist = fetch_dist_cent(conn)

    dist["no"] = dist["no"].astype("int64")
    dist = dist.drop_duplicates(subset=["no"])

    a = dist.rename(
        columns={
            "no": "no_start",
            "x": "x_start",
            "y": "y_start",
            **{f"feature_{i}": f"feature_{i}_start" for i in range(1, 23)},
        }
    )
    b = dist.rename(
        columns={
            "no": "no_end",
            "x": "x_end",
            "y": "y_end",
            **{f"feature_{i}": f"feature_{i}_end" for i in range(1, 23)},
        }
    )

    df = taxi.merge(a, on="no_start", how="left").merge(b, on="no_end", how="left")
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    model_info = load_latest_model_from_mlflow()
    payload = model_info["payload"]
    scaler = payload["scaler"]
    clf = payload["clf"]
    thr = float(payload.get("threshold", float(os.getenv("INFER_THRESHOLD", "0.7"))))

    X = build_X(df)
    Xs = scaler.transform(X)
    proba = clf.predict_proba(Xs)[:, 1].astype(np.float32)
    trip_cnt_pred = (proba >= thr).astype(np.int64)

    run_ts = datetime.now(timezone.utc)
    out = pd.DataFrame(
        {
            "run_ts": run_ts.isoformat(),
            "no_start": df["no_start"].astype("int64"),
            "no_end": df["no_end"].astype("int64"),
            "trip_cnt_pred": trip_cnt_pred.astype("int64"),
        }
    )

    filled = taxi.copy()
    filled["trip_cnt"] = trip_cnt_pred.astype("int64")
    filled_path = data_dir / "taxi_filled.csv"
    filled.to_csv(filled_path, index=False)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM mart.ur_trip_preds WHERE run_ts = %s", (run_ts,))
            rows = list(out.itertuples(index=False, name=None))
            buf = "\n".join(",".join(map(str, r)) for r in rows) + "\n"
            with cur.copy(
                "COPY mart.ur_trip_preds (run_ts, no_start, no_end, trip_cnt_pred) FROM STDIN WITH CSV"
            ) as cp:
                cp.write(buf)
        conn.commit()


if __name__ == "__main__":
    main()
