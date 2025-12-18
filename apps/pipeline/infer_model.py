from __future__ import annotations

import os
from datetime import datetime, timezone
from io import StringIO

import joblib
import mlflow
import numpy as np
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


def build_X_and_dist(rows: list[tuple], col_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    X = np.empty((len(rows), len(FEATURE_COLS) + 1), dtype=np.float32)
    dist = np.empty((len(rows),), dtype=np.float32)

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
        d = float(np.sqrt((xs - xe) ** 2 + (ys - ye) ** 2))
        dist[r_i] = d
        X[r_i, len(FEATURE_COLS)] = d

    return X, dist


def copy_scores(cur, rows: list[tuple]) -> None:
    buf = StringIO()
    for r in rows:
        buf.write(",".join(str(x) for x in r))
        buf.write("\n")
    buf.seek(0)
    with cur.copy(
        'COPY mart.ur_link_scores (run_ts, no_start, no_end, p_demand, distance_km) FROM STDIN WITH CSV'
    ) as cp:
        cp.write(buf.getvalue())


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


def main() -> None:
    batch_size = int(os.getenv("INFER_BATCH_SIZE", "50000"))
    dsn = get_db_dsn()

    model_info = load_latest_model_from_mlflow()
    model_run_id = model_info["run_id"]
    payload = model_info["payload"]
    scaler = payload["scaler"]
    clf = payload["clf"]

    run_ts = datetime.now(timezone.utc)

    select_sql = f"""
      SELECT
        no_start, no_end,
        {", ".join(FEATURE_COLS)}
      FROM mart.no_links
      ORDER BY no_start, no_end
    """

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as curw:
            curw.execute("DELETE FROM mart.ur_link_scores WHERE run_ts = %s", (run_ts,))
        conn.commit()

        with psycopg.connect(dsn) as connr:
            with connr.cursor(name="stream_no_links_infer") as cur:
                cur.itersize = batch_size
                cur.execute(select_sql)

                cols = [d.name for d in cur.description]
                col_idx = {c: i for i, c in enumerate(cols)}

                out_rows: list[tuple] = []

                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break

                    X, dist = build_X_and_dist(rows, col_idx)
                    Xs = scaler.transform(X)
                    proba = clf.predict_proba(Xs)[:, 1].astype(float)

                    for i, r in enumerate(rows):
                        no_start = int(r[col_idx["no_start"]])
                        no_end = int(r[col_idx["no_end"]])
                        p = float(proba[i])
                        d = float(dist[i])
                        out_rows.append((run_ts.isoformat(), no_start, no_end, p, d))

                    if len(out_rows) >= batch_size:
                        with psycopg.connect(dsn) as connw:
                            with connw.cursor() as curw:
                                copy_scores(curw, out_rows)
                            connw.commit()
                        out_rows = []

                if out_rows:
                    with psycopg.connect(dsn) as connw:
                        with connw.cursor() as curw:
                            copy_scores(curw, out_rows)
                        connw.commit()

    if os.getenv("MLFLOW_TRACKING_URI", "").strip():
        mlflow.set_tag("ur_infer_model_run_id", model_run_id)


if __name__ == "__main__":
    main()
