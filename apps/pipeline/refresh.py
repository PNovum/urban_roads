from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pandas as pd
import psycopg


def get_db_dsn() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url[len("postgresql+psycopg://") :]
    return url


def copy_df(cur, df: pd.DataFrame, table: str) -> None:
    buf = StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    cols = ",".join(f'"{c}"' for c in df.columns)
    with cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH CSV") as cp:
        cp.write(buf.getvalue())


def refresh() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "/data"))

    dist = pd.read_csv(data_dir / "dist_cent.csv")
    taxi = pd.read_csv(data_dir / "taxi.csv")

    dist = dist.rename(columns={"NO": "no", "NAME": "name"})
    dist["no"] = pd.to_numeric(dist["no"], errors="coerce")
    dist = dist.dropna(subset=["no"])
    dist["no"] = dist["no"].astype("int64")

    for i in range(1, 23):
        dist[f"feature_{i}"] = pd.to_numeric(dist.get(f"Признак {i}"), errors="coerce")

    dist["x"] = pd.to_numeric(dist.get("x"), errors="coerce")
    dist["y"] = pd.to_numeric(dist.get("y"), errors="coerce")

    dist = dist[
        ["fid", "no", "name"]
        + [f"feature_{i}" for i in range(1, 23)]
        + ["x", "y"]
    ].drop_duplicates(subset=["no"])

    taxi["no_start"] = pd.to_numeric(taxi.get("no_start"), errors="coerce")
    taxi["no_end"] = pd.to_numeric(taxi.get("no_end"), errors="coerce")
    taxi = taxi.dropna(subset=["no_start", "no_end"])
    taxi["no_start"] = taxi["no_start"].astype("int64")
    taxi["no_end"] = taxi["no_end"].astype("int64")

    taxi["trip_cnt"] = pd.to_numeric(taxi.get("trip_cnt"), errors="coerce")
    taxi["session_dur"] = pd.to_numeric(taxi.get("session_dur"), errors="coerce")

    keep = ["no_start", "no_end", "transport_type", "vehicle_type", "trip_cnt", "session_dur"]
    taxi = taxi[[c for c in keep if c in taxi.columns]]

    dsn = get_db_dsn()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE raw.dist_cent;")
            cur.execute("TRUNCATE raw.taxi RESTART IDENTITY;")
            cur.execute("TRUNCATE mart.taxi_agg;")
            cur.execute("TRUNCATE mart.no_links;")

            copy_df(cur, dist, "raw.dist_cent")
            copy_df(cur, taxi, "raw.taxi")

            cur.execute("""
                INSERT INTO mart.taxi_agg
                SELECT no_start, no_end, SUM(COALESCE(trip_cnt, 0))
                FROM raw.taxi
                GROUP BY 1,2
            """)

            cur.execute("""
                INSERT INTO mart.no_links
                SELECT
                  a.no, b.no, a.name,
                  a.feature_1, a.feature_2, a.feature_3, a.feature_4, a.feature_5,
                  a.feature_6, a.feature_7, a.feature_8, a.feature_9, a.feature_10,
                  a.feature_11, a.feature_12, a.feature_13, a.feature_14, a.feature_15,
                  a.feature_16, a.feature_17, a.feature_18, a.feature_19, a.feature_20,
                  a.feature_21, a.feature_22,
                  b.feature_1, b.feature_2, b.feature_3, b.feature_4, b.feature_5,
                  b.feature_6, b.feature_7, b.feature_8, b.feature_9, b.feature_10,
                  b.feature_11, b.feature_12, b.feature_13, b.feature_14, b.feature_15,
                  b.feature_16, b.feature_17, b.feature_18, b.feature_19, b.feature_20,
                  b.feature_21, b.feature_22,
                  a.x, a.y, b.x, b.y,
                  COALESCE(t.trip_cnt, 0)
                FROM raw.dist_cent a
                CROSS JOIN raw.dist_cent b
                LEFT JOIN mart.taxi_agg t
                  ON t.no_start = a.no AND t.no_end = b.no
                WHERE a.no <> b.no
            """)

        conn.commit()


if __name__ == "__main__":
    refresh()
