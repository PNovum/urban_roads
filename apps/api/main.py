from __future__ import annotations

import os
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Query


def get_db_dsn() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url[len("postgresql+psycopg://") :]
    if url:
        return url

    host = os.getenv("PG_HOST", os.getenv("PGHOST", "localhost"))
    port = os.getenv("PG_PORT", os.getenv("PGPORT", "5432"))
    db = os.getenv("PG_DB", os.getenv("PGDATABASE", "postgres"))
    user = os.getenv("PG_USER", os.getenv("PGUSER", "postgres"))
    password = os.getenv("PG_PASSWORD", os.getenv("PGPASSWORD", "postgres"))
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def fetchall_dict(cur: psycopg.Cursor) -> list[dict[str, Any]]:
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


DB_DSN = get_db_dsn()
app = FastAPI(title="Urban Roads API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/links/by-start")
def links_by_start(
    no_start: int = Query(...),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    q = """
    SELECT *
    FROM mart.no_links
    WHERE no_start = %s
    ORDER BY no_end
    LIMIT %s OFFSET %s;
    """
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(q, (no_start, limit, offset))
            rows = fetchall_dict(cur)
    return {"no_start": no_start, "count": len(rows), "items": rows}


@app.get("/links/one")
def link_one(no_start: int, no_end: int):
    q = """
    SELECT *
    FROM mart.no_links
    WHERE no_start = %s AND no_end = %s;
    """
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(q, (no_start, no_end))
            rows = fetchall_dict(cur)
    if not rows:
        raise HTTPException(status_code=404, detail="not found")
    return rows[0]
