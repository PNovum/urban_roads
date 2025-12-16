CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS mart;

--- как пришло из файлов ---

CREATE TABLE IF NOT EXISTS raw.taxi (
  id BIGSERIAL PRIMARY KEY,
  no_start TEXT NOT NULL,
  no_end   TEXT NOT NULL,
  transport_type TEXT,
  vehicle_type   TEXT,
  trip_cnt DOUBLE PRECISION,
  session_dur DOUBLE PRECISION
);

-- lt.csv
CREATE TABLE IF NOT EXISTS raw.lt (
  id BIGSERIAL PRIMARY KEY,
  start_district_nm TEXT NOT NULL,
  end_district_nm   TEXT NOT NULL,
  transport_type TEXT,
  vehicle_type   TEXT,
  trip_cnt DOUBLE PRECISION,
  trip_dur DOUBLE PRECISION
);

-- dist_cent.csv
CREATE TABLE IF NOT EXISTS raw.dist_cent (
  id BIGSERIAL PRIMARY KEY,
  fid BIGINT,
  no BIGINT,
  district_code TEXT,
  district_name TEXT,
  x DOUBLE PRECISION,
  y DOUBLE PRECISION
);

--- витрины для продукта ---

CREATE TABLE IF NOT EXISTS mart.od_flow (
  run_ts TIMESTAMPTZ NOT NULL,
  a TEXT NOT NULL,
  b TEXT NOT NULL,

  taxi_time DOUBLE PRECISION,
  lt_time   DOUBLE PRECISION,
  delta     DOUBLE PRECISION,
  taxi_faster BOOLEAN NOT NULL,

  taxi_trips DOUBLE PRECISION,
  lt_trips   DOUBLE PRECISION,

  PRIMARY KEY (run_ts, a, b)
);

CREATE OR REPLACE VIEW mart.od_flow_latest AS
SELECT *
FROM mart.od_flow
WHERE run_ts = (SELECT max(run_ts) FROM mart.od_flow);
