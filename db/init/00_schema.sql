CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS mart;

CREATE TABLE IF NOT EXISTS raw.taxi (
  id BIGSERIAL PRIMARY KEY,
  no_start BIGINT NOT NULL,
  no_end   BIGINT NOT NULL,
  transport_type TEXT,
  vehicle_type   TEXT,
  trip_cnt DOUBLE PRECISION,
  session_dur DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS raw_taxi_no_idx ON raw.taxi (no_start, no_end);

CREATE TABLE IF NOT EXISTS raw.lt (
  id BIGSERIAL PRIMARY KEY,
  start_district_nm TEXT NOT NULL,
  end_district_nm   TEXT NOT NULL,
  transport_type TEXT,
  vehicle_type   TEXT,
  trip_cnt DOUBLE PRECISION,
  trip_dur DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS raw.dist_cent (
  fid BIGINT,
  no BIGINT PRIMARY KEY,
  name TEXT NOT NULL,
  feature_1  DOUBLE PRECISION,
  feature_2  DOUBLE PRECISION,
  feature_3  DOUBLE PRECISION,
  feature_4  DOUBLE PRECISION,
  feature_5  DOUBLE PRECISION,
  feature_6  DOUBLE PRECISION,
  feature_7  DOUBLE PRECISION,
  feature_8  DOUBLE PRECISION,
  feature_9  DOUBLE PRECISION,
  feature_10 DOUBLE PRECISION,
  feature_11 DOUBLE PRECISION,
  feature_12 DOUBLE PRECISION,
  feature_13 DOUBLE PRECISION,
  feature_14 DOUBLE PRECISION,
  feature_15 DOUBLE PRECISION,
  feature_16 DOUBLE PRECISION,
  feature_17 DOUBLE PRECISION,
  feature_18 DOUBLE PRECISION,
  feature_19 DOUBLE PRECISION,
  feature_20 DOUBLE PRECISION,
  feature_21 DOUBLE PRECISION,
  feature_22 DOUBLE PRECISION,
  x DOUBLE PRECISION,
  y DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS mart.taxi_agg (
  no_start BIGINT NOT NULL,
  no_end   BIGINT NOT NULL,
  trip_cnt DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (no_start, no_end)
);

CREATE TABLE IF NOT EXISTS mart.no_links (
  no_start BIGINT NOT NULL,
  no_end   BIGINT NOT NULL,
  name     TEXT NOT NULL,
  feature_1_start  DOUBLE PRECISION,
  feature_2_start  DOUBLE PRECISION,
  feature_3_start  DOUBLE PRECISION,
  feature_4_start  DOUBLE PRECISION,
  feature_5_start  DOUBLE PRECISION,
  feature_6_start  DOUBLE PRECISION,
  feature_7_start  DOUBLE PRECISION,
  feature_8_start  DOUBLE PRECISION,
  feature_9_start  DOUBLE PRECISION,
  feature_10_start DOUBLE PRECISION,
  feature_11_start DOUBLE PRECISION,
  feature_12_start DOUBLE PRECISION,
  feature_13_start DOUBLE PRECISION,
  feature_14_start DOUBLE PRECISION,
  feature_15_start DOUBLE PRECISION,
  feature_16_start DOUBLE PRECISION,
  feature_17_start DOUBLE PRECISION,
  feature_18_start DOUBLE PRECISION,
  feature_19_start DOUBLE PRECISION,
  feature_20_start DOUBLE PRECISION,
  feature_21_start DOUBLE PRECISION,
  feature_22_start DOUBLE PRECISION,
  feature_1_end  DOUBLE PRECISION,
  feature_2_end  DOUBLE PRECISION,
  feature_3_end  DOUBLE PRECISION,
  feature_4_end  DOUBLE PRECISION,
  feature_5_end  DOUBLE PRECISION,
  feature_6_end  DOUBLE PRECISION,
  feature_7_end  DOUBLE PRECISION,
  feature_8_end  DOUBLE PRECISION,
  feature_9_end  DOUBLE PRECISION,
  feature_10_end DOUBLE PRECISION,
  feature_11_end DOUBLE PRECISION,
  feature_12_end DOUBLE PRECISION,
  feature_13_end DOUBLE PRECISION,
  feature_14_end DOUBLE PRECISION,
  feature_15_end DOUBLE PRECISION,
  feature_16_end DOUBLE PRECISION,
  feature_17_end DOUBLE PRECISION,
  feature_18_end DOUBLE PRECISION,
  feature_19_end DOUBLE PRECISION,
  feature_20_end DOUBLE PRECISION,
  feature_21_end DOUBLE PRECISION,
  feature_22_end DOUBLE PRECISION,
  x_start DOUBLE PRECISION,
  y_start DOUBLE PRECISION,
  x_end   DOUBLE PRECISION,
  y_end   DOUBLE PRECISION,
  trip_cnt DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (no_start, no_end)
);
CREATE INDEX IF NOT EXISTS mart_no_links_start_idx ON mart.no_links (no_start);
CREATE INDEX IF NOT EXISTS mart_no_links_end_idx ON mart.no_links (no_end);

CREATE TABLE IF NOT EXISTS mart.ur_model_runs (
  run_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  model_name TEXT NOT NULL,
  run_id TEXT,
  train_rows BIGINT NOT NULL,
  test_rows  BIGINT NOT NULL,
  pos_rate DOUBLE PRECISION NOT NULL,
  roc_auc DOUBLE PRECISION,
  pr_auc  DOUBLE PRECISION,
  f1      DOUBLE PRECISION,
  precision DOUBLE PRECISION,
  recall DOUBLE PRECISION,
  PRIMARY KEY (run_ts, model_name)
);

CREATE OR REPLACE VIEW mart.ur_model_runs_latest AS
SELECT *
FROM mart.ur_model_runs
WHERE run_ts = (SELECT max(run_ts) FROM mart.ur_model_runs);

CREATE TABLE IF NOT EXISTS mart.ur_trip_preds (
  run_ts TIMESTAMPTZ NOT NULL,
  no_start BIGINT NOT NULL,
  no_end   BIGINT NOT NULL,
  trip_cnt_pred DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (run_ts, no_start, no_end)
);
CREATE INDEX IF NOT EXISTS ur_trip_preds_no_start_idx ON mart.ur_trip_preds (no_start);
CREATE INDEX IF NOT EXISTS ur_trip_preds_no_end_idx ON mart.ur_trip_preds (no_end);
