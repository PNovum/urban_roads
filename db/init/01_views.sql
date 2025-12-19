CREATE OR REPLACE VIEW mart.ur_trip_preds_latest AS
WITH last AS (
  SELECT max(run_ts) AS run_ts
  FROM mart.ur_trip_preds
)
SELECT
  p.run_ts,
  p.no_start,
  a.name AS name_start,
  p.no_end,
  b.name AS name_end,
  p.trip_cnt_pred
FROM mart.ur_trip_preds p
JOIN last t ON p.run_ts = t.run_ts
LEFT JOIN raw.dist_cent a ON a.no = p.no_start
LEFT JOIN raw.dist_cent b ON b.no = p.no_end;
