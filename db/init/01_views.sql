CREATE OR REPLACE VIEW mart.ur_link_scores_latest AS
WITH last AS (
  SELECT max(run_ts) AS run_ts
  FROM mart.ur_link_scores
)
SELECT
  s.run_ts,
  s.no_start,
  a.name AS name_start,
  s.no_end,
  b.name AS name_end,
  s.p_demand,
  s.distance_km,
  COALESCE(l.trip_cnt, 0) AS trip_cnt
FROM mart.ur_link_scores s
JOIN last t ON s.run_ts = t.run_ts
LEFT JOIN raw.dist_cent a ON a.no = s.no_start
LEFT JOIN raw.dist_cent b ON b.no = s.no_end
LEFT JOIN mart.no_links l
  ON l.no_start = s.no_start AND l.no_end = s.no_end;
