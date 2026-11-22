-- Cleaned, typed Statcast pitches: one row per pitch, keyed on
-- (game_pk, at_bat_number, pitch_number).
--
-- Built from pitch_checks (silver_pitch_checks.sql), which types the latest bronze file per
-- day and applies the data quality rules. Rejected pitches are left out, and implausible
-- tracking values are already NULL; both are listed in pitches_quarantine.
--
-- Dropped from bronze: columns Savant marks deprecated or leaves empty (spin_dir,
-- *_deprecated, umpire, sv_id), the legacy age columns, and player_name (the pitcher's name;
-- names come from the players table).
CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${silver_schema}.pitches
CLUSTER BY (game_date, pitcher_id)
COMMENT 'Statcast pitches, typed, deduplicated, and validated. One row per pitch.'
AS
SELECT * EXCEPT (checks, failed_rules, is_rejected, checked_values)
FROM pitch_checks
WHERE NOT is_rejected;
