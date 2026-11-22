-- Every pitch that failed a data quality rule (see silver_pitch_checks.sql), with the rules
-- it failed and the original values the rules checked.
--
-- action says what happened to the pitch in silver.pitches:
--   rejected       the pitch is left out (a row rule failed)
--   values_nulled  the pitch is kept, with the implausible measurements set to NULL
CREATE OR REFRESH MATERIALIZED VIEW ${catalog}.${silver_schema}.pitches_quarantine
CLUSTER BY (game_date)
COMMENT 'Pitches that failed a data quality rule: the rules failed, the action taken, and the original values.'
AS
SELECT
  game_pk,
  at_bat_number,
  pitch_number,
  game_date,
  pitcher_id,
  batter_id,
  failed_rules,
  if(is_rejected, 'rejected', 'values_nulled') AS action,
  checked_values AS original_values,
  _source_file,
  _source_file_modified_at
FROM pitch_checks
WHERE size(failed_rules) > 0;
