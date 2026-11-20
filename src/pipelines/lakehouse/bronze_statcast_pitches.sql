-- Raw Statcast pitches, one row per pitch per landed file. All columns are kept as strings;
-- typing and cleaning happen in silver.
--
-- A corrected re-pull of a day lands as a new file, so the same pitch can appear more than
-- once here. Silver keeps the rows from the latest file for each game date.
CREATE OR REFRESH STREAMING TABLE statcast_pitches
COMMENT 'Raw Statcast pitch-level rows from Baseball Savant (strings, as landed).'
AS SELECT
  *,
  _metadata.file_path AS _source_file,
  _metadata.file_modification_time AS _source_file_modified_at,
  current_timestamp() AS _ingested_at
FROM STREAM read_files(
  '${landing_path}/statcast/',
  format => 'csv',
  header => true,
  inferColumnTypes => false,
  multiLine => true
);
