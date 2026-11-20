-- Raw MLB Stats API responses (schedule, teams, players), one row per landed file.
-- The whole response is stored as VARIANT so bronze doesn't depend on the API's nested
-- schema; silver extracts the fields it needs.
CREATE OR REFRESH STREAMING TABLE statsapi_responses
COMMENT 'Raw MLB Stats API JSON responses, one row per landed file.'
AS SELECT
  regexp_extract(_metadata.file_path, '/statsapi/([^/]+)/', 1) AS endpoint,
  parse_json(value) AS payload,
  _metadata.file_path AS _source_file,
  _metadata.file_modification_time AS _source_file_modified_at,
  current_timestamp() AS _ingested_at
FROM STREAM read_files(
  '${landing_path}/statsapi/',
  format => 'text',
  wholeText => true
);
