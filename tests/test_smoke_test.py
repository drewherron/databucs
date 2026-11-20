from databucs.jobs.smoke_test import missing_schemas, parse_args


def test_parse_args_splits_schemas():
    args = parse_args(["--catalog", "workspace", "--schemas", "dev_bronze, dev_silver,,dev_gold"])
    assert args.catalog == "workspace"
    assert args.schemas == ["dev_bronze", "dev_silver", "dev_gold"]


def test_missing_schemas_reports_only_absent_ones():
    assert missing_schemas(["bronze", "silver"], ["default", "BRONZE"]) == ["silver"]


def test_missing_schemas_empty_when_all_present():
    assert missing_schemas(["bronze"], ["bronze", "gold"]) == []
