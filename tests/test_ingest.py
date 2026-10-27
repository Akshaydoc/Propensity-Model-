import pandas as pd
from csu.ingest import validate


def test_bad_rows_are_quarantined_not_silently_dropped():
    df = pd.DataFrame({"customer_number": ["1", None, "3"], "onboarding_date": ["2020-01-01"] * 3, "incorporation_date": ["2010-01-01"] * 3,
                       "industry_desc": ["x"] * 3, "sector_code": ["S1"] * 3, "cust_llc_flag": ["Y"] * 3, "cust_sub_segment": ["SEG_A"] * 3, "rm_id": ["RM01"] * 3})
    clean, bad, issues = validate("dim_customer", df)
    assert len(clean) == 2 and len(bad) == 1 and issues


def test_missing_columns_are_fatal():
    clean, bad, issues = validate("dim_customer", pd.DataFrame({"customer_number": ["1"]}))
    assert clean.empty and "missing columns" in issues[0]
