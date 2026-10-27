import json
import joblib
import pandas as pd
from csu.config import paths
from csu.target import build_target, candidate_columns, load_360, split


def test_360_has_one_row_per_customer_month(env):
    f, _ = load_360(env)
    assert not f.duplicated(["customer_number", "ym"]).any()
    assert {"cust_age", "cust_lor", "PIT_LOAN_CNT_FLAG_OW1", "LOAN_CNT_OW6", "CC_ACQ_FLG_PW3"} <= set(f.columns)


def test_no_leakage_columns_and_customer_disjoint_split(env):
    f, meta = load_360(env)
    for p in env["products"]:
        cols = candidate_columns(f, p, env, meta)
        pre = env["products"][p]["prefix"]
        assert not any("_PW" in c for c in cols)
        assert not any(c.startswith(pre + "_") or f"_{pre}_" in c for c in cols)
        tr, te, oot = split(build_target(f, p, env), env)
        assert not set(tr.customer_number) & set(te.customer_number)
        assert tr.ym.max() < oot.ym.min()


def test_models_and_reports_written(env):
    P = paths(env)
    for p in env["products"]:
        b = joblib.load(P["models"] / f"{p}_cross_sell" / "bundle.joblib")
        r = json.loads((P["reports"] / f"{p}_cross_sell" / "metrics.json").read_text())
        assert len(r["models"]) == 6 and b["baseline"]["oot_gini"] > 0.1
        assert 2 <= len(b["features"]) <= env["woe"]["max_features"]


def test_recommendations_are_eligible_and_ranked(env):
    rec = pd.read_parquet(paths(env)["gold"] / "recommendations.parquet")
    assert set(rec["product"]) == set(env["products"])
    assert rec.groupby("customer_number").rank_in_customer.min().eq(1).all()
    assert (rec.expected_value >= 0).all() and rec.propensity.between(0, 1).all()


def test_drift_demo_triggers_retrain_and_clean_run_does_not(env):
    from csu.monitor import monitor_all
    assert not any(r["retrain_required"] for r in monitor_all(env))
    drifted = monitor_all(env, drift_demo=True)
    assert all(r["status"] == "ALERT" and r["retrain_required"] for r in drifted)


def test_stream_consumer_validates_and_flags_drift(env):
    from csu.stream import consume, produce
    produce(env, 600, seed=1)
    s1 = consume(env)
    assert s1["events"] == 600 and s1.get("note") == "baseline created"
    produce(env, 600, drift=True, seed=2)
    s2 = consume(env)
    assert s2["rejected"] > 0 and s2["alert"] is True
