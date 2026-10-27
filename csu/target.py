"""Population + binary target per product.

cross_sell : customer does NOT hold the product in OW1..OW6 -> target 1 if acquired in PW1..PW3
upsell     : customer holds the product at OW1             -> target 1 if balance in PW3 >= (1+g) x balance in OW1
Non-events come from *every* observation month (the old single-month sampling skewed time), and
train/test are split by customer so the same client never sits on both sides.
"""
import json
import numpy as np
import pandas as pd
from .config import paths, prefix

CAT_COLS = ["industry_desc", "cust_llc_flag", "cust_sub_segment", "sector_code"]


def population_mask(f, pre, cfg, mode=None):
    mode = mode or cfg["target"]["mode"]
    ow = cfg["windows"]["obs"]
    base = (f.cust_lor >= cfg["target"]["min_lor_months"]) & (f.LOAN_DEL_FLG_OW1 == 0) & (f.TRADE_DEL_FLG_OW1 == 0)
    # active relationship: holds at least one product at the reference month (the old "sum_status > 0" rule)
    base &= f[[c for c in f.columns if c.startswith("PIT_") and c.endswith("_CNT_FLAG_OW1")]].sum(axis=1) > 0
    if mode == "cross_sell":
        held = f[[f"{pre}_CNT_OW{k}" for k in range(1, ow + 1)]].sum(axis=1) > 0
        return base & ~held
    return base & (f[f"{pre}_CNT_OW1"] > 0) & (f[f"{pre}_BAL_OW1"] > 0)


def build_target(f, product, cfg, mode=None):
    mode = mode or cfg["target"]["mode"]
    pre = prefix(cfg, product)
    pw = cfg["windows"]["perf"]
    d = f[population_mask(f, pre, cfg, mode) & (f.ym >= cfg["split"]["first_obs_ym"]) & f[f"{pre}_CNT_PW{pw}"].notna()].copy()
    if mode == "cross_sell":
        d["target"] = (d[[f"{pre}_ACQ_FLG_PW{k}" for k in range(1, pw + 1)]].max(axis=1) > 0).astype(int)
    else:
        g = cfg["target"]["upsell_growth"]
        d["target"] = ((d[f"{pre}_CNT_PW{pw}"] > 0) & (d[f"{pre}_BAL_PW{pw}"] >= (1 + g) * d[f"{pre}_BAL_OW1"])).astype(int)
    return d


def candidate_columns(f, product, cfg, col_meta):
    """Drops performance-window columns, OW2-OW6 raw columns, PIT flags beyond OW1, the product's own metrics and constants."""
    pre = prefix(cfg, product)
    own = lambda m: m == pre or m.startswith(pre + "_")
    keep = []
    for c, m in col_meta.items():
        if c not in f.columns or "_PW" in c or own(m):
            continue
        if any(c.endswith(f"_OW{k}") for k in range(2, cfg["windows"]["obs"] + 1)):
            continue
        keep.append(c)
    keep += ["cust_age", "cust_lor"] + CAT_COLS
    return [c for c in keep if f[c].nunique(dropna=True) > 1]


def split(d, cfg, seed=None):
    s = cfg["split"]
    dev = d[d.ym <= s["train_end_ym"]]
    oot = d[d.ym >= s["oot_start_ym"]]
    rng = np.random.default_rng(s["random_state"])
    custs = rng.permutation(np.array(sorted(dev.customer_number.unique()), dtype=object))
    test_c = set(custs[: int(len(custs) * s["test_size"])])
    is_test = dev.customer_number.isin(test_c)
    return dev[~is_test], dev[is_test], oot


def load_360(cfg):
    P = paths(cfg)
    f = pd.read_parquet(P["gold"] / "cust_360.parquet")
    meta = json.loads((P["gold"] / "col_meta.json").read_text())
    return f, meta
