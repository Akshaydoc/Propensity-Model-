"""Feature layer: lookups -> product snapshots -> customer views -> rebased (OW/PW) views -> customer 360 mart.

Same idea as a notebook-per-product build, but one generic implementation driven by config:
every metric is rebased into OW1..OW6 (6 observation months, OW1 = reference month) and PW1..PW3
(3 performance months), then statistics (mean/std/max/min/sum/median/growth/vol/recency/freq) and
point-in-time flags are generated for every metric.
"""
import json
import logging
import numpy as np
import pandas as pd
from .config import paths

log = logging.getLogger(__name__)
STATS = ["mean", "std", "max", "min", "sum", "median", "growth", "vol", "recency", "freq"]
PRODUCT_TABLES = {"casa": "CASA", "term_loan": "LOAN", "funded_trade": "TRADE", "credit_card": "CC", "fd": "FD", "lc": "LC", "lg": "LG"}
TYPES_EXTRA = {"fd": ["Fixed Deposits"], "lc": ["LC"], "lg": ["LG"]}


def _load(P):
    B = {f.stem: pd.read_parquet(f) for f in P["bronze"].glob("*.parquet") if f.stem != "txn_events"}
    for t in ("sme_portfolio", "casa_txn_monthly", "cc_txn_monthly", "loan_snapshot"):
        B[t]["timekey"] = B[t]["timekey"].astype("int64")
    B["fact_balance_sheet"]["time_key"] = B["fact_balance_sheet"]["time_key"].astype("int64")
    return B


def month_lookup(cfg):
    """month-end lookup (one row per month) + the rolling OW/PW definition table."""
    T, s = cfg["data"]["n_months"], cfg["data"]["start_ym"]
    per = pd.period_range(f"{s // 100}-{s % 100:02d}", periods=T, freq="M")
    lk = pd.DataFrame({"ym": per.year * 100 + per.month, "y": per.year, "m": per.month,
                       "timekey": per.end_time.strftime("%Y%m%d").astype(int), "month_end": per.end_time.normalize()})
    w = cfg["windows"]
    rw = pd.DataFrame([("OW%d" % k, k, -(k - 1)) for k in range(1, w["obs"] + 1)] +
                      [("PW%d" % k, k, k) for k in range(1, w["perf"] + 1)], columns=["window", "k", "offset"])
    return lk, rw


def snapshot(B, types, excl):
    """portfolio -> + balance fact (+ currency, accounts); exclusions + 4-digit GL rule applied like the source pipeline."""
    p = B["sme_portfolio"]
    p = p[p.product_type.isin(types) & ~p.exclusion_1.isin(excl)].copy()
    p["gl_no_1"] = p.gl_no.astype(str).str.zfill(4)
    p = p[p.gl_no_1.str.len() == 4]
    fb = B["fact_balance_sheet"].copy()
    fb["gl_no_1"] = fb.gl_no.astype(str).str.zfill(4)
    s = p.merge(fb[["time_key", "acc_deal_number", "gl_no_1", "currency_key", "deal_balance"]],
                left_on=["account_no", "timekey", "gl_no_1"], right_on=["acc_deal_number", "time_key", "gl_no_1"], how="left")
    s = s.merge(B["dim_currency"], on="currency_key", how="left")
    acc = B["dim_accounts"].query("latest_flag == 'Y'")[["account_number", "account_open_date", "account_status"]]
    s = s.merge(acc, left_on="account_no", right_on="account_number", how="left")
    s["tdate"] = pd.to_datetime(s.timekey.astype(str), format="%Y%m%d")
    s["mob"] = ((s.tdate - s.account_open_date).dt.days / 365 * 12).round()
    s["acq_flg"] = ((s.account_open_date.dt.year * 100 + s.account_open_date.dt.month) == s.timekey // 100).astype(int)
    s["deal_balance"] = s.deal_balance.fillna(0)
    return s


def _agg(s, pre, extra):
    g = s.groupby(["customer_number", "timekey"])
    out = g.agg(CNT=("account_no", "nunique"), BAL=("bal", "sum"), MAX_MOB=("mob", "max"), ACQ_FLG=("acq_flg", "max"), **extra)
    out.columns = [f"{pre}_{c}" for c in out.columns]
    return out.reset_index()


def customer_views(B, cfg):
    excl = cfg["exclusions"]
    views = {}
    # CASA: overdrawn handling, average balance, sub-types
    s = snapshot(B, cfg["catalog"]["casa"]["types"], excl)
    s = s.merge(B["casa_avg_balance"], left_on=["account_no", "tdate"], right_on=["account_number", "period_date"], how="left", suffixes=("", "_ab"))
    s["od"] = (s.closing_balance < 0).astype(int)
    s["nd"] = 1 - s.od
    s["bal"] = s.deal_balance * s.nd
    s["avg_bal"] = s.monthly_average_balance.fillna(0) * s.nd
    s["cur"] = ((s.product_type == "Current Accounts") & (s.nd == 1)).astype(int)
    s["sav"] = ((s.product_type == "Savings Account") & (s.nd == 1)).astype(int)
    v = _agg(s, "CASA", dict(AVG_BAL=("avg_bal", "sum"), OVERDRAWN_CNT=("od", "sum"), CURRENT_CNT=("cur", "sum"), SAVINGS_CNT=("sav", "sum")))
    v["CASA_CNT"] = v["CASA_CNT"] - v["CASA_OVERDRAWN_CNT"]
    views["casa"] = v
    # loan / funded trade: delinquency from loan snapshot
    for prod, pre in (("term_loan", "LOAN"), ("funded_trade", "TRADE")):
        s = snapshot(B, cfg["catalog"][prod]["types"], excl)
        s = s.merge(B["loan_snapshot"], left_on=["account_no", "timekey"], right_on=["contract_id", "timekey"], how="left")
        s["bal"] = s.outstanding.fillna(s.deal_balance)
        s["del"] = (s.dpd.fillna(0) > 90).astype(int)
        views[prod] = _agg(s, pre, dict(DEL_FLG=("del", "max"), DPD_MAX=("dpd", "max")))
    for prod, pre in (("credit_card", "CC"),):
        s = snapshot(B, cfg["catalog"][prod]["types"], excl)
        s["bal"] = s.deal_balance
        views[prod] = _agg(s, pre, {})
    for prod, pre in (("fd", "FD"), ("lc", "LC"), ("lg", "LG")):
        s = snapshot(B, TYPES_EXTRA[prod], excl)
        s["bal"] = s.deal_balance
        views[prod] = _agg(s, pre, {})
        views[prod] = views[prod].drop(columns=[f"{pre}_ACQ_FLG"])
    # transactions
    t = B["casa_txn_monthly"].pivot_table(index=["customer_number", "timekey"], columns="txn_type", values=["txn_cnt", "txn_amt"], aggfunc="sum", fill_value=0)
    t.columns = [f"CASA_TXN_{b}_{a.split('_')[1].upper()}" for a, b in t.columns]
    views["casa_txn"] = t.reset_index()
    c = B["cc_txn_monthly"].groupby(["customer_number", "timekey"]).agg(CC_TXN_CNT=("txn_cnt", "sum"), CC_TXN_AMT=("txn_amt", "sum"), CC_TXN_CASH_AMT=("cash_amt", "sum"))
    views["cc_txn"] = c.reset_index()
    return views


def rebase(views, customers, yms, cfg):
    """customer x ym grid -> metric_OW1..6 / metric_PW1..3 + statistics + point-in-time flags."""
    ow, pw = cfg["windows"]["obs"], cfg["windows"]["perf"]
    n, T = len(customers), len(yms)
    js = np.arange(ow - 1, T)
    J = len(js)
    cidx = pd.Index(customers)
    pw_keep = {f"{pre}_{m}" for pre in ("CASA", "LOAN", "TRADE", "CC") for m in ("ACQ_FLG", "BAL", "CNT")}
    out = {"customer_number": np.tile(np.asarray(customers), J), "ym": np.repeat(yms[js], n)}
    meta = {}
    for name, v in views.items():
        v = v.assign(ym=v.timekey // 100)
        for m in [c for c in v.columns if c not in ("customer_number", "timekey", "ym")]:
            grid = v.set_index(["customer_number", "ym"])[m].unstack("ym").reindex(index=cidx, columns=yms).fillna(0)
            arr = grid.to_numpy(np.float32)
            owk = [arr[:, js - (k - 1)].T.ravel() for k in range(1, ow + 1)]
            for k, a in enumerate(owk, 1):
                out[f"{m}_OW{k}"] = a
                meta[f"{m}_OW{k}"] = m
            if m in pw_keep:
                for k in range(1, pw + 1):
                    idx = js + k
                    ok = idx < T
                    a = np.full((n, J), np.nan, np.float32)
                    a[:, ok] = arr[:, idx[ok]]
                    out[f"{m}_PW{k}"] = a.T.ravel()
                    meta[f"{m}_PW{k}"] = m
            if m.endswith("_CNT"):
                for k, a in enumerate(owk, 1):
                    out[f"PIT_{m}_FLAG_OW{k}"] = (a > 0).astype(np.int8)
                    meta[f"PIT_{m}_FLAG_OW{k}"] = m
            if m.endswith("_FLG"):
                continue
            st = np.stack(owk)
            mean, std = st.mean(0), st.std(0)
            b = st[3:].sum(0)
            rec = np.zeros_like(mean)
            for k in range(ow, 0, -1):
                rec = np.where(st[k - 1] != 0, ow + 1 - k, rec)
            vals = dict(mean=mean, std=std, max=st.max(0), min=st.min(0), sum=st.sum(0), median=np.median(st, 0),
                        growth=np.where(b != 0, st[:3].sum(0) / np.where(b == 0, 1, b), np.nan),
                        vol=np.where(mean != 0, std / np.where(mean == 0, 1, mean), np.nan), recency=rec, freq=(st != 0).sum(0))
            for s_, a in vals.items():
                out[f"{s_}_{m}"] = a.astype(np.float32)
                meta[f"{s_}_{m}"] = m
    return pd.DataFrame(out), meta


def build_features(cfg):
    P = paths(cfg)
    B = _load(P)
    lk, rw = month_lookup(cfg)
    lk.to_parquet(P["silver"] / "month_end_dates.parquet", index=False)
    rw.to_parquet(P["silver"] / "rolling_window.parquet", index=False)
    views = customer_views(B, cfg)
    for k, v in views.items():
        v.to_parquet(P["gold"] / f"{k}_cust_view.parquet", index=False)
    port = B["sme_portfolio"]
    excluded = set(port.loc[port.exclusion_1.isin(cfg["exclusions"]), "customer_number"])
    dimc = B["dim_customer"]
    customers = sorted(set(dimc.customer_number) & set(port.customer_number) - excluded)
    f, meta = rebase(views, customers, lk.ym.to_numpy(), cfg)
    f["ymd"] = pd.to_datetime(f.ym.astype(str) + "01") + pd.offsets.MonthEnd(0)
    f = f.merge(dimc, on="customer_number", how="left")
    f["cust_age"] = ((f.ymd - f.incorporation_date).dt.days / 365.25).round(1).astype("float32")
    f["cust_lor"] = ((f.ymd - f.onboarding_date).dt.days / 30.4375).astype(int).astype("float32")
    f = f.drop(columns=["ymd", "onboarding_date", "incorporation_date"])
    f.to_parquet(P["gold"] / "cust_360.parquet", index=False)
    (P["gold"] / "col_meta.json").write_text(json.dumps(meta))
    log.info("customer 360: %d rows x %d cols, %d customers", len(f), f.shape[1], len(customers))
    return f.shape
