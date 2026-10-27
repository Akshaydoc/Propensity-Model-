"""Scoring + next-best-offer ranking: calibrated propensity x revenue potential, filtered by the eligibility policy."""
import logging
import joblib
import numpy as np
import pandas as pd
from .config import paths, prefix
from .target import load_360, population_mask
from .woe import apply_woe

log = logging.getLogger(__name__)


def load_bundle(cfg, product, mode=None):
    mode = mode or cfg["target"]["mode"]
    return joblib.load(paths(cfg)["models"] / f"{product}_{mode}" / "bundle.joblib")


def score_frame(bundle, df):
    W = apply_woe(df, bundle["specs"], bundle["features"])
    raw = bundle["model"].predict_proba(W)[:, 1]
    decile = 10 - np.searchsorted(bundle["score_cuts"], raw, side="right")
    return raw, np.clip(bundle["calibrator"].predict(raw), 0, 1), decile


def policy_mask(df, pol):
    ok = (df.cust_lor >= pol.min_lor_months) & (df.cust_age >= pol.min_cust_age)
    ok &= df.cust_sub_segment.isin([s for s in str(pol.allowed_segments).split(",") if s])
    blocked = [s for s in str(pol.blocked_industries or "").split(",") if s and s != "nan"]
    return ok & ~df.industry_desc.isin(blocked)


def score_product(cfg, product, f, B, mode=None, ym=None):
    mode = mode or cfg["target"]["mode"]
    bundle = load_bundle(cfg, product, mode)
    ym = ym or int(f.ym.max())
    cur = f[f.ym == ym]
    size_pct = cur["sum_CASA_TXN_C_AMT"].rank(pct=True)
    pop = cur[population_mask(cur, prefix(cfg, product), cfg, mode)]
    pol = B["eligibility_policy"].set_index("product").loc[product]
    pol["product"] = product
    elig = pop[policy_mask(pop, pol)]
    raw, prob, dec = score_frame(bundle, elig)
    rev = B["product_revenue"].set_index("product").loc[product]
    revenue = float(rev.avg_ticket) * float(rev.margin_rate) * (0.5 + 1.5 * size_pct.loc[elig.index].to_numpy())
    return pd.DataFrame({
        "customer_number": elig.customer_number.to_numpy(), "rm_id": elig.rm_id.to_numpy(), "product": product,
        "label": cfg["products"][product]["label"], "propensity_raw": raw.round(4), "propensity": prob.round(4), "decile": dec,
        "contact": (raw >= bundle["threshold"]).astype(int), "revenue_potential": revenue.round(0),
        "expected_value": (np.maximum(prob, 0.001) * revenue).round(1), "ym": ym})


def score_all(cfg, f=None, mode=None):
    P = paths(cfg)
    mode = mode or cfg["target"]["mode"]
    if f is None:
        f, _ = load_360(cfg)
    B = {n: pd.read_parquet(P["bronze"] / f"{n}.parquet") for n in ("eligibility_policy", "product_revenue")}
    rec = pd.concat([score_product(cfg, p, f, B, mode) for p in cfg["products"]], ignore_index=True)
    rec["rank_in_customer"] = rec.groupby("customer_number").expected_value.rank(ascending=False, method="first").astype(int)
    rec["rank_in_rm"] = rec.groupby(["rm_id", "product"]).expected_value.rank(ascending=False, method="first").astype(int)
    rec = rec.sort_values(["customer_number", "rank_in_customer"]).reset_index(drop=True)
    rec.to_parquet(P["gold"] / "recommendations.parquet", index=False)
    rec[rec.rank_in_customer == 1].sort_values("expected_value", ascending=False).to_csv(P["reports"] / "next_best_offer.csv", index=False)
    log.info("recommendations: %d rows, %d customers", len(rec), rec.customer_number.nunique())
    return rec
