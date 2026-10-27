"""Model + data monitoring: schema/data-quality checks, PSI (score), CSI (WOE bins), KS (raw features),
matured-performance check, Evidently HTML report and an automatic retrain decision."""
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from .config import paths, prefix
from .score import load_bundle, score_frame
from .target import build_target, load_360, population_mask
from .woe import assign_bins

log = logging.getLogger(__name__)
EPS = 1e-4


def psi_samples(ref, cur, bins=10):
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref) + EPS
    a = np.histogram(cur, edges)[0] / len(cur) + EPS
    return float(np.sum((a - e) * np.log(a / e)))


def psi_shares(ref: dict, cur: dict):
    """Population/Characteristic stability index on pre-binned shares."""
    keys = set(ref) | set(cur)
    e = np.array([ref.get(k, 0) + EPS for k in keys]); a = np.array([cur.get(k, 0) + EPS for k in keys])
    return float(np.sum((a - e) * np.log(a / e)))


def gini(y, p):
    from sklearn.metrics import roc_auc_score
    return 2 * roc_auc_score(y, p) - 1


def schema_checks(cur, bundle, max_null):
    issues = []
    miss = [c for c in bundle["features"] if c not in cur.columns]
    if miss:
        issues.append(dict(severity="critical", msg=f"missing model features: {miss}"))
    for c in [c for c in bundle["features"] if c in cur.columns]:
        nr = float(cur[c].isna().mean())
        if nr > max_null:
            issues.append(dict(severity="warning", msg=f"{c}: null rate {nr:.0%}"))
        sp = bundle["specs"][c]
        if sp["type"] == "numerical" and not pd.api.types.is_numeric_dtype(cur[c]):
            issues.append(dict(severity="critical", msg=f"{c}: expected numeric"))
        if sp["type"] == "categorical":
            new = set(cur[c].dropna().astype(str)) - set(sp["top"])
            share = float(cur[c].astype(str).isin(new).mean()) if new else 0.0
            if share > .05:
                issues.append(dict(severity="warning", msg=f"{c}: {share:.0%} rows in unseen categories"))
    return issues


def perturb(cur, bundle):
    """drift demo: simulate an upstream shift in the model's own inputs (+1.5 std on numeric features, 30% of rows re-labelled in categoricals)."""
    c = cur.copy()
    for v in bundle["features"]:
        sp = bundle["specs"][v]
        if sp["type"] == "numerical":
            c[v] = c[v] + 1.5 * (float(bundle["ref_sample"][v].std()) or 1.0)
        else:
            c.loc[c.sample(frac=.3, random_state=3).index, v] = sp["top"][-1]
    return c


def evidently_html(ref, cur, path):
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
        r = Report([DataDriftPreset()]).run(Dataset.from_pandas(cur, data_definition=DataDefinition()),
                                            Dataset.from_pandas(ref, data_definition=DataDefinition()))
        r.save_html(str(path))
        return True
    except Exception as e1:
        try:  # older evidently API
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
            r = Report(metrics=[DataDriftPreset()])
            r.run(reference_data=ref, current_data=cur)
            r.save_html(str(path))
            return True
        except Exception as e2:
            log.warning("evidently report skipped (%s | %s)", str(e1)[:80], str(e2)[:80])
            return False


def run_monitor(cfg, product, f=None, mode=None, drift_demo=False):
    P = paths(cfg)
    m = cfg["monitoring"]
    mode = mode or cfg["target"]["mode"]
    bundle = load_bundle(cfg, product, mode)
    if f is None:
        f, _ = load_360(cfg)
    ym = int(f.ym.max())
    cur = f[(f.ym == ym)]
    cur = cur[population_mask(cur, prefix(cfg, product), cfg, mode)]
    if drift_demo:
        cur = perturb(cur, bundle)
    issues = schema_checks(cur, bundle, m["max_null_rate"])
    rep = dict(product=product, mode=mode, ym=ym, n_current=len(cur), checked_at=datetime.now().isoformat(timespec="seconds"),
               drift_demo=drift_demo, schema_issues=issues, model=bundle["model_name"], trained_at=bundle["trained_at"])
    if any(i["severity"] == "critical" for i in issues):
        rep.update(status="BLOCKED", retrain_required=False, reasons=["critical schema issue - scoring blocked"])
    else:
        raw, _, _ = score_frame(bundle, cur)
        rep["score_psi"] = round(psi_samples(bundle["ref_score"], raw), 4)
        csi = {v: round(psi_shares(bundle["ref_shares"][v], assign_bins(cur[v], bundle["specs"][v]).value_counts(normalize=True).to_dict()), 4) for v in bundle["features"]}
        ks = {}
        for v in bundle["features"]:
            if bundle["specs"][v]["type"] == "numerical":
                r = ks_2samp(bundle["ref_sample"][v].dropna(), cur[v].dropna())
                ks[v] = dict(stat=round(float(r.statistic), 4), p=float(r.pvalue), drift=bool(r.pvalue < m["ks_alpha"] and r.statistic > .1))
        rep["csi"], rep["ks"] = csi, ks
        # performance on the latest matured months (labels available), vs the baseline OOT Gini
        mat = build_target(f, product, cfg, mode)
        mat = mat[mat.ym >= int(mat.ym.max()) - 2]
        if mat.target.sum() >= 30:
            rm_raw, _, _ = score_frame(bundle, mat)
            rep["recent_gini"] = round(gini(mat.target, rm_raw), 4)
            rep["gini_drop"] = round(bundle["baseline"]["oot_gini"] - rep["recent_gini"], 4)
        reasons = []
        if rep["score_psi"] >= m["psi_alert"]:
            reasons.append(f"score PSI {rep['score_psi']:.2f} >= {m['psi_alert']}")
        n_csi = [v for v, x in csi.items() if x >= m["csi_alert"]]
        if len(n_csi) >= m["min_alert_features"]:
            reasons.append(f"{len(n_csi)} features with CSI >= {m['csi_alert']}")
        if rep.get("gini_drop", 0) >= m["gini_drop_alert"]:
            reasons.append(f"Gini dropped {rep['gini_drop']:.2f}")
        warn = rep["score_psi"] >= m["psi_warn"] or any(x >= m["psi_warn"] for x in csi.values()) or any(k["drift"] for k in ks.values()) or bool(issues)
        rep.update(status="ALERT" if reasons else ("WARN" if warn else "OK"), retrain_required=bool(reasons), reasons=reasons)
        feats = bundle["features"]
        ref_df = bundle["ref_sample"][feats].copy(); ref_df["score"] = bundle["ref_score"][: len(ref_df)] if len(bundle["ref_score"]) >= len(ref_df) else np.resize(bundle["ref_score"], len(ref_df))
        cur_df = cur[feats].reset_index(drop=True).copy(); cur_df["score"] = raw
        rep["evidently_report"] = evidently_html(ref_df, cur_df, P["monitoring"] / f"{product}_{mode}_evidently.html")
    (P["monitoring"] / f"{product}_{mode}_latest.json").write_text(json.dumps(rep, default=float, indent=1))
    h = P["monitoring"] / "monitoring_history.csv"
    row = {k: rep.get(k) for k in ("checked_at", "product", "ym", "status", "score_psi", "recent_gini", "gini_drop", "retrain_required", "drift_demo")}
    row["max_csi"] = max(rep.get("csi", {0: 0}).values())
    pd.DataFrame([row]).to_csv(h, mode="a", header=not h.exists(), index=False)
    log.info("monitor %-13s %-7s score_psi=%s max_csi=%s retrain=%s", product, rep["status"], rep.get("score_psi"), row["max_csi"], rep["retrain_required"])
    return rep


def monitor_all(cfg, drift_demo=False, auto_retrain=False, mode=None):
    from .train import train_product
    f, meta = load_360(cfg)
    out = []
    for p in cfg["products"]:
        r = run_monitor(cfg, p, f, mode, drift_demo)
        if r["retrain_required"] and auto_retrain and not drift_demo:
            log.warning("auto-retrain triggered for %s: %s", p, r["reasons"])
            train_product(cfg, p, f, meta, mode)
            r["retrained"] = True
        out.append(r)
    return out
