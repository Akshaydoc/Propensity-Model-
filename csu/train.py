"""Train + compare 6 models per product on WOE features; pick the champion on out-of-time Gini; log to MLflow."""
import json
import logging
import time
from datetime import datetime
import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from .config import paths
from .target import build_target, candidate_columns, load_360, split
from .tracking import get_mlflow, log_model_and_register
from .woe import apply_woe, assign_bins, fit_binning, select_variables, variable_power

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)
THR_GRID = np.round(np.linspace(0.05, 0.95, 91), 2)


def make_models(seed=42):
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=50, n_jobs=-1, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, subsample=.8, random_state=seed),
        "ada_boost": AdaBoostClassifier(n_estimators=100, learning_rate=.5, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=400, early_stopping=True, random_state=seed),
        "naive_bayes": GaussianNB(),
    }


def decile_table(y, p, n=10):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = np.floor(np.arange(len(d)) * n / len(d)).astype(int) + 1
    g = d.groupby("decile").agg(count=("y", "size"), events=("y", "sum"), avg_score=("p", "mean"))
    g["event_rate"] = g.events / g["count"]
    g["lift"] = g.event_rate / max(d.y.mean(), 1e-9)
    g["cum_events"] = g.events.cumsum()
    g["capture_rate"] = g.cum_events / max(d.y.sum(), 1)
    g["cum_lift"] = g.capture_rate / (g["count"].cumsum() / len(d))
    return g.reset_index()


def metrics(y, p, thr=.5):
    y = np.asarray(y)
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    pred = (p >= thr).astype(int)
    dt = decile_table(y, p)
    return dict(auc=auc, gini=2 * auc - 1, ks=float(np.max(tpr - fpr)), pr_auc=average_precision_score(y, p),
                accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0), threshold=float(thr),
                top10_capture=float(dt.capture_rate.iloc[0]), top20_capture=float(dt.capture_rate.iloc[1]), top10_lift=float(dt.lift.iloc[0]))


def best_threshold(y, p):
    f1s = np.array([f1_score(y, (p >= t).astype(int), zero_division=0) for t in THR_GRID])
    return float(THR_GRID[f1s.argmax()]), f1s


def _fit(model, X, y, w):
    try:
        model.fit(X, y, sample_weight=w)
    except TypeError:
        model.fit(X, y)
    return model


def _lift_plot(dt, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar(dt.decile, dt.lift, color="#2f6fed"); ax[0].axhline(1, color="grey", ls="--")
    ax[0].set(title=f"Lift by decile - {title}", xlabel="decile (1 = highest score)", ylabel="lift")
    ax[1].plot(np.r_[0, dt.decile * 10], np.r_[0, dt.capture_rate * 100], marker="o", color="#2f6fed")
    ax[1].plot([0, 100], [0, 100], "--", color="grey")
    ax[1].set(title="Cumulative capture", xlabel="% of customers contacted", ylabel="% of converters captured")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def train_product(cfg, product, f=None, meta=None, mode=None, use_mlflow=True):
    P = paths(cfg)
    mode = mode or cfg["target"]["mode"]
    if f is None:
        f, meta = load_360(cfg)
    d = build_target(f, product, cfg, mode)
    tr, te, oot = split(d, cfg)
    cols = candidate_columns(f, product, cfg, meta)
    wcfg = cfg["woe"]
    tab, specs = fit_binning(tr, tr.target, cols, wcfg["max_bins"], wcfg["top_categories"])
    chosen, iv = select_variables(tab, specs, tr, wcfg)
    if len(chosen) < 2:
        raise RuntimeError(f"{product}: fewer than 2 variables passed the IV filter")
    X = {k: apply_woe(x, specs, chosen) for k, x in (("train", tr), ("test", te), ("oot", oot))}
    y = {"train": tr.target.to_numpy(), "test": te.target.to_numpy(), "oot": oot.target.to_numpy()}
    power = variable_power(X["train"], tr.target)
    r = y["train"].mean()
    w = np.where(y["train"] == 1, (1 - r) / r, 1.0)  # cost-sensitive weights instead of SMOTE
    ml = get_mlflow(cfg) if use_mlflow else None
    run = ml.start_run(run_name=f"{product}_{mode}") if ml else None
    if ml:
        ml.log_params(dict(product=product, mode=mode, n_train=len(tr), n_test=len(te), n_oot=len(oot), event_rate=round(r, 4),
                           n_candidates=len(cols), n_features=len(chosen), features=",".join(chosen)[:490]))
    fitted, rows, probs = {}, [], {}
    for name, model in make_models(cfg["seed"]).items():
        t0 = time.time()
        _fit(model, X["train"], y["train"], w)
        pr = {k: model.predict_proba(X[k])[:, 1] for k in X}
        thr, f1s = best_threshold(y["test"], pr["test"])
        row = dict(model=name, fit_seconds=round(time.time() - t0, 2), train_gini=2 * roc_auc_score(y["train"], pr["train"]) - 1)
        for k in ("test", "oot"):
            for t_name, t in (("", .5), ("_tuned", thr)):
                for mk, mv in metrics(y[k], pr[k], t).items():
                    if t_name == "" or mk in ("accuracy", "precision", "recall", "f1", "threshold"):
                        row[f"{k}{t_name}_{mk}"] = mv
        row["gini_gap_train_test"] = row["train_gini"] - row["test_gini"]
        rows.append(row); fitted[name] = model; probs[name] = pr
        log.info("%-14s %-19s test gini %.3f | oot gini %.3f | tuned thr %.2f", product, name, row["test_gini"], row["oot_gini"], thr)
        if ml:
            with ml.start_run(run_name=name, nested=True):
                ml.log_param("model", name)
                ml.log_metrics({k: float(v) for k, v in row.items() if k != "model" and isinstance(v, (int, float, np.floating))})
    comp = pd.DataFrame(rows).sort_values("oot_gini", ascending=False).reset_index(drop=True)
    champ = comp.model.iloc[0]
    pc = probs[champ]
    thr, f1s = best_threshold(y["test"], pc["test"])
    f1_oot = [f1_score(y["oot"], (pc["oot"] >= t).astype(int), zero_division=0) for t in THR_GRID]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pc["test"], y["test"])  # calibrated propensity for expected value
    dt_oot = decile_table(y["oot"], pc["oot"])
    rank_corr = -float(spearmanr(dt_oot.decile, dt_oot.event_rate).statistic)  # +1 = perfectly rank-ordered
    cuts = np.quantile(pc["oot"], np.linspace(0, 1, 11)[1:-1])
    ref_shares = {v: assign_bins(tr[v], specs[v]).value_counts(normalize=True).to_dict() for v in chosen}
    bundle = dict(product=product, mode=mode, model_name=champ, model=fitted[champ], specs={v: specs[v] for v in chosen}, features=chosen,
                  threshold=thr, calibrator=iso, score_cuts=cuts, ref_score=pc["train"][:5000], ref_shares=ref_shares,
                  ref_sample=tr[chosen].sample(min(5000, len(tr)), random_state=1).reset_index(drop=True),
                  baseline=dict(test_gini=float(comp.test_gini.iloc[0]), oot_gini=float(comp.oot_gini.iloc[0])),
                  trained_at=datetime.now().isoformat(timespec="seconds"), last_matured_ym=int(d.ym.max()))
    mdir = P["models"] / f"{product}_{mode}"
    rdir = P["reports"] / f"{product}_{mode}"
    mdir.mkdir(parents=True, exist_ok=True); rdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, mdir / "bundle.joblib", compress=3)
    _lift_plot(dt_oot, f"{product} ({champ})", rdir / "lift.png")
    tab[tab.variable_name.isin(chosen)].to_csv(rdir / "binning.csv", index=False)
    report = dict(product=product, mode=mode, label=cfg["products"][product]["label"], champion=champ, threshold=thr, trained_at=bundle["trained_at"],
                  sizes=dict(train=len(tr), test=len(te), oot=len(oot), train_events=int(y["train"].sum()), test_events=int(y["test"].sum()), oot_events=int(y["oot"].sum())),
                  features=chosen, n_candidates=len(cols), models=comp.round(4).to_dict("records"), oot_deciles=dt_oot.round(4).to_dict("records"),
                  rank_order_spearman=rank_corr, f1_sweep=dict(threshold=THR_GRID.tolist(), test=np.round(f1s, 4).tolist(), oot=np.round(f1_oot, 4).tolist()),
                  variable_power=power.round(4).to_dict("records"), iv=iv.head(25).round(4).to_dict(),
                  monotonic={v: specs[v]["monotonic"] for v in chosen}, baseline=bundle["baseline"])
    (rdir / "metrics.json").write_text(json.dumps(report, default=float))
    if ml:
        with ml.start_run(run_name=f"champion_{champ}", nested=True):
            ml.log_param("model", champ)
            ml.log_metrics({"oot_gini": float(comp.oot_gini.iloc[0]), "test_gini": float(comp.test_gini.iloc[0]), "threshold": thr})
            ml.log_artifact(str(rdir / "lift.png")); ml.log_artifact(str(rdir / "metrics.json")); ml.log_artifact(str(rdir / "binning.csv"))
            report["registry_version"] = log_model_and_register(ml, fitted[champ], f"csu_{product}_{mode}")
        ml.end_run()
    return report
