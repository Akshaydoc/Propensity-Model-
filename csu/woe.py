"""Binning, WOE, IV, Gini / Somers' D, variable selection (IV cut-off + correlation pruning) and WOE application.

WOE = ln(dist_non_event / dist_event);  IV = sum((dist_non_event - dist_event) * WOE)
Numerical: up to 10 quantile bins, nulls -> bin -1.  Categorical: top-N levels + OTHERS, nulls -> bin -1.
Everything fitted on TRAIN only and re-applied to test / OOT / scoring data through the `spec` dict.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

SMOOTH = 0.5


def is_categorical(s: pd.Series) -> bool:
    n = s.name.lower() if isinstance(s.name, str) else ""
    return ("flag" in n) or ("flg" in n) or not pd.api.types.is_numeric_dtype(s)


def _stats(bins, y, nb):
    g = pd.DataFrame({"b": bins, "y": y}).groupby("b").y.agg(["sum", "count"])
    ev, non = g["sum"].astype(float), (g["count"] - g["sum"]).astype(float)
    de = (ev + SMOOTH) / (ev.sum() + SMOOTH * nb)
    dn = (non + SMOOTH) / (non.sum() + SMOOTH * nb)
    woe = np.log(dn / de)
    return pd.DataFrame({"non_event": non, "event": ev, "dist_non_event": dn, "dist_event": de, "woe": woe, "iv": (dn - de) * woe})


def fit_variable(x: pd.Series, y: pd.Series, max_bins=10, top_cat=12):
    """returns (binning rows, spec) or (None, None) when the variable is constant."""
    name = x.name
    if x.dropna().nunique() < 2:
        return None, None
    if is_categorical(x):
        xs = x.astype("object").where(x.notna(), None)
        top = xs.value_counts().head(top_cat).index.tolist()
        lab = xs.map(lambda v: -1 if v is None else (v if v in top else "OTHERS")).astype(str).replace("-1", "NULL")
        t = _stats(lab, y.to_numpy(), lab.nunique())
        t.insert(0, "bin_label", t.index)
        spec = dict(type="categorical", top=[str(v) for v in top], woe={str(k): float(v) for k, v in t.woe.items()})
        t["min_value"] = t["max_value"] = t.index
        t["bin"] = np.arange(len(t))
    else:
        xn = x.astype(float)
        q = np.unique(np.nanquantile(xn, np.linspace(0, 1, max_bins + 1)))
        inner = q[1:-1]
        b = pd.Series(np.where(xn.isna(), -1, np.searchsorted(inner, xn.fillna(0), side="left")), index=x.index)
        t = _stats(b.to_numpy(), y.to_numpy(), b.nunique())
        t["bin"] = t.index
        t["bin_label"] = ["NULL" if i == -1 else f"BIN_{i}" for i in t.index]
        mm = pd.DataFrame({"b": b.to_numpy(), "x": xn.to_numpy()}).groupby("b").x.agg(["min", "max"])
        t["min_value"], t["max_value"] = mm["min"], mm["max"]
        spec = dict(type="numerical", edges=[float(e) for e in inner], woe={int(k): float(v) for k, v in t.woe.items()})
    t.insert(0, "variable_name", name)
    t.insert(1, "variable_type", spec["type"])
    # monotonic trend check (rank corr between bin order and WOE) - reported, not enforced
    tt = t[t.bin != -1] if spec["type"] == "numerical" else t.iloc[0:0]
    spec["monotonic"] = bool(len(tt) > 2 and abs(pd.Series(tt.woe.to_numpy()).corr(pd.Series(np.arange(len(tt))), method="spearman")) > 0.99)
    t["monotonic"] = spec["monotonic"]
    return t.reset_index(drop=True), spec


def fit_binning(df, y, cols, max_bins=10, top_cat=12):
    tabs, specs = [], {}
    for c in cols:
        t, s = fit_variable(df[c], y, max_bins, top_cat)
        if t is not None:
            tabs.append(t)
            specs[c] = s
    return pd.concat(tabs, ignore_index=True), specs


def assign_bins(x: pd.Series, spec) -> pd.Series:
    """bin identifier per row (used by WOE mapping and by CSI monitoring)."""
    if spec["type"] == "numerical":
        xn = x.astype(float)
        return pd.Series(np.where(xn.isna(), -1, np.searchsorted(spec["edges"], xn.fillna(0), side="left")), index=x.index).astype(int).astype(str)
    xs = x.astype("object").where(x.notna(), None)
    return xs.map(lambda v: "NULL" if v is None else (str(v) if str(v) in spec["top"] else "OTHERS"))


def apply_woe(df, specs, variables):
    out = {}
    for v in variables:
        sp = specs[v]
        b = assign_bins(df[v], sp)
        m = {str(k): w for k, w in sp["woe"].items()}
        fallback = m.get("OTHERS", 0.0)
        out[f"{v}_woe"] = b.map(lambda k: m.get(k, fallback)).astype(float).to_numpy()
    return pd.DataFrame(out, index=df.index)


def variable_iv(tab):
    return tab.groupby("variable_name").iv.sum().sort_values(ascending=False)


def select_variables(tab, specs, df, cfg_woe):
    """IV window (drops weak + suspiciously strong) then greedy correlation pruning on the WOE columns."""
    iv = variable_iv(tab)
    cand = iv[(iv >= cfg_woe["iv_min"]) & (iv <= cfg_woe["iv_max"])].index.tolist()
    if not cand:
        return [], iv
    W = apply_woe(df, specs, cand)
    chosen = []
    for v in cand:
        if len(chosen) >= cfg_woe["max_features"]:
            break
        if all(abs(W[f"{v}_woe"].corr(W[f"{c}_woe"])) <= cfg_woe["corr_max"] for c in chosen):
            chosen.append(v)
    return chosen, iv


def variable_power(W: pd.DataFrame, y: pd.Series):
    """Gini = 2*AUC-1 and Somers' D (same quantity) with a Mann-Whitney p-value for significance."""
    rows = []
    for c in W.columns:
        auc = roc_auc_score(y, W[c])
        p = mannwhitneyu(W[c][y == 1], W[c][y == 0], method="asymptotic").pvalue
        rows.append(dict(feature=c, gini=abs(2 * auc - 1), somers_d=2 * auc - 1, p_value=p, significant=bool(p < .05)))
    return pd.DataFrame(rows).sort_values("gini", ascending=False)
