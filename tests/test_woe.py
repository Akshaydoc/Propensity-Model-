import numpy as np
import pandas as pd
from csu.woe import apply_woe, fit_binning, fit_variable, select_variables


def _data(n=6000, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(size=n)
    y = (r.random(n) < 1 / (1 + np.exp(-(-3 + 1.2 * x)))).astype(int)
    return pd.DataFrame({"good": x, "noise": r.normal(size=n), "cat": r.choice(list("abc"), n), "good_copy": x + r.normal(0, .01, n)}), pd.Series(y)


def test_informative_variable_has_iv_and_noise_does_not():
    df, y = _data()
    tg, _ = fit_variable(df["good"], y)
    tn, _ = fit_variable(df["noise"], y)
    assert tg.iv.sum() > 0.2 and tn.iv.sum() < 0.05


def test_apply_woe_handles_nulls_and_unseen_categories():
    df, y = _data()
    tab, specs = fit_binning(df, y, ["good", "cat"])
    new = pd.DataFrame({"good": [np.nan, 0.1], "cat": ["zzz", None]})
    w = apply_woe(new, specs, ["good", "cat"])
    assert w.shape == (2, 2) and not w.isna().any().any()


def test_selection_drops_correlated_and_weak():
    df, y = _data()
    tab, specs = fit_binning(df, y, list(df.columns))
    chosen, _ = select_variables(tab, specs, df, dict(iv_min=.02, iv_max=5, corr_max=.85, max_features=10))
    assert "noise" not in chosen and not ({"good", "good_copy"} <= set(chosen))
