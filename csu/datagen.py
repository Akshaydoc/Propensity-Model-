"""Synthetic SME banking data generator.

Produces the same table/column layout the feature pipeline expects (portfolio, balance-sheet fact,
accounts, avg balances, loan snapshot, monthly txns, eligibility + revenue tables).
Product acquisitions are driven by a hidden hazard model so that the downstream models have real,
learnable (and slightly non-linear) signal to find.
"""
import logging
import numpy as np
import pandas as pd
from .config import paths

log = logging.getLogger(__name__)
PRODS = ["casa", "term_loan", "funded_trade", "credit_card", "fd", "lc", "lg"]
SCALE = dict(casa=80_000, term_loan=400_000, funded_trade=300_000, credit_card=30_000, fd=200_000, lc=150_000, lg=120_000)
INDUSTRIES = ["Trading", "Manufacturing", "Services", "Construction", "Retail", "Logistics", "Hospitality", "Technology"]
IND_P = [.18, .14, .17, .10, .13, .10, .08, .10]
TRADE_AFF = dict(Trading=1.0, Manufacturing=.8, Logistics=.7, Retail=.3, Construction=.2, Services=.1, Hospitality=.05, Technology=.1)
PTYPES = {"casa": (["Current Accounts", "Savings Account", "Call Accounts", "Margin Accounts"], [.55, .30, .10, .05]),
          "term_loan": (["Business Loans", "Term Loans", "Asset Finance"], [.4, .4, .2]),
          "funded_trade": (["Bills Discounted", "Trust Receipts"], [.6, .4]),
          "credit_card": (["Corporate Cards"], [1.0]), "fd": (["Fixed Deposits"], [1.0]), "lc": (["LC"], [1.0]), "lg": (["LG"], [1.0])}
GL = {"Current Accounts": "1234", "Savings Account": "1240", "Call Accounts": "1250", "Margin Accounts": "1260",
      "Business Loans": "2101", "Term Loans": "2110", "Asset Finance": "2120", "Bills Discounted": "2201",
      "Trust Receipts": "2210", "Corporate Cards": "2301", "Fixed Deposits": "1555", "LC": "0251", "LG": "0261"}
CAT_CODE = {"Current Accounts": "1077", "Savings Account": "1075", "Call Accounts": "1076", "Margin Accounts": "1078"}
PREF = dict(casa="CA", term_loan="TL", funded_trade="FT", credit_card="CC", fd="FD", lc="LC", lg="LG")


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _mstart(start_ym, idx):
    """vectorised: first day of month (start_ym + idx months)"""
    base = np.datetime64(f"{start_ym // 100}-{start_ym % 100:02d}", "M")
    return (base + np.asarray(idx).astype("timedelta64[M]")).astype("datetime64[D]")


def generate(cfg, n_customers=None, seed=None):
    P = paths(cfg)
    rng = np.random.default_rng(seed or cfg["seed"])
    N = n_customers or cfg["data"]["n_customers"]
    T = cfg["data"]["n_months"]
    s_ym = cfg["data"]["start_ym"]
    ms = _mstart(s_ym, np.arange(T))
    me = (ms.astype("datetime64[M]") + 1).astype("datetime64[D]") - np.timedelta64(1, "D")
    timekeys = pd.to_datetime(me).strftime("%Y%m%d").astype(int).to_numpy()

    # ---- customers -------------------------------------------------------------------
    cid = np.array([str(1_000_001 + i) for i in range(N)])
    lor0 = rng.integers(0, 121, N)
    new = rng.random(N) < 0.15
    onb = np.where(new, rng.integers(1, T - 6, N), -lor0)
    inc_years = rng.integers(1, 35, N)
    industry = rng.choice(INDUSTRIES, N, p=IND_P)
    llc = rng.random(N) < 0.45
    seg = rng.choice(["SEG_A", "SEG_B"], N, p=[.7, .3])
    size = rng.lognormal(0, .6, N)
    size_z = np.log(size) / .6
    aff = np.array([TRADE_AFF[i] for i in industry])
    excl = np.where(rng.random(N) < 0.04, rng.choice(cfg["exclusions"], N), None)
    onb_date = _mstart(s_ym, onb) + rng.integers(0, 27, N).astype("timedelta64[D]")
    ptype = {p: rng.choice(PTYPES[p][0], N, p=PTYPES[p][1]) for p in PRODS}

    hold = {p: np.zeros((N, T), bool) for p in PRODS}
    bal = {p: np.zeros((N, T)) for p in PRODS}
    opn = {p: np.full(N, 10 ** 6) for p in PRODS}
    delinq = np.zeros((N, T), bool)
    cr = np.zeros((N, T)); crn = np.zeros((N, T)); dr = np.zeros((N, T)); drn = np.zeros((N, T))
    cca = np.zeros((N, T)); ccn = np.zeros((N, T)); ccc = np.zeros((N, T))

    init_p = dict(casa=.6, term_loan=.15, funded_trade=.07, credit_card=.2, fd=.25, lc=.06, lg=.05)
    for p in PRODS:
        h0 = (onb <= 0) & (rng.random(N) < init_p[p])
        hold[p][:, 0] = h0
        opn[p][h0] = np.maximum(-rng.integers(0, 121, N)[h0], onb[h0])
        bal[p][:, 0] = h0 * size * SCALE[p] * rng.lognormal(0, .4, N)

    def _activity(t):
        a = hold["casa"][:, t]
        act = rng.uniform(.5, 1.5, N) * a
        cr[:, t] = bal["casa"][:, t] * act * rng.uniform(.8, 1.4, N)
        dr[:, t] = cr[:, t] * rng.uniform(.8, 1.0, N)
        crn[:, t] = rng.poisson(18 * np.sqrt(size)) * a
        drn[:, t] = rng.poisson(22 * np.sqrt(size)) * a
        c = hold["credit_card"][:, t]
        cca[:, t] = bal["credit_card"][:, t] * rng.uniform(.1, .7, N) * c
        ccn[:, t] = rng.poisson(14 * np.sqrt(size)) * c
        ccc[:, t] = cca[:, t] * rng.uniform(0, .15, N)
        hl = hold["term_loan"][:, t] | hold["funded_trade"][:, t]
        prev = delinq[:, t - 1] if t else np.zeros(N, bool)
        delinq[:, t] = hl & np.where(prev, rng.random(N) < .7, rng.random(N) < .02)

    _activity(0)
    for t in range(1, T):
        for p in PRODS:
            hold[p][:, t] = hold[p][:, t - 1]
            bal[p][:, t] = bal[p][:, t - 1] * np.exp(rng.normal(.01, .08, N)) * hold[p][:, t - 1]
        h = {p: hold[p][:, t - 1] for p in PRODS}
        casa_z = np.where(h["casa"], (np.log1p(bal["casa"][:, t - 1]) - 11) / 1.5, -1.0)
        txn_z = np.where(h["casa"], (np.log1p(cr[:, t - 1]) - 10.5) / 1.5, -1.0)
        lorm = np.clip((t - 1 - onb) / 60, 0, 2)
        age_z = (inc_years + t / 12 - 15) / 10
        fd, lc, lg_, tl = (h[k].astype(float) for k in ("fd", "lc", "lg", "term_loan"))
        logit = {
            "casa": -5.8 + 1.9 * fd + 1.5 * lc + 1.5 * lg_ + 1.9 * tl + 1.3 * h["credit_card"] + .7 * size_z + .6 * lorm,
            "term_loan": -5.3 + .9 * size_z + .8 * casa_z + .7 * fd + .4 * lorm + .5 * (txn_z > .8),
            "funded_trade": -5.4 + 1.3 * lc + 1.1 * lg_ + 1.0 * aff + .5 * txn_z + .4 * size_z + .6 * ((txn_z > .5) & (aff > .5)),
            "credit_card": -4.9 + .8 * txn_z + .5 * casa_z + .4 * age_z + .4 * llc - .4 * tl + .5 * (txn_z > 1),
            "fd": -5.4 + .5 * casa_z + .4 * size_z, "lc": -5.9 + 1.0 * aff + .4 * size_z,
            "lg": -5.9 + .8 * aff + .4 * size_z + .5 * tl}
        active = onb <= t
        for p in PRODS:
            prob = _sig(logit[p]) * (1 - .8 * delinq[:, t - 1])
            acq = (~h[p]) & active & (rng.random(N) < prob)
            if p == "casa":
                acq |= (onb == t) & (rng.random(N) < .7)
            hold[p][:, t] |= acq
            opn[p][acq] = t
            bal[p][acq, t] = size[acq] * SCALE[p] * rng.lognormal(0, .4, acq.sum())
        _activity(t)

    # ---- tables ------------------------------------------------------------------------
    inport = np.any([hold[p].any(axis=1) for p in PRODS], axis=0)
    dim_customer = pd.DataFrame({
        "customer_number": cid, "onboarding_date": onb_date.astype("datetime64[D]"),
        "incorporation_date": onb_date - (inc_years * 365).astype("timedelta64[D]"), "industry_desc": industry,
        "sector_code": "", "cust_llc_flag": np.where(llc, "Y", "N"),
        "cust_sub_segment": seg, "rm_id": [f"RM{r:02d}" for r in rng.integers(1, 41, N)]})
    dim_customer["sector_code"] = "S" + (dim_customer["industry_desc"].map({k: i for i, k in enumerate(INDUSTRIES)}) + 1).astype(str)
    dim_customer = dim_customer[inport].reset_index(drop=True)

    port, fact, acc, avgb, loan = [], [], [], [], []
    for p in PRODS:
        ii, tt = np.nonzero(hold[p])
        pt = ptype[p][ii]
        acct = np.array([PREF[p] + c for c in cid])[ii]
        gl = np.array([GL[x] for x in pt])
        ccy = np.where(((ii * 7 + 3) % 100) < 85, 1, 2)
        port.append(pd.DataFrame({"account_no": acct, "customer_number": cid[ii], "timekey": timekeys[tt], "product_type": pt,
                                  "gl_no": np.where(gl == "0251", "251", gl), "exclusion_1": excl[ii]}))
        fact.append(pd.DataFrame({"time_key": timekeys[tt], "contract_id": acct, "acc_deal_number": acct, "gl_no": gl,
                                  "currency_key": ccy, "deal_balance": bal[p][ii, tt].round(2),
                                  "category_code": [CAT_CODE.get(x) for x in pt]}))
        ever = np.nonzero(hold[p].any(axis=1))[0]
        od = _mstart(s_ym, opn[p][ever]) + rng.integers(0, 27, len(ever)).astype("timedelta64[D]")
        acc.append(pd.DataFrame({"account_number": PREF[p] + cid[ever], "customer_number": cid[ever], "account_open_date": od,
                                 "closure_date": pd.NaT, "account_status": "ACTIVE", "latest_flag": "Y"}))
        if p == "casa":
            b = bal[p][ii, tt]
            neg = rng.random(len(b)) < .03
            avgb.append(pd.DataFrame({"account_number": acct, "period_date": me[tt], "monthly_average_balance": (b * rng.uniform(.9, 1.1, len(b))).round(2),
                                      "closing_balance": np.where(neg, -b * .1, b).round(2)}))
        if p in ("term_loan", "funded_trade"):
            dq = delinq[ii, tt]
            loan.append(pd.DataFrame({"contract_id": acct, "timekey": timekeys[tt], "dpd": np.where(dq, rng.integers(91, 200, len(ii)), rng.integers(0, 12, len(ii))),
                                      "outstanding": (bal[p][ii, tt] * .95).round(2), "tenor_months": rng.choice([12, 24, 36, 60], len(ii))}))
    ii, tt = np.nonzero(hold["casa"])
    casa_txn = pd.concat([
        pd.DataFrame({"customer_number": cid[ii], "timekey": timekeys[tt], "txn_type": k, "txn_cnt": n[ii, tt].astype(int), "txn_amt": a[ii, tt].round(2)})
        for k, n, a in (("C", crn, cr), ("D", drn, dr))])
    ii, tt = np.nonzero(hold["credit_card"])
    cc_txn = pd.DataFrame({"customer_number": cid[ii], "timekey": timekeys[tt], "logo_type": "C", "txn_cnt": ccn[ii, tt].astype(int),
                           "txn_amt": cca[ii, tt].round(2), "cash_amt": ccc[ii, tt].round(2)})
    tables = {
        "dim_customer": dim_customer, "sme_portfolio": pd.concat(port), "fact_balance_sheet": pd.concat(fact),
        "dim_accounts": pd.concat(acc), "casa_avg_balance": pd.concat(avgb), "loan_snapshot": pd.concat(loan),
        "casa_txn_monthly": casa_txn, "cc_txn_monthly": cc_txn,
        "dim_currency": pd.DataFrame({"currency_key": [1, 2], "ccy_id": ["LCY", "USD"]}),
        "eligibility_policy": pd.DataFrame({
            "product": ["casa", "term_loan", "funded_trade", "credit_card"], "min_lor_months": [7, 12, 12, 9],
            "min_cust_age": [1, 2, 2, 2], "allowed_segments": ["SEG_A,SEG_B", "SEG_A,SEG_B", "SEG_A", "SEG_A,SEG_B"],
            "blocked_industries": ["", "Construction", "Hospitality", ""]}),
        "product_revenue": pd.DataFrame({"product": ["casa", "term_loan", "funded_trade", "credit_card"],
                                         "margin_rate": [.02, .035, .025, .06], "avg_ticket": [80000, 400000, 300000, 30000]}),
    }
    for f in P["raw"].glob("*.csv"):
        f.unlink()
    for k, v in tables.items():
        v.to_csv(P["raw"] / f"{k}.csv", index=False)
    # small Excel sample (first 150 customers, last 6 months) for the batch-ingestion demo
    keep = set(cid[:150]); last6 = set(timekeys[-6:])
    with pd.ExcelWriter(P["raw"] / "sample_input.xlsx") as xw:
        for k, v in tables.items():
            if "customer_number" in v:
                v = v[v.customer_number.isin(keep)]
            tk = "timekey" if "timekey" in v else "time_key" if "time_key" in v else None
            if tk:
                v = v[v[tk].isin(last6)]
            v.head(2000).to_excel(xw, sheet_name=k, index=False)
    log.info("generated %d customers, %d portfolio rows", N, len(tables["sme_portfolio"]))
    return {k: len(v) for k, v in tables.items()}
