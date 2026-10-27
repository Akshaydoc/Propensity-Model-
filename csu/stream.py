"""Lightweight continuous ingestion: a producer drops JSONL micro-batches into a landing folder, the consumer
validates + appends them to bronze, keeps month-to-date live signals and watches the stream for drift (PSI).
It is a file-queue on purpose (free, Windows-friendly, no broker); swapping the folder for Kafka/Redpanda only
changes `_read_batches`."""
import json
import logging
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
from .config import paths

log = logging.getLogger(__name__)
REQUIRED = ["event_id", "ts", "customer_number", "channel", "txn_type", "amount"]


def produce(cfg, n_events=500, drift=False, seed=None):
    """Simulates the upstream transaction feed. drift=True inflates ticket sizes and injects some bad events."""
    P = paths(cfg)
    rng = np.random.default_rng(seed)
    cust = pd.read_parquet(P["bronze"] / "dim_customer.parquet", columns=["customer_number"]).customer_number.to_numpy()
    cust = rng.choice(cust, n_events)
    amt = rng.lognormal(7.0, .8, n_events) * (2.6 if drift else 1.0)
    ev = pd.DataFrame({"event_id": [f"E{datetime.now():%H%M%S%f}{i}" for i in range(n_events)],
                       "ts": datetime.now().isoformat(timespec="seconds"), "customer_number": cust,
                       "channel": rng.choice(["CASA", "CARD"], n_events, p=[.65, .35]),
                       "txn_type": rng.choice(["C", "D"], n_events), "amount": amt.round(2)})
    if drift:  # a few malformed events to exercise validation
        ev.loc[ev.sample(max(1, n_events // 50), random_state=1).index, "amount"] = -1
        ev.loc[ev.sample(max(1, n_events // 50), random_state=2).index, "customer_number"] = None
    f = P["landing"] / f"batch_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl"
    ev.to_json(f, orient="records", lines=True)
    return f


def _read_batches(P):
    yield from sorted(P["landing"].glob("*.jsonl"))


def _psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    a = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((a - e) * np.log(a / e)))


def consume(cfg):
    P = paths(cfg)
    mon = cfg["monitoring"]
    store = P["bronze"] / "txn_events.parquet"
    status = dict(ts=datetime.now().isoformat(timespec="seconds"), batches=0, events=0, rejected=0, stream_psi=None, alert=False)
    new = []
    for f in _read_batches(P):
        df = pd.read_json(f, lines=True, dtype={"customer_number": str})
        ok = df.reindex(columns=REQUIRED).notna().all(axis=1) & (df.get("amount", pd.Series(dtype=float)) > 0)
        if (~ok).any():
            df[~ok].to_json(P["quarantine"] / "stream_rejected.jsonl", orient="records", lines=True, mode="a")
        status["rejected"] += int((~ok).sum())
        new.append(df[ok])
        shutil.move(str(f), P["processed"] / f.name)
        status["batches"] += 1
    if not new:
        return status
    new = pd.concat(new)
    status["events"] = len(new)
    old = pd.read_parquet(store) if store.exists() else None
    allev = pd.concat([old, new]).drop_duplicates("event_id") if old is not None else new
    allev.to_parquet(store, index=False)
    # baseline = first healthy micro-batches ever seen; later batches are compared with it
    ref_f = P["models"] / "stream_reference.npy"
    if not ref_f.exists():
        np.save(ref_f, new.amount.to_numpy()[:5000])
        status["note"] = "baseline created"
    else:
        status["stream_psi"] = round(_psi(np.load(ref_f), new.amount.to_numpy()), 4)
        status["alert"] = status["stream_psi"] >= mon["psi_alert"]
    # month-to-date live signals per customer
    allev["ts"] = pd.to_datetime(allev["ts"])
    cur = allev[allev.ts.dt.to_period("M") == allev.ts.max().to_period("M")]
    live = cur.groupby("customer_number").agg(mtd_txn_cnt=("event_id", "count"), mtd_txn_amt=("amount", "sum")).reset_index()
    live.to_parquet(P["gold"] / "live_signals.parquet", index=False)
    (P["monitoring"] / "stream_status.json").write_text(json.dumps(status, indent=2))
    h = P["monitoring"] / "stream_history.csv"
    pd.DataFrame([status]).to_csv(h, mode="a", header=not h.exists(), index=False)
    log.info("stream: %s", status)
    return status


def compact_month(cfg, ym):
    """Month-end: roll raw stream events into the monthly txn tables the feature pipeline reads."""
    P = paths(cfg)
    ev = pd.read_parquet(P["bronze"] / "txn_events.parquet")
    ev["ts"] = pd.to_datetime(ev["ts"])
    ev = ev[ev.ts.dt.year * 100 + ev.ts.dt.month == ym]
    tk = int((pd.Period(f"{ym // 100}-{ym % 100:02d}") .end_time).strftime("%Y%m%d"))
    casa = ev[ev.channel == "CASA"].groupby(["customer_number", "txn_type"]).agg(txn_cnt=("event_id", "count"), txn_amt=("amount", "sum")).reset_index()
    casa["timekey"] = tk
    card = ev[ev.channel == "CARD"].groupby("customer_number").agg(txn_cnt=("event_id", "count"), txn_amt=("amount", "sum")).reset_index()
    card["timekey"], card["logo_type"], card["cash_amt"] = tk, "C", 0.0
    for name, df, key in (("casa_txn_monthly", casa, ["customer_number", "timekey", "txn_type"]),
                          ("cc_txn_monthly", card, ["customer_number", "timekey"])):
        f = P["bronze"] / f"{name}.parquet"
        old = pd.read_parquet(f)
        pd.concat([old, df]).drop_duplicates(key, keep="last").to_parquet(f, index=False)
    return dict(ym=ym, casa_rows=len(casa), card_rows=len(card))
