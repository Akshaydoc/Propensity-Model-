"""Batch ingestion: csv / xlsx / parquet drops -> schema validation -> bronze parquet (idempotent upsert)."""
import logging
from datetime import datetime
import pandas as pd
from .config import paths
from .schemas import SCHEMAS, RANGES

log = logging.getLogger(__name__)


def validate(table: str, df: pd.DataFrame):
    """Returns (clean_df, bad_df, issues). Missing columns are fatal; bad rows are quarantined, not dropped silently."""
    spec = SCHEMAS[table]
    issues = []
    missing = [c for c in spec["cols"] if c not in df.columns]
    if missing:
        return df.iloc[0:0], df, [f"missing columns: {missing}"]
    df = df[list(spec["cols"])].copy()
    bad = pd.Series(False, index=df.index)
    for c, kind in spec["cols"].items():
        if kind == "num":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif kind == "date":
            df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            df[c] = df[c].astype("object").where(df[c].notna(), None)
            df[c] = df[c].map(lambda v: None if v is None else str(v))
    for c in spec["not_null"]:
        m = df[c].isna()
        if m.any():
            issues.append(f"{c}: {int(m.sum())} nulls")
        bad |= m
    for (t, c), (lo, hi) in RANGES.items():
        if t == table and c in df:
            m = (df[c] < lo) | (df[c] > hi)
            if m.any():
                issues.append(f"{c}: {int(m.sum())} values out of range")
            bad |= m.fillna(False)
    dup = df.duplicated(spec["key"], keep="last")
    if dup.any():
        issues.append(f"{int(dup.sum())} duplicate keys (kept last)")
    bad_df = df[bad]
    return df[~bad & ~dup], bad_df, issues


def _read_sources(source):
    from pathlib import Path
    source = Path(source)
    files = [source] if source.is_file() else sorted(source.glob("*"))
    for f in files:
        if f.suffix == ".csv" and f.stem in SCHEMAS:
            yield f.stem, pd.read_csv(f, dtype=str, keep_default_na=True), f.name
        elif f.suffix == ".parquet" and f.stem in SCHEMAS:
            yield f.stem, pd.read_parquet(f), f.name
        elif f.suffix == ".xlsx":
            for sheet, d in pd.read_excel(f, sheet_name=None, dtype=str).items():
                if sheet in SCHEMAS:
                    yield sheet, d, f"{f.name}:{sheet}"


def ingest_batch(cfg, source=None):
    P = paths(cfg)
    source = source or P["raw"]
    log_rows = []
    for table, df, origin in _read_sources(source):
        clean, bad, issues = validate(table, df)
        if len(bad):
            bad.to_csv(P["quarantine"] / f"{table}_{datetime.now():%Y%m%d%H%M%S}.csv", index=False)
        out = P["bronze"] / f"{table}.parquet"
        if out.exists() and len(clean):
            old = pd.read_parquet(out)
            clean = pd.concat([old, clean]).drop_duplicates(SCHEMAS[table]["key"], keep="last")
        if len(clean):
            clean.to_parquet(out, index=False)
        log_rows.append(dict(ts=datetime.now().isoformat(timespec="seconds"), table=table, origin=origin, rows_in=len(df),
                             rows_bronze=len(clean), quarantined=len(bad), issues="; ".join(issues)))
        log.info("ingested %-20s in=%-7d bronze=%-7d bad=%d %s", table, len(df), len(clean), len(bad), issues)
    lg = pd.DataFrame(log_rows)
    f = P["reports"] / "ingestion_log.csv"
    lg.to_csv(f, mode="a", header=not f.exists(), index=False)
    return lg
