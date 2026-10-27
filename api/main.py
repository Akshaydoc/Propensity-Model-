"""FastAPI service: scoring, ranked prospects, model reports, drift status + the dashboard page."""
import json
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from csu.config import ROOT, load_config, paths

app = FastAPI(title="Cross-sell / Upsell Propensity API", version="1.0.0")


@lru_cache
def cfg():
    return load_config()


def P():
    return paths(cfg())


_bundles = {}


def bundle(product):
    mode = cfg()["target"]["mode"]
    f = P()["models"] / f"{product}_{mode}" / "bundle.joblib"
    if product not in cfg()["products"] or not f.exists():
        raise HTTPException(404, f"no trained model for '{product}'")
    key = (product, f.stat().st_mtime)
    if key not in _bundles:
        _bundles.clear()
        _bundles[key] = joblib.load(f)
    return _bundles[key]


def _json(path):
    return json.loads(path.read_text()) if path.exists() else None


def _recs():
    f = P()["gold"] / "recommendations.parquet"
    if not f.exists():
        raise HTTPException(404, "recommendations not generated yet - run the pipeline")
    return _cached_recs(str(f), f.stat().st_mtime)


@lru_cache(maxsize=2)
def _cached_recs(path, _mtime):
    return pd.read_parquet(path)


class ScoreRequest(BaseModel):
    product: str
    features: dict


@app.get("/health")
def health():
    mode = cfg()["target"]["mode"]
    ready = [p for p in cfg()["products"] if (P()["models"] / f"{p}_{mode}" / "bundle.joblib").exists()]
    return {"status": "ok", "mode": mode, "models_ready": ready}


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


@app.get("/api/summary")
def summary():
    mode, out = cfg()["target"]["mode"], []
    for p in cfg()["products"]:
        r = _json(P()["reports"] / f"{p}_{mode}" / "metrics.json")
        if not r:
            continue
        champ = next(m for m in r["models"] if m["model"] == r["champion"])
        mon = _json(P()["monitoring"] / f"{p}_{mode}_latest.json") or {}
        out.append(dict(product=p, label=r["label"], champion=r["champion"], threshold=r["threshold"], trained_at=r["trained_at"],
                        oot_gini=champ["oot_gini"], oot_auc=champ["oot_auc"], oot_ks=champ["oot_ks"], test_gini=champ["test_gini"],
                        top10_lift=champ["oot_top10_lift"], top20_capture=champ["oot_top20_capture"], status=mon.get("status", "n/a"),
                        sizes=r["sizes"]))
    return out


@app.get("/api/models/{product}")
def model_report(product: str):
    r = _json(P()["reports"] / f"{product}_{cfg()['target']['mode']}" / "metrics.json")
    if not r:
        raise HTTPException(404, "report not found")
    return r


@app.get("/api/lift/{product}.png")
def lift_png(product: str):
    f = P()["reports"] / f"{product}_{cfg()['target']['mode']}" / "lift.png"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/rms")
def rms():
    return sorted(_recs().rm_id.unique().tolist())


@app.get("/api/prospects")
def prospects(product: str | None = None, rm_id: str | None = None, limit: int = Query(50, le=500), best_only: bool = False):
    d = _recs()
    if product:
        d = d[d["product"] == product]
    if rm_id:
        d = d[d.rm_id == rm_id]
    if best_only:
        d = d[d.rank_in_customer == 1]
    d = d.sort_values("expected_value", ascending=False).head(limit)
    return d.drop(columns=["ym"]).to_dict("records")


@app.get("/api/customer/{customer_number}")
def customer(customer_number: str):
    d = _recs()
    d = d[d.customer_number == customer_number].sort_values("rank_in_customer")
    if d.empty:
        raise HTTPException(404, "customer not found / not eligible for any offer")
    live = P()["gold"] / "live_signals.parquet"
    sig = None
    if live.exists():
        l = pd.read_parquet(live)
        l = l[l.customer_number == customer_number]
        sig = l.iloc[0].to_dict() if len(l) else None
    return {"customer_number": customer_number, "offers": d.drop(columns=["ym"]).to_dict("records"), "live_signals": sig}


@app.get("/api/drift")
def drift():
    mode = cfg()["target"]["mode"]
    return {"models": [_json(P()["monitoring"] / f"{p}_{mode}_latest.json") for p in cfg()["products"]
                       if (P()["monitoring"] / f"{p}_{mode}_latest.json").exists()],
            "stream": _json(P()["monitoring"] / "stream_status.json")}


@app.get("/api/evidently/{product}", response_class=HTMLResponse)
def evidently(product: str):
    f = P()["monitoring"] / f"{product}_{cfg()['target']['mode']}_evidently.html"
    if not f.exists():
        raise HTTPException(404, "no evidently report yet")
    return f.read_text(encoding="utf-8")


@app.get("/api/example/{product}")
def example(product: str):
    b = bundle(product)
    row = b["ref_sample"].iloc[0]
    return {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}


@app.post("/score")
def score(req: ScoreRequest):
    from csu.score import score_frame
    b = bundle(req.product)
    row, warnings = {}, []
    for v in b["features"]:
        x = req.features.get(v)
        if x is None:
            warnings.append(f"missing '{v}' - scored as NULL bin")
        if b["specs"][v]["type"] == "numerical":
            try:
                row[v] = np.nan if x is None else float(x)
            except (TypeError, ValueError):
                raise HTTPException(422, f"'{v}' must be numeric")
        else:
            row[v] = x
    raw, prob, dec = score_frame(b, pd.DataFrame([row]))
    return {"product": req.product, "model": b["model_name"], "propensity_raw": round(float(raw[0]), 4), "propensity": round(float(prob[0]), 4),
            "decile": int(dec[0]), "contact": bool(raw[0] >= b["threshold"]), "threshold": b["threshold"], "warnings": warnings}
