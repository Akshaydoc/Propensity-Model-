"""Command line entry point:  python -m csu.cli <command> [options]"""
import argparse
import json
import logging
import sys
from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="csu", description="Cross-sell / upsell propensity pipeline")
    ap.add_argument("command", choices=["generate", "ingest", "features", "train", "score", "monitor", "stream-produce", "stream-consume",
                                        "stream-compact", "run-all"])
    ap.add_argument("--config"); ap.add_argument("--customers", type=int); ap.add_argument("--source")
    ap.add_argument("--product"); ap.add_argument("--mode", choices=["cross_sell", "upsell"])
    ap.add_argument("--no-mlflow", action="store_true"); ap.add_argument("--drift-demo", action="store_true")
    ap.add_argument("--drift", action="store_true", help="stream-produce: simulate drifted/bad events")
    ap.add_argument("--auto-retrain", action="store_true"); ap.add_argument("--events", type=int, default=500)
    ap.add_argument("--ym", type=int)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config)
    if a.mode:
        cfg["target"]["mode"] = a.mode
    if a.product:
        cfg["products"] = {a.product: cfg["products"][a.product]}
    c = a.command
    if c in ("generate", "run-all"):
        from .datagen import generate
        generate(cfg, a.customers)
    if c in ("ingest", "run-all"):
        from .ingest import ingest_batch
        ingest_batch(cfg, a.source)
    if c in ("features", "run-all"):
        from .features import build_features
        build_features(cfg)
    if c in ("train", "run-all"):
        from .train import train_product
        from .target import load_360
        f, meta = load_360(cfg)
        for p in cfg["products"]:
            r = train_product(cfg, p, f, meta, use_mlflow=not a.no_mlflow)
            print(f"[{p}] champion={r['champion']} oot_gini={r['baseline']['oot_gini']:.3f} features={len(r['features'])}")
    if c in ("score", "run-all"):
        from .score import score_all
        score_all(cfg)
    if c in ("monitor", "run-all"):
        from .monitor import monitor_all
        monitor_all(cfg, drift_demo=a.drift_demo, auto_retrain=a.auto_retrain)
    if c == "stream-produce":
        from .stream import produce
        print(produce(cfg, a.events, a.drift))
    if c == "stream-consume":
        from .stream import consume
        print(json.dumps(consume(cfg), indent=1))
    if c == "stream-compact":
        from .stream import compact_month
        print(compact_month(cfg, a.ym))
    return 0


if __name__ == "__main__":
    sys.exit(main())
