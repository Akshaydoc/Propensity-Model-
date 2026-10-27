import os
import sys
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    """Runs the whole pipeline once on a small synthetic book (1,500 customers, 2 products)."""
    d = tmp_path_factory.mktemp("csu")
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    cfg["paths"] = {k: str(d / k) for k in ("data", "models", "reports")}
    cfg["data"]["n_customers"] = 1500
    cfg["catalog"] = dict(cfg["products"])
    cfg["products"] = {k: v for k, v in cfg["products"].items() if k in ("credit_card", "term_loan")}
    f = d / "config.yaml"
    f.write_text(yaml.safe_dump(cfg))
    os.environ["CSU_CONFIG"] = str(f)
    from csu.cli import main
    main(["run-all", "--config", str(f), "--no-mlflow"])
    from csu.config import load_config
    return load_config(str(f))
