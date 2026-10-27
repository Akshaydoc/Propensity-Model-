from fastapi.testclient import TestClient


def _client(env):
    import importlib
    import api.main as m
    m.cfg.cache_clear()
    importlib.reload(m)
    return TestClient(m.app)


def test_health_summary_score_and_prospects(env):
    c = _client(env)
    assert c.get("/health").json()["models_ready"]
    assert len(c.get("/api/summary").json()) == 2
    ex = c.get("/api/example/credit_card").json()
    r = c.post("/score", json={"product": "credit_card", "features": ex})
    assert r.status_code == 200 and 0 <= r.json()["propensity"] <= 1
    assert c.post("/score", json={"product": "nope", "features": {}}).status_code == 404
    assert len(c.get("/api/prospects?product=credit_card&limit=5").json()) == 5
    assert c.get("/").status_code == 200
