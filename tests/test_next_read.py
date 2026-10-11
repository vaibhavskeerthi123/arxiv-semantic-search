"""Tests for the gap-aware 'What Should I Read Next?' feature."""
from __future__ import annotations


def _svc(pipeline):
    pipeline.graph.build()          # entities power the gap analysis
    return pipeline.next_read


# ------------------------------------------------------- reading list CRUD
def test_reading_list_crud(pipeline):
    nr = _svc(pipeline)
    sess = "crud_sess"
    nr.add_to_list(sess, "2503.23013", "saved")
    nr.add_to_list(sess, "2402.01767", "read")
    lst = nr.get_list(sess)
    assert {x["arxiv_id"] for x in lst} == {"2503.23013", "2402.01767"}
    # idempotent re-add updates action, no duplicate
    nr.add_to_list(sess, "2503.23013", "read")
    assert len(nr.get_list(sess)) == 2
    nr.remove_from_list(sess, "2503.23013")
    assert {x["arxiv_id"] for x in nr.get_list(sess)} == {"2402.01767"}


# ------------------------------------------------------------ recommend
def test_empty_history_returns_prompt(pipeline):
    nr = _svc(pipeline)
    res = nr.recommend("nobody_sess")
    assert res["mode"] == "empty"
    assert res["recommendations"] == []


def test_single_paper_falls_back_to_similarity(pipeline):
    nr = _svc(pipeline)
    sess = "single_sess"
    nr.add_to_list(sess, "2503.23013", "saved")
    res = nr.recommend(sess, top_k=3)
    assert res["mode"] == "similarity_fallback"
    # recommendations never include an already-read paper
    assert all(r["arxiv_id"] != "2503.23013" for r in res["recommendations"])


def test_gap_analysis_with_multiple_papers(pipeline):
    nr = _svc(pipeline)
    sess = "gap_sess"
    # papers spanning different sub-topics in the test corpus
    for aid in ("2503.23013", "2508.08742", "2411.16019"):
        nr.add_to_list(sess, aid, "read")
    res = nr.recommend(sess, top_k=5)
    assert res["mode"] == "gap_analysis"
    assert res["n_read"] == 3
    read = {"2503.23013", "2508.08742", "2411.16019"}
    for r in res["recommendations"]:
        assert r["arxiv_id"] not in read        # never recommend read papers
        assert r["why"]                         # always an explanation
        assert "score" in r


def test_next_reads_api(pipeline):
    from fastapi.testclient import TestClient

    import backend.app.services.pipeline as pl
    from backend.app.main import app
    pl._pipeline = pipeline
    c = TestClient(app)

    sess = "api_next_sess"
    for aid in ("2503.23013", "2508.08742"):
        r = c.post("/api/v1/reading-list",
                   json={"session_id": sess, "arxiv_id": aid, "action": "read"})
        assert r.status_code == 200
    assert c.get(f"/api/v1/reading-list?session_id={sess}").status_code == 200
    nr = c.get(f"/api/v1/next-reads?session_id={sess}&top_k=4")
    assert nr.status_code == 200
    body = nr.json()
    assert body["mode"] in ("gap_analysis", "similarity_fallback")
    # delete endpoint
    d = c.request("DELETE",
                  f"/api/v1/reading-list?session_id={sess}&arxiv_id=2503.23013")
    assert d.status_code == 200
