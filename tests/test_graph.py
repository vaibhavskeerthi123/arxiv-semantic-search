"""Tests for the Paper -> Knowledge Graph feature (extractor + builder + API)."""
from __future__ import annotations

from backend.app.services.graph.extractor import extract_entities


# --------------------------------------------------------------- extractor
def test_extractor_types_and_relations():
    ms = extract_entities(
        "DAT: Dynamic Alpha Tuning",
        "We propose dynamic alpha tuning, a method that fuses BM25 sparse "
        "retrieval and dense retrieval. Evaluated on BEIR and MS MARCO with nDCG.",
    )
    by_type = {m.etype for m in ms}
    assert {"method", "dataset", "metric"} <= by_type
    names = {m.name.lower() for m in ms}
    assert "bm25" in names and "beir" in names
    # the proposed method should be tagged 'introduces'
    dat = [m for m in ms if "alpha tuning" in m.name.lower()]
    assert dat and dat[0].relation == "introduces"


def test_extractor_handles_empty_and_garbage():
    assert extract_entities("", "") == []
    assert extract_entities(None, None) == []          # type: ignore[arg-type]
    # must not raise on weird unicode / no entities
    out = extract_entities("随机标题", "no recognizable entities here at all 123")
    assert isinstance(out, list)


def test_extractor_summary_style_introduction():
    # third-person condensed abstracts (as stored in the seed corpus)
    ms = extract_entities(
        "Attention Is All You Need",
        "Introduces the Transformer, a sequence architecture based on self-attention.",
    )
    intro = [m for m in ms if m.relation == "introduces"]
    assert any("transformer" in m.name.lower() for m in intro)


# ----------------------------------------------------------------- builder
def test_build_graph_from_corpus(pipeline):
    summary = pipeline.graph.build()
    assert summary["papers_processed"] >= 6
    assert summary["entities"] > 0
    assert summary["paper_entity_links"] > 0
    assert pipeline.graph.is_built()


def test_graph_view_structure(pipeline):
    pipeline.graph.build()
    g = pipeline.graph.graph(max_nodes=50)
    assert g["nodes"] and g["edges"]
    # nodes carry a type and color; edges reference node ids
    node_ids = {n["id"] for n in g["nodes"]}
    for e in g["edges"]:
        assert e["source"] in node_ids and e["target"] in node_ids
    # at least one paper node and one concept node
    types = {n["type"] for n in g["nodes"]}
    assert "paper" in types and (types - {"paper"})


def test_graph_type_filter(pipeline):
    pipeline.graph.build()
    g = pipeline.graph.graph(etypes=["dataset"], max_nodes=50)
    concept_types = {n["type"] for n in g["nodes"] if n["type"] != "paper"}
    assert concept_types <= {"dataset"}


def test_entity_papers_introduce_vs_use(pipeline):
    pipeline.graph.build()
    from sqlalchemy import select

    from backend.app.db.graph_models import Entity
    from backend.app.db.session import db_session
    with db_session() as s:
        # BM25 appears in several abstracts -> should have 'uses' links
        e = s.execute(
            select(Entity).where(Entity.norm_name == "bm25")
        ).scalar_one_or_none()
    assert e is not None
    res = pipeline.graph.entity_papers(e.id)
    assert res["entity"]["name"].lower() == "bm25"
    assert res["by_relation"]           # at least one relation group


def test_ego_graph(pipeline):
    pipeline.graph.build()
    ego = pipeline.graph.ego_graph("2503.23013")
    assert ego["nodes"]
    center = [n for n in ego["nodes"] if n.get("center")]
    assert len(center) == 1 and center[0]["arxiv_id"] == "2503.23013"


def test_graph_api_endpoints(pipeline):
    from fastapi.testclient import TestClient

    import backend.app.services.pipeline as pl
    from backend.app.main import app
    pl._pipeline = pipeline
    c = TestClient(app)

    assert c.post("/api/v1/graph/build", json={}).status_code == 200
    g = c.get("/api/v1/graph?max_nodes=40")
    assert g.status_code == 200 and g.json()["nodes"]
    assert c.get("/api/v1/graph/stats").status_code == 200
    assert c.get("/api/v1/graph/paper/2503.23013").status_code == 200
    # unknown entity -> 404
    assert c.get("/api/v1/graph/entity/999999").status_code == 404
