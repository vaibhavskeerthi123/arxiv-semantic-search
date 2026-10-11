"""REST API routes (functional requirement: 'provide APIs for retrieval and
query operations')."""
from __future__ import annotations

import dataclasses
import logging

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from backend.app.api import schemas
from backend.app.db.models import Paper, QueryLog
from backend.app.db.session import db_session
from backend.app.services.pipeline import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------ search
@router.post("/search", response_model=schemas.SearchResponse)
def search(req: schemas.SearchRequest):
    pipe = get_pipeline()
    return pipe.search(
        req.query,
        top_k=req.top_k,
        fetch_arxiv=req.fetch_arxiv,
        session_id=req.session_id,
    )


# ------------------------------------------------------------------ papers
@router.get("/papers/{arxiv_id}", response_model=schemas.PaperOut)
def get_paper(arxiv_id: str):
    with db_session() as s:
        p = s.execute(select(Paper).where(Paper.arxiv_id == arxiv_id)).scalar_one_or_none()
        if not p:
            raise HTTPException(404, f"paper {arxiv_id} not indexed")
        return p.to_dict()


@router.post("/papers/{arxiv_id}/index")
def index_paper(arxiv_id: str):
    """On-demand full-text indexing: download -> parse -> chunk -> embed.

    Called when the user selects a paper for QA (idempotent — an already
    embedded paper returns immediately). Search never triggers this.
    """
    pipe = get_pipeline()
    try:
        n = pipe.ensure_paper_indexed(arxiv_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    with db_session() as s:
        p = s.execute(select(Paper).where(Paper.arxiv_id == arxiv_id)).scalar_one_or_none()
        status = p.pdf_status if p else "unknown"
    return {"arxiv_id": arxiv_id, "n_chunks": n, "pdf_status": status}


@router.get("/papers/{arxiv_id}/recommendations")
def recommendations(arxiv_id: str, top_k: int = Query(default=5, ge=1, le=20)):
    pipe = get_pipeline()
    recs = pipe.recommender.similar_papers(arxiv_id, top_k=top_k)
    return {"arxiv_id": arxiv_id, "recommendations": recs}


# ---------------------------------------------------------------------- QA
@router.post("/qa/single", response_model=schemas.QAResponse)
def qa_single(req: schemas.SingleQARequest):
    pipe = get_pipeline()
    try:
        res = pipe.ask_single(
            req.question,
            req.arxiv_id,
            history=[t.model_dump() for t in req.history],
            session_id=req.session_id,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return _qa_response(res)


@router.post("/qa/multi", response_model=schemas.QAResponse)
def qa_multi(req: schemas.MultiQARequest):
    pipe = get_pipeline()
    res = pipe.ask_multi(
        req.question,
        paper_ids=req.paper_ids,
        history=[t.model_dump() for t in req.history],
        session_id=req.session_id,
    )
    return _qa_response(res)


def _qa_response(res) -> dict:
    return {
        "answer": res.answer,
        "mode": res.mode,
        "papers": res.papers,
        "sources": [dataclasses.asdict(s) for s in res.sources],
    }


# ------------------------------------------------------------------- admin
@router.get("/admin/stats")
def stats():
    return get_pipeline().stats()


@router.post("/admin/rebuild")
def rebuild():
    return get_pipeline().rebuild_indexes()


@router.get("/admin/history")
def history(limit: int = Query(default=50, ge=1, le=500), session_id: str = ""):
    with db_session() as s:
        q = select(QueryLog).order_by(QueryLog.id.desc()).limit(limit)
        if session_id:
            q = select(QueryLog).where(QueryLog.session_id == session_id).order_by(
                QueryLog.id.desc()
            ).limit(limit)
        rows = s.execute(q).scalars().all()
        return [
            {
                "id": r.id,
                "mode": r.mode,
                "query": r.query,
                "latency_ms": r.latency_ms,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


# ================================================================= GRAPH
@router.post("/graph/build")
def graph_build(req: schemas.GraphBuildRequest):
    """(Re)build the knowledge graph from the current paper corpus."""
    from backend.app.core.config import settings

    pipe = get_pipeline()
    use_llm = settings.graph_use_llm if req.use_llm is None else req.use_llm
    return pipe.graph.build(use_llm=use_llm, limit=req.limit)


@router.get("/graph")
def graph_view(
    etypes: str = Query(default="", description="comma-separated entity types"),
    min_degree: int = Query(default=1, ge=1, le=100),
    max_nodes: int = Query(default=150, ge=10, le=1000),
):
    """Filtered knowledge graph for visualization. Auto-builds if empty."""
    pipe = get_pipeline()
    if not pipe.graph.is_built():
        pipe.graph.build()
    types = [t.strip() for t in etypes.split(",") if t.strip()] or None
    return pipe.graph.graph(etypes=types, min_degree=min_degree, max_nodes=max_nodes)


@router.get("/graph/stats")
def graph_stats():
    return get_pipeline().graph.stats()


@router.get("/graph/paper/{arxiv_id}")
def graph_ego(arxiv_id: str, hops: int = Query(default=1, ge=1, le=2)):
    """Ego-graph centered on one paper (its concepts + papers sharing them)."""
    pipe = get_pipeline()
    if not pipe.graph.is_built():
        pipe.graph.build()
    return pipe.graph.ego_graph(arxiv_id, hops=hops)


@router.get("/graph/entity/{entity_id}")
def graph_entity(entity_id: int):
    """Which papers introduce / use / evaluate-on this concept."""
    res = get_pipeline().graph.entity_papers(entity_id)
    if not res:
        raise HTTPException(404, f"entity {entity_id} not found")
    return res


# ============================================================= READ NEXT
@router.post("/reading-list")
def reading_list_add(req: schemas.ReadingListRequest):
    pipe = get_pipeline()
    return pipe.next_read.add_to_list(req.session_id, req.arxiv_id, req.action)


@router.delete("/reading-list")
def reading_list_remove(session_id: str, arxiv_id: str):
    pipe = get_pipeline()
    return pipe.next_read.remove_from_list(session_id, arxiv_id)


@router.get("/reading-list")
def reading_list_get(session_id: str):
    return get_pipeline().next_read.get_list(session_id)


@router.get("/next-reads")
def next_reads(session_id: str, top_k: int = Query(default=6, ge=1, le=20)):
    """Gap-aware 'what should I read next' recommendations with explanations."""
    return get_pipeline().next_read.recommend(session_id, top_k=top_k)


@router.get("/health")
def health():
    return {"status": "ok"}
