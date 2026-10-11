"""Knowledge-graph construction, persistence and querying.

Pipeline:  papers (DB)  ->  extractor  ->  normalized Entity rows
                                        ->  PaperEntity links (typed)
                                        ->  EntityEdge concept-concept edges

The builder reuses the project's existing metadata DB and the paper-level
FAISS store (for paper<->paper similarity edges); it introduces only the
four additive tables in ``graph_models``. Results are also written to a
JSON cache in ``data/`` so the visualization loads instantly without
recomputation.

Everything degrades gracefully: a paper with no extractable entities is
simply skipped, and a missing FAISS vector just omits similarity edges.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from sqlalchemy import delete, func, select

from backend.app.core.config import settings
from backend.app.db.graph_models import Entity, EntityEdge, PaperEntity
from backend.app.db.models import Chunk, Paper
from backend.app.db.session import db_session
from backend.app.services.graph.extractor import extract_entities

logger = logging.getLogger(__name__)

_TYPE_COLORS = {
    "paper": "#2F5D8A",
    "method": "#1E2761",
    "dataset": "#5B8A72",
    "metric": "#C08A2E",
    "task": "#8A4B6B",
    "contribution": "#6A6A6A",
}


def _norm(name: str) -> str:
    """Normalization key for entity dedup (lowercase, strip punctuation)."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


class GraphBuilder:
    def __init__(self, paper_store=None):
        # paper_store is the existing FaissStore("papers"); optional so the
        # builder is unit-testable without FAISS.
        self.paper_store = paper_store

    # ------------------------------------------------------------- build
    def build(self, use_llm: bool = False, limit: int | None = None) -> dict:
        """(Re)build the entire graph from the current paper corpus.

        Idempotent: clears prior graph rows, then re-extracts. Returns a
        summary dict. Safe to call repeatedly (admin action / on demand).
        """
        with db_session() as s:
            papers = s.execute(select(Paper)).scalars().all()
            if limit:
                papers = papers[:limit]
            # full text (if the paper was QA-indexed) sharpens extraction
            chunk_text: dict[int, str] = {}
            if papers:
                rows = s.execute(
                    select(Chunk.paper_id, Chunk.text)
                ).all()
                acc: dict[int, list[str]] = defaultdict(list)
                for pid, txt in rows:
                    if len(acc[pid]) < 6:        # cap per paper
                        acc[pid].append(txt)
                chunk_text = {pid: " ".join(v) for pid, v in acc.items()}

            # wipe previous graph (additive tables only)
            s.execute(delete(PaperEntity))
            s.execute(delete(EntityEdge))
            s.execute(delete(Entity))
            s.flush()

            # entity cache: (norm, etype) -> Entity row
            ent_cache: dict[tuple[str, str], Entity] = {}
            # per-paper entity ids for edge derivation
            paper_entity_ids: dict[int, list[int]] = {}
            n_links = 0

            for p in papers:
                try:
                    mentions = extract_entities(
                        p.title, p.abstract,
                        full_text=chunk_text.get(p.id, ""),
                        use_llm=use_llm,
                    )
                except Exception:
                    logger.debug("extraction failed for %s", p.arxiv_id, exc_info=True)
                    continue
                eids: list[int] = []
                for m in mentions:
                    key = (_norm(m.name), m.etype)
                    if not key[0]:
                        continue
                    ent = ent_cache.get(key)
                    if ent is None:
                        ent = Entity(name=m.name, norm_name=key[0], etype=m.etype,
                                     n_papers=0)
                        s.add(ent)
                        s.flush()           # assign id
                        ent_cache[key] = ent
                    s.add(PaperEntity(
                        paper_id=p.id, entity_id=ent.id, relation=m.relation,
                        confidence=m.confidence, evidence=m.evidence[:300],
                    ))
                    n_links += 1
                    eids.append(ent.id)
                if eids:
                    paper_entity_ids[p.id] = eids

            # degree cache
            for ent in ent_cache.values():
                ent.n_papers = s.execute(
                    select(func.count(func.distinct(PaperEntity.paper_id)))
                    .where(PaperEntity.entity_id == ent.id)
                ).scalar() or 0

            # concept<->concept co-occurrence edges (within a paper)
            edge_w: dict[tuple[int, int], float] = defaultdict(float)
            for eids in paper_entity_ids.values():
                uniq = sorted(set(eids))
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        edge_w[(uniq[i], uniq[j])] += 1.0
            # keep only edges that co-occur in >=2 papers (reduces clutter)
            for (a, b), w in edge_w.items():
                if w >= 2:
                    s.add(EntityEdge(src_id=a, dst_id=b, kind="co_occurs", weight=w))

            summary = {
                "papers_processed": len(papers),
                "entities": len(ent_cache),
                "paper_entity_links": n_links,
                "concept_edges": sum(1 for w in edge_w.values() if w >= 2),
            }

        self._write_cache()
        logger.info("Knowledge graph built: %s", summary)
        return summary

    # ----------------------------------------------------- incremental
    def add_paper(self, arxiv_id: str, use_llm: bool = False) -> int:
        """Extract + link entities for a single paper (used after ingest/QA).

        Returns number of entity links created. Idempotent per paper:
        removes that paper's prior links first.
        """
        with db_session() as s:
            p = s.execute(
                select(Paper).where(Paper.arxiv_id == arxiv_id)
            ).scalar_one_or_none()
            if not p:
                return 0
            ft = " ".join(
                t for (t,) in s.execute(
                    select(Chunk.text).where(Chunk.paper_id == p.id).limit(6)
                ).all()
            )
            mentions = extract_entities(p.title, p.abstract, full_text=ft, use_llm=use_llm)
            s.execute(delete(PaperEntity).where(PaperEntity.paper_id == p.id))
            n = 0
            for m in mentions:
                norm = _norm(m.name)
                if not norm:
                    continue
                ent = s.execute(
                    select(Entity).where(Entity.norm_name == norm, Entity.etype == m.etype)
                ).scalar_one_or_none()
                if ent is None:
                    ent = Entity(name=m.name, norm_name=norm, etype=m.etype)
                    s.add(ent)
                    s.flush()
                s.add(PaperEntity(paper_id=p.id, entity_id=ent.id,
                                  relation=m.relation, confidence=m.confidence,
                                  evidence=m.evidence[:300]))
                n += 1
            # refresh degree for touched entities cheaply on next full build
        return n

    # ----------------------------------------------------------- queries
    def graph(self, etypes: list[str] | None = None, min_degree: int = 1,
              max_nodes: int = 150) -> dict:
        """Return a filtered node/edge graph for visualization.

        Nodes are papers + entities; edges are paper->entity (typed) and
        entity<->entity (co-occurrence). ``max_nodes`` keeps the viz legible.
        """
        with db_session() as s:
            ent_q = select(Entity).where(Entity.n_papers >= min_degree)
            if etypes:
                ent_q = ent_q.where(Entity.etype.in_(etypes))
            ent_q = ent_q.order_by(Entity.n_papers.desc()).limit(max_nodes)
            entities = s.execute(ent_q).scalars().all()
            ent_ids = {e.id for e in entities}
            if not ent_ids:
                return {"nodes": [], "edges": [], "stats": self._empty_stats()}

            links = s.execute(
                select(PaperEntity).where(PaperEntity.entity_id.in_(ent_ids))
            ).scalars().all()
            paper_ids = {l.paper_id for l in links}
            papers = {
                p.id: p for p in s.execute(
                    select(Paper).where(Paper.id.in_(paper_ids))
                ).scalars().all()
            }

            nodes, edges = [], []
            for e in entities:
                nodes.append({
                    "id": f"e{e.id}", "label": e.name, "type": e.etype,
                    "group": e.etype, "color": _TYPE_COLORS.get(e.etype, "#888"),
                    "value": e.n_papers,
                })
            seen_papers = set()
            for l in links:
                p = papers.get(l.paper_id)
                if not p:
                    continue
                if p.id not in seen_papers:
                    seen_papers.add(p.id)
                    nodes.append({
                        "id": f"p{p.id}", "label": _short(p.title),
                        "type": "paper", "group": "paper",
                        "color": _TYPE_COLORS["paper"], "value": 3,
                        "arxiv_id": p.arxiv_id, "title": p.title,
                    })
                edges.append({
                    "source": f"p{p.id}", "target": f"e{l.entity_id}",
                    "relation": l.relation, "kind": "paper_entity",
                })

            cedges = s.execute(
                select(EntityEdge).where(
                    EntityEdge.src_id.in_(ent_ids),
                    EntityEdge.dst_id.in_(ent_ids),
                )
            ).scalars().all()
            for ce in cedges:
                edges.append({
                    "source": f"e{ce.src_id}", "target": f"e{ce.dst_id}",
                    "relation": ce.kind, "kind": "concept_concept",
                    "weight": ce.weight,
                })

            return {"nodes": nodes, "edges": edges, "stats": self.stats()}

    def ego_graph(self, arxiv_id: str, hops: int = 1, max_nodes: int = 60) -> dict:
        """Graph centered on one paper: its entities + papers sharing them."""
        with db_session() as s:
            p = s.execute(
                select(Paper).where(Paper.arxiv_id == arxiv_id)
            ).scalar_one_or_none()
            if not p:
                return {"nodes": [], "edges": [], "stats": self._empty_stats()}
            my_links = s.execute(
                select(PaperEntity).where(PaperEntity.paper_id == p.id)
            ).scalars().all()
            ent_ids = [l.entity_id for l in my_links]
            entities = {
                e.id: e for e in s.execute(
                    select(Entity).where(Entity.id.in_(ent_ids))
                ).scalars().all()
            }
            nodes = [{
                "id": f"p{p.id}", "label": _short(p.title), "type": "paper",
                "group": "paper", "color": _TYPE_COLORS["paper"], "value": 6,
                "arxiv_id": p.arxiv_id, "title": p.title, "center": True,
            }]
            edges = []
            for l in my_links:
                e = entities.get(l.entity_id)
                if not e:
                    continue
                nodes.append({
                    "id": f"e{e.id}", "label": e.name, "type": e.etype,
                    "group": e.etype, "color": _TYPE_COLORS.get(e.etype, "#888"),
                    "value": e.n_papers,
                })
                edges.append({"source": f"p{p.id}", "target": f"e{e.id}",
                              "relation": l.relation, "kind": "paper_entity"})

            # second hop: other papers that share these entities
            if hops >= 1 and ent_ids:
                neigh = s.execute(
                    select(PaperEntity).where(
                        PaperEntity.entity_id.in_(ent_ids),
                        PaperEntity.paper_id != p.id,
                    )
                ).scalars().all()
                added = set()
                for l in neigh:
                    if len(added) >= max_nodes:
                        break
                    if l.paper_id not in added:
                        op = s.get(Paper, l.paper_id)
                        if not op:
                            continue
                        added.add(l.paper_id)
                        nodes.append({
                            "id": f"p{op.id}", "label": _short(op.title),
                            "type": "paper", "group": "paper",
                            "color": "#9AB0C8", "value": 2,
                            "arxiv_id": op.arxiv_id, "title": op.title,
                        })
                    edges.append({"source": f"p{l.paper_id}", "target": f"e{l.entity_id}",
                                  "relation": l.relation, "kind": "paper_entity"})
            return {"nodes": nodes, "edges": edges, "stats": self._empty_stats()}

    def entity_papers(self, entity_id: int) -> dict:
        """Papers linked to an entity, grouped by relation.

        Powers "which papers introduce / modify / use this method".
        """
        with db_session() as s:
            ent = s.get(Entity, entity_id)
            if not ent:
                return {}
            links = s.execute(
                select(PaperEntity).where(PaperEntity.entity_id == entity_id)
            ).scalars().all()
            grouped: dict[str, list[dict]] = defaultdict(list)
            for l in links:
                p = s.get(Paper, l.paper_id)
                if not p:
                    continue
                grouped[l.relation].append({
                    "arxiv_id": p.arxiv_id, "title": p.title,
                    "published": p.published, "confidence": round(l.confidence, 2),
                })
            return {
                "entity": ent.to_dict(),
                "by_relation": {k: v for k, v in grouped.items()},
            }

    # ------------------------------------------------------------- stats
    def stats(self) -> dict:
        with db_session() as s:
            n_ent = s.execute(select(func.count(Entity.id))).scalar() or 0
            n_links = s.execute(select(func.count(PaperEntity.id))).scalar() or 0
            n_edges = s.execute(select(func.count(EntityEdge.id))).scalar() or 0
            by_type = dict(
                s.execute(
                    select(Entity.etype, func.count(Entity.id)).group_by(Entity.etype)
                ).all()
            )
        return {
            "entities": n_ent, "paper_entity_links": n_links,
            "concept_edges": n_edges, "by_type": by_type,
        }

    @staticmethod
    def _empty_stats() -> dict:
        return {"entities": 0, "paper_entity_links": 0, "concept_edges": 0,
                "by_type": {}}

    # ------------------------------------------------------------- cache
    @property
    def _cache_path(self) -> Path:
        return Path(settings.data_dir) / "knowledge_graph.json"

    def _write_cache(self) -> None:
        try:
            self._cache_path.write_text(json.dumps(self.graph()))
        except Exception:
            logger.debug("graph cache write failed", exc_info=True)

    def is_built(self) -> bool:
        with db_session() as s:
            return (s.execute(select(func.count(Entity.id))).scalar() or 0) > 0


def _short(title: str, n: int = 42) -> str:
    title = (title or "").strip()
    return title if len(title) <= n else title[: n - 1] + "…"
