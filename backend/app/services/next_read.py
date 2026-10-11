"""Gap-aware "What Should I Read Next?" recommendations.

This goes **beyond** the existing cosine-similarity ``Recommender``. Instead
of "papers similar to what you already read" (which tends to recommend more
of the same), it looks for **structural holes** in the user's personal
research graph and recommends papers that *bridge* concept clusters the user
has touched but that are not yet connected in their reading.

Algorithm
---------
1.  **Profile.** Gather the user's read/saved papers and their extracted
    entities (reusing Feature-1's knowledge graph). This is the set of
    concepts the user already engages with.
2.  **Cluster.** Group the user's papers by embedding proximity (reusing the
    existing paper-level FAISS vectors) into a few coarse interest clusters.
3.  **Find bridges.** For every candidate paper in the corpus (not already
    read), score how well it *connects* the user's clusters:
      * **bridge score** — does it share entities with >=2 different clusters
        that currently have few/no shared concepts? (fills a structural hole)
      * **novelty** — moderate embedding distance from the user's centroid
        (not a near-duplicate, not unrelated)
      * **relevance** — still within the user's broad area (centroid cosine)
    Final score rewards papers that connect otherwise-disconnected interests.
4.  **Explain.** Produce a plain-language reason naming the concepts the paper
    bridges. Uses ``safe_generate`` when a real LLM is present, otherwise a
    grounded template — so it always returns a human-readable explanation.

Everything reuses existing infrastructure (FAISS paper vectors, the metadata
DB, the knowledge-graph tables) and degrades gracefully: with too little
history it falls back to similarity-based suggestions and says so.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sqlalchemy import select

from backend.app.db.graph_models import Entity, PaperEntity, ReadingListItem
from backend.app.db.models import Paper
from backend.app.db.session import db_session

logger = logging.getLogger(__name__)


class NextReadService:
    def __init__(self, paper_store, embedder, recommender=None):
        self.paper_store = paper_store
        self.embedder = embedder
        self.recommender = recommender

    # ------------------------------------------------------- reading list
    def add_to_list(self, session_id: str, arxiv_id: str, action: str = "saved") -> dict:
        with db_session() as s:
            exists = s.execute(
                select(ReadingListItem).where(
                    ReadingListItem.session_id == session_id,
                    ReadingListItem.arxiv_id == arxiv_id,
                )
            ).scalar_one_or_none()
            if exists:
                exists.action = action
            else:
                s.add(ReadingListItem(
                    session_id=session_id, arxiv_id=arxiv_id, action=action
                ))
        return {"session_id": session_id, "arxiv_id": arxiv_id, "action": action}

    def remove_from_list(self, session_id: str, arxiv_id: str) -> dict:
        with db_session() as s:
            row = s.execute(
                select(ReadingListItem).where(
                    ReadingListItem.session_id == session_id,
                    ReadingListItem.arxiv_id == arxiv_id,
                )
            ).scalar_one_or_none()
            if row:
                s.delete(row)
        return {"removed": arxiv_id}

    def get_list(self, session_id: str) -> list[dict]:
        with db_session() as s:
            rows = s.execute(
                select(ReadingListItem).where(
                    ReadingListItem.session_id == session_id
                ).order_by(ReadingListItem.id.desc())
            ).scalars().all()
            out = []
            for r in rows:
                p = s.execute(
                    select(Paper).where(Paper.arxiv_id == r.arxiv_id)
                ).scalar_one_or_none()
                out.append({
                    "arxiv_id": r.arxiv_id,
                    "action": r.action,
                    "title": p.title if p else r.arxiv_id,
                })
            return out

    # ------------------------------------------------------- helpers
    def _vec_for(self, arxiv_id: str) -> np.ndarray | None:
        for vid, meta in self.paper_store.all_meta().items():
            if meta.get("arxiv_id") == arxiv_id:
                return self.paper_store.get_vector(vid)
        return None

    def _entities_for(self, s, paper_arxiv_ids: list[str]) -> dict[str, set[int]]:
        """arxiv_id -> set of entity ids (from the knowledge graph)."""
        out: dict[str, set[int]] = defaultdict(set)
        if not paper_arxiv_ids:
            return out
        papers = s.execute(
            select(Paper).where(Paper.arxiv_id.in_(paper_arxiv_ids))
        ).scalars().all()
        id_to_arxiv = {p.id: p.arxiv_id for p in papers}
        links = s.execute(
            select(PaperEntity).where(PaperEntity.paper_id.in_(list(id_to_arxiv)))
        ).scalars().all()
        for l in links:
            out[id_to_arxiv[l.paper_id]].add(l.entity_id)
        return out

    def _cluster(self, arxiv_ids: list[str], vecs: dict[str, np.ndarray],
                 k: int = 3) -> dict[str, int]:
        """Tiny k-means over the user's paper vectors -> cluster label per paper.

        Pure-numpy, no sklearn dependency. k is capped by #papers.
        """
        items = [(a, vecs[a]) for a in arxiv_ids if a in vecs]
        if len(items) <= 1:
            return {a: 0 for a, _ in items}
        k = max(2, min(k, len(items)))
        X = np.vstack([v for _, v in items]).astype("float32")
        rng = np.random.default_rng(42)
        centers = X[rng.choice(len(X), k, replace=False)]
        labels = np.zeros(len(X), dtype=int)
        for _ in range(12):
            # cosine == dot (vectors are L2-normalized)
            sims = X @ centers.T
            new = sims.argmax(axis=1)
            if (new == labels).all():
                break
            labels = new
            for c in range(k):
                members = X[labels == c]
                if len(members):
                    nc = members.mean(axis=0)
                    n = np.linalg.norm(nc)
                    centers[c] = nc / n if n > 0 else nc
        return {items[i][0]: int(labels[i]) for i in range(len(items))}

    # ------------------------------------------------------- main
    def recommend(self, session_id: str, top_k: int = 6) -> dict:
        with db_session() as s:
            read = s.execute(
                select(ReadingListItem.arxiv_id).where(
                    ReadingListItem.session_id == session_id
                )
            ).scalars().all()
            read = list(read)

            if not read:
                return {
                    "mode": "empty",
                    "message": "Save or read a few papers first — then this "
                               "will find papers that bridge the gaps between "
                               "your interests.",
                    "recommendations": [],
                }

            vecs = {a: v for a in read if (v := self._vec_for(a)) is not None}
            # Too little signal for gap analysis -> fall back to similarity.
            if len(vecs) < 2:
                return self._similarity_fallback(s, session_id, read, top_k)

            clusters = self._cluster(read, vecs)
            n_clusters = len(set(clusters.values()))
            read_entities = self._entities_for(s, read)
            # entities present in each cluster
            cluster_entities: dict[int, set[int]] = defaultdict(set)
            for a, c in clusters.items():
                cluster_entities[c] |= read_entities.get(a, set())

            # which cluster pairs are currently NOT bridged (share < 1 concept)
            weak_pairs = []
            cids = sorted(cluster_entities)
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    shared = cluster_entities[cids[i]] & cluster_entities[cids[j]]
                    if len(shared) <= 1:
                        weak_pairs.append((cids[i], cids[j]))

            centroid = np.mean(np.vstack(list(vecs.values())), axis=0)
            cn = np.linalg.norm(centroid)
            if cn > 0:
                centroid = centroid / cn

            # score every candidate paper in the corpus
            read_set = set(read)
            all_meta = self.paper_store.all_meta()
            scored = []
            for vid, meta in all_meta.items():
                aid = meta.get("arxiv_id")
                if not aid or aid in read_set:
                    continue
                cand_vec = self.paper_store.get_vector(vid)
                if cand_vec is None:
                    continue
                cand_ents = self._entities_for(s, [aid]).get(aid, set())
                # bridge: does this paper connect clusters of a weak pair?
                bridged = 0
                bridged_pairs = []
                for (ci, cj) in weak_pairs:
                    if (cand_ents & cluster_entities[ci]) and (cand_ents & cluster_entities[cj]):
                        bridged += 1
                        bridged_pairs.append((ci, cj))
                relevance = float(cand_vec @ centroid)        # in [-1,1]
                # novelty: reward moderate distance (sweet spot ~0.45 cosine)
                novelty = 1.0 - abs(relevance - 0.45) / 0.55
                novelty = max(0.0, min(1.0, novelty))
                # entities shared with the user overall (connectedness)
                user_all_ents = set().union(*read_entities.values()) if read_entities else set()
                overlap = len(cand_ents & user_all_ents)
                score = (
                    2.0 * bridged                     # structural-hole bridging
                    + 0.6 * novelty                   # not a near-duplicate
                    + 0.2 * min(overlap, 5) / 5.0     # still connected to you
                    + 0.3 * max(0.0, relevance)       # within your area
                )
                scored.append({
                    "arxiv_id": aid, "score": score, "bridged": bridged,
                    "bridged_pairs": bridged_pairs, "relevance": relevance,
                    "novelty": novelty, "shared": overlap,
                    "entities": cand_ents,
                })

            scored.sort(key=lambda d: d["score"], reverse=True)
            top = scored[:top_k]
            recs = [self._enrich(s, r, cluster_entities) for r in top]
            return {
                "mode": "gap_analysis",
                "n_read": len(read),
                "n_clusters": n_clusters,
                "weak_connections": len(weak_pairs),
                "message": (
                    f"Analyzed {len(read)} papers across {n_clusters} interest "
                    f"clusters and found {len(weak_pairs)} pair(s) of topics you "
                    f"haven't connected yet. These papers bridge them:"
                ),
                "recommendations": recs,
            }

    # ------------------------------------------------------- explanations
    def _enrich(self, s, r: dict, cluster_entities) -> dict:
        p = s.execute(
            select(Paper).where(Paper.arxiv_id == r["arxiv_id"])
        ).scalar_one_or_none()
        # names of the concepts this paper bridges
        bridge_concepts: list[str] = []
        for (ci, cj) in r["bridged_pairs"]:
            inter = r["entities"] & (cluster_entities[ci] | cluster_entities[cj])
            names = self._entity_names(s, list(inter)[:4])
            bridge_concepts.extend(names)
        bridge_concepts = list(dict.fromkeys(bridge_concepts))[:5]
        why = self._explanation(p, r, bridge_concepts)
        return {
            "arxiv_id": r["arxiv_id"],
            "title": p.title if p else r["arxiv_id"],
            "abstract": (p.abstract[:280] + "…") if p and len(p.abstract) > 280 else (p.abstract if p else ""),
            "published": p.published if p else "",
            "score": round(r["score"], 3),
            "bridges": r["bridged"],
            "bridge_concepts": bridge_concepts,
            "why": why,
        }

    def _entity_names(self, s, ids: list[int]) -> list[str]:
        if not ids:
            return []
        rows = s.execute(select(Entity).where(Entity.id.in_(ids))).scalars().all()
        return [e.name for e in rows]

    def _explanation(self, paper, r: dict, concepts: list[str]) -> str:
        if r["bridged"] and concepts:
            base = (
                f"Connects ideas you've read about — "
                f"{', '.join(concepts[:3])} — that you haven't seen together yet."
            )
        elif r["novelty"] > 0.5:
            base = ("A natural next step: related to your interests but covering "
                    "ground your current reading doesn't.")
        else:
            base = "Closely related to papers you've saved."

        # optional LLM polish (grounded; falls back to base on any issue)
        try:
            from backend.app.services.llm.generator import get_llm
            llm = get_llm()
            if llm.name != "extractive" and paper:
                prompt = (
                    f"A researcher has read papers about: {', '.join(concepts) or 'retrieval and NLP'}.\n"
                    f"Recommend this paper in ONE sentence, explaining why it bridges "
                    f"those topics. Paper: {paper.title}. Abstract: {paper.abstract[:300]}"
                )
                out = llm.generate(prompt, system="One concise sentence, no preamble.")
                if out and 20 < len(out) < 400:
                    return out.strip()
        except Exception:
            logger.debug("LLM explanation failed", exc_info=True)
        return base

    # ------------------------------------------------------- fallback
    def _similarity_fallback(self, s, session_id, read, top_k) -> dict:
        recs = []
        if self.recommender and read:
            seen = set(read)
            for aid in read:
                for rec in self.recommender.similar_papers(aid, top_k=top_k):
                    if rec["arxiv_id"] not in seen:
                        seen.add(rec["arxiv_id"])
                        recs.append({
                            "arxiv_id": rec["arxiv_id"],
                            "title": rec["title"],
                            "abstract": rec.get("abstract", "")[:280],
                            "published": rec.get("published", ""),
                            "score": rec.get("score", 0.0),
                            "bridges": 0,
                            "bridge_concepts": [],
                            "why": "Similar to a paper you saved (save one more "
                                   "from a different topic to unlock gap analysis).",
                        })
                    if len(recs) >= top_k:
                        break
                if len(recs) >= top_k:
                    break
        return {
            "mode": "similarity_fallback",
            "n_read": len(read),
            "message": "Save papers from at least two different topics to unlock "
                       "gap-bridging recommendations. For now, here are similar papers:",
            "recommendations": recs,
        }
