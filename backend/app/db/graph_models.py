"""Additive persistence for the Knowledge-Graph and Read-Next features.

These tables extend the existing metadata store (same ``Base`` as
``models.py``) without touching the ``papers`` / ``chunks`` / ``query_log``
schema. Importing this module is enough to register the tables, so
``init_db()`` (which calls ``Base.metadata.create_all``) creates them
automatically on the next boot — no migration tool required for SQLite,
and a plain ``create_all`` for PostgreSQL.

Design notes
------------
* An ``Entity`` is a normalized concept (a method, dataset, metric, task or
  contribution). ``norm_name`` is the dedup key so "BERT" and "bert" collapse.
* ``PaperEntity`` is the typed link between a paper and an entity, carrying
  the *relation* (introduces / uses / evaluates_on / addresses / reports)
  and a confidence score — this is what powers "which papers introduce or
  modify this method".
* ``EntityEdge`` stores derived entity<->entity relationships (co-occurrence
  / method-on-dataset etc.) so the graph view can draw concept-to-concept
  links without recomputing them each request.
* ``ReadingListItem`` records what a session saved or read; the Read-Next
  feature reasons over it. Keyed by (session_id, arxiv_id) so re-saving is
  idempotent.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.models import Base, utcnow

# Entity type vocabulary (kept small and presentation-friendly).
ENTITY_TYPES = ("method", "dataset", "metric", "task", "contribution")

# Relation vocabulary for paper -> entity links.
RELATIONS = ("introduces", "uses", "evaluates_on", "addresses", "reports")


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("norm_name", "etype", name="uq_entity_norm_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))          # display form
    norm_name: Mapped[str] = mapped_column(String(256), index=True)  # dedup key
    etype: Mapped[str] = mapped_column(String(32), index=True)       # ENTITY_TYPES
    n_papers: Mapped[int] = mapped_column(Integer, default=0)        # degree cache
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)

    links: Mapped[list["PaperEntity"]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "etype": self.etype,
            "n_papers": self.n_papers,
        }


class PaperEntity(Base):
    __tablename__ = "paper_entities"
    __table_args__ = (
        UniqueConstraint("paper_id", "entity_id", "relation", name="uq_paper_entity_rel"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id"), index=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), index=True)
    relation: Mapped[str] = mapped_column(String(32), default="reports")  # RELATIONS
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    evidence: Mapped[str] = mapped_column(Text, default="")  # snippet it came from

    entity: Mapped[Entity] = relationship(back_populates="links")


class EntityEdge(Base):
    """Derived concept<->concept edge (undirected, stored once with src<dst)."""

    __tablename__ = "entity_edges"
    __table_args__ = (
        UniqueConstraint("src_id", "dst_id", "kind", name="uq_entity_edge"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    src_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), index=True)
    dst_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32), default="co_occurs")
    weight: Mapped[float] = mapped_column(Float, default=1.0)


class ReadingListItem(Base):
    __tablename__ = "reading_list"
    __table_args__ = (
        UniqueConstraint("session_id", "arxiv_id", name="uq_reading_session_paper"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    arxiv_id: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(16), default="saved")  # saved | read
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
