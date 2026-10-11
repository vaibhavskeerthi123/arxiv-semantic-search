"""Typed entity + relation extraction from research-paper text.

Extracts five presentation-relevant entity types — **method, dataset,
metric, task, contribution** — and the *relation* each has to the paper
(introduces / uses / evaluates_on / addresses / reports).

Design goals
------------
* **Always works offline.** The baseline is a deterministic
  gazetteer + linguistic-pattern extractor with zero ML dependencies, so
  it runs in the same environment as the hash-embedder / extractive-LLM
  fallback. This matters: the deployed demo has no torch and no Ollama.
* **Optionally smarter.** When a real LLM backend is available
  (``safe_generate`` returns something better than extractive), an LLM pass
  can enrich the baseline — but the baseline alone produces a usable graph.
* **Fails soft.** Missing/empty abstracts, odd unicode, or a raised LLM
  call never crash extraction; the worst case is "no entities for this
  paper", which the builder handles.

The extractor returns ``Mention`` records (surface name + type + relation
+ confidence + evidence snippet). Normalization / dedup happens in the
builder, so the extractor stays stateless and easy to test.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Mention:
    name: str
    etype: str          # one of ENTITY_TYPES
    relation: str       # one of RELATIONS
    confidence: float
    evidence: str = ""


# --------------------------------------------------------------------------
# Gazetteers — curated, high-precision seeds for the CS/ML arXiv domain.
# These are deliberately well-known names so matches are trustworthy even
# without an LLM. Matching is word-boundary, case-insensitive.
# --------------------------------------------------------------------------
_METHODS = [
    "BM25", "TF-IDF", "BERT", "RoBERTa", "DistilBERT", "SciBERT", "SPECTER",
    "Sentence-BERT", "SBERT", "MiniLM", "ColBERT", "DPR", "RAG", "Self-RAG",
    "Corrective RAG", "GraphRAG", "RAPTOR", "HyDE", "FiD", "REALM", "RETRO",
    "Atlas", "Transformer", "Transformer-XL", "Longformer", "GPT", "GPT-3",
    "GPT-4", "LLaMA", "Llama 2", "T5", "FLAN", "CLIP", "LoRA", "QLoRA",
    "FlashAttention", "FAISS", "HNSW", "IVF", "IVF-PQ", "ScaNN",
    "cross-encoder", "bi-encoder", "dual-encoder", "reranker", "reranking",
    "dense retrieval", "sparse retrieval", "hybrid retrieval",
    "reciprocal rank fusion", "RRF", "contrastive learning",
    "knowledge distillation", "instruction tuning", "RLHF", "chain-of-thought",
    "map-reduce", "approximate nearest neighbor", "ANN", "quantization",
    "dynamic alpha tuning", "cross-attention", "self-attention",
]
_DATASETS = [
    "MS MARCO", "BEIR", "MTEB", "Natural Questions", "TriviaQA", "SQuAD",
    "HotpotQA", "FEVER", "GLUE", "SuperGLUE", "ImageNet", "C4", "Wikipedia",
    "arXiv", "SciRerankBench", "TREC", "KILT", "Common Crawl", "Pile",
    "MMLU", "NFCorpus", "FiQA", "SciFact", "SciDocs", "Quora",
]
_METRICS = [
    "Recall@k", "Recall@10", "Recall@100", "MRR", "nDCG", "nDCG@10",
    "NDCG", "MAP", "precision", "recall", "F1", "F1 score", "accuracy",
    "BLEU", "ROUGE", "perplexity", "exact match", "EM", "hit rate",
    "latency", "throughput", "faithfulness", "answer relevance",
]
# Tasks / research problems (phrased as the problem being solved).
_TASKS = [
    "information retrieval", "passage retrieval", "document retrieval",
    "open-domain question answering", "question answering", "multi-document QA",
    "multi-hop reasoning", "semantic search", "text classification",
    "named entity recognition", "summarization", "machine translation",
    "recommendation", "citation matching", "fact verification",
    "retrieval-augmented generation", "re-ranking", "entity linking",
    "knowledge graph construction", "text embedding", "sentence similarity",
]

# Precompiled, longest-first so "dense retrieval" wins over "retrieval".
def _compile(terms: list[str]) -> list[tuple[str, re.Pattern]]:
    uniq = sorted(set(terms), key=len, reverse=True)
    out = []
    for t in uniq:
        pat = re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.IGNORECASE)
        out.append((t, pat))
    return out


_GAZETTEERS = {
    "method": _compile(_METHODS),
    "dataset": _compile(_DATASETS),
    "metric": _compile(_METRICS),
    "task": _compile(_TASKS),
}

# --------------------------------------------------------------------------
# Linguistic patterns — catch entities not in the gazetteer by how they are
# introduced. Group 1 is the candidate name.
# --------------------------------------------------------------------------
_I = re.IGNORECASE
_NAME = r"([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,3})"
# A looser, NON-GREEDY name that also allows lowercase multi-word method
# names, common in ML abstracts ("we propose dynamic alpha tuning"). Kept in
# check by the stop-list, length bounds and a trailing method cue.
_NAME_LC = r"([A-Za-z][A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+){0,3}?)"

_INTRODUCE_PATTERNS = [
    # lowercase/any-case method name followed by an appositive cue (strongest)
    re.compile(r"\bwe (?:propose|present|introduce|develop|design)\s+" + _NAME_LC +
               r"\s*,\s*(?:a|an)\s+(?:novel\s+|new\s+)?"
               r"(?:method|model|framework|approach|system|technique|strategy)", _I),
    re.compile(r"\bwe (?:propose|present|introduce|develop|design)\s+" + _NAME, _I),
    re.compile(r"\b(?:propose|present|introduce)s?\s+(?:a|an|the)?\s*" + _NAME +
               r"(?:,| that| which| method| model| framework| approach| system)", _I),
    # summary-style openings ("Introduces the Transformer, …",
    # "Proposes BERT, which …", "Presents RETRO …") — common when the corpus
    # stores condensed third-person abstracts.
    re.compile(r"(?:^|\.\s+)(?:introduce|propose|present)s?\s+(?:a|an|the)?\s*" + _NAME, _I),
    re.compile(_NAME + r"\s*,?\s*(?:a|an)\s+(?:novel|new)\s+"
               r"(?:method|model|framework|approach|system|architecture)", _I),
]
_USES_PATTERNS = [
    re.compile(r"\b(?:evaluated?|evaluation|experiments?|benchmark(?:ed)?|tested?)"
               r"\s+on\s+(?:the\s+)?" + _NAME, _I),
    re.compile(r"\bon\s+the\s+" + _NAME + r"\s+(?:dataset|benchmark|corpus)", _I),
    re.compile(r"\b(?:using|based on|built on|leverages?)\s+" + _NAME, _I),
]
_METRIC_CONTEXT = re.compile(
    r"\b(?:achieves?|improv\w+|outperform\w+|gains?|reaches?)\b", re.IGNORECASE
)

# Phrases that are too generic to be useful entity names.
_STOP_NAMES = {
    "we", "this", "the", "our", "these", "they", "it", "a", "an",
    "paper", "method", "model", "approach", "system", "results", "figure",
    "table", "section", "however", "moreover", "finally", "in", "for",
    "experiments", "experiment", "evaluation", "related work", "introduction",
}

_MAX_ENTITIES_PER_PAPER = 40


def _clean(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip(" .,:;()[]\"'")


def _is_noise(name: str) -> bool:
    low = name.lower()
    if low in _STOP_NAMES or len(low) < 2 or len(name) > 60:
        return True
    if not re.search(r"[A-Za-z]", name):
        return True
    # conjunction artifacts from greedy pattern capture ("BEIR and MS MARCO")
    if re.search(r"\b(?:and|or|with|for|the|using)\b", low) and " " in name:
        # allow known multi-word names that legitimately contain such words
        if low not in {"bag of words"}:
            return True
    return False


def extract_gazetteer(text: str) -> list[Mention]:
    """High-precision dictionary matches (always on)."""
    mentions: list[Mention] = []
    seen: set[tuple[str, str]] = set()
    for etype, patterns in _GAZETTEERS.items():
        for canonical, pat in patterns:
            m = pat.search(text)
            if not m:
                continue
            key = (canonical.lower(), etype)
            if key in seen:
                continue
            seen.add(key)
            # relation heuristic: datasets -> evaluates_on/uses, metrics ->
            # reports, tasks -> addresses, methods -> uses (default).
            relation = {
                "dataset": "evaluates_on",
                "metric": "reports",
                "task": "addresses",
                "method": "uses",
            }[etype]
            snippet = _snippet(text, m.start(), m.end())
            mentions.append(Mention(canonical, etype, relation, 0.85, snippet))
    return mentions


def extract_patterns(text: str) -> list[Mention]:
    """Lower-precision pattern matches for out-of-gazetteer names."""
    mentions: list[Mention] = []
    seen: set[str] = set()

    def _add(name: str, etype: str, relation: str, conf: float, span):
        name = _clean(name)
        if _is_noise(name) or name.lower() in seen:
            return
        seen.add(name.lower())
        mentions.append(Mention(name, etype, relation, conf, _snippet(text, *span)))

    for pat in _INTRODUCE_PATTERNS:
        for m in pat.finditer(text):
            _add(m.group(1), "method", "introduces", 0.6, (m.start(1), m.end(1)))
    for pat in _USES_PATTERNS:
        for m in pat.finditer(text):
            # a proper-noun phrase after "evaluated on" is usually a dataset
            _add(m.group(1), "dataset", "evaluates_on", 0.55, (m.start(1), m.end(1)))
    return mentions


def extract_contributions(text: str) -> list[Mention]:
    """Pull 1-2 short contribution statements as their own entity type."""
    out: list[Mention] = []
    for m in re.finditer(
        r"\bwe (propose|present|introduce|show|demonstrate|develop)\b[^.]{0,140}\.",
        text, re.IGNORECASE,
    ):
        sent = _clean(m.group(0))
        if 20 <= len(sent) <= 180:
            # name is a trimmed label; evidence keeps the full sentence
            label = sent[:70] + ("…" if len(sent) > 70 else "")
            out.append(Mention(label, "contribution", "reports", 0.5, sent))
        if len(out) >= 2:
            break
    return out


def _snippet(text: str, start: int, end: int, pad: int = 60) -> str:
    a = max(0, start - pad)
    b = min(len(text), end + pad)
    return _clean(text[a:b])


def extract_entities(
    title: str,
    abstract: str,
    full_text: str = "",
    use_llm: bool = False,
) -> list[Mention]:
    """Main entry point. Combines gazetteer + patterns (+ optional LLM).

    Never raises: any internal failure degrades to the subset that worked.
    ``full_text`` (concatenated chunk text, when the paper has been QA-
    indexed) sharpens extraction but is optional.
    """
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if not title and not abstract and not full_text:
        return []

    # Weight the title/abstract; cap full_text so a huge PDF doesn't dominate.
    text = f"{title}. {abstract}"
    if full_text:
        text += "\n" + full_text[:4000]

    mentions: list[Mention] = []
    for fn in (extract_gazetteer, extract_patterns, extract_contributions):
        try:
            mentions.extend(fn(text))
        except Exception:  # one extractor failing must not sink the rest
            logger.debug("extractor %s failed", fn.__name__, exc_info=True)

    if use_llm:
        try:
            mentions.extend(_llm_enrich(title, abstract))
        except Exception:
            logger.debug("LLM enrichment failed", exc_info=True)

    # Merge duplicates (same normalized name+type): keep the highest
    # confidence and prefer a more specific relation than the default.
    merged: dict[tuple[str, str], Mention] = {}
    _REL_PRIORITY = {"introduces": 3, "evaluates_on": 2, "addresses": 2,
                     "uses": 1, "reports": 0}
    for m in mentions:
        key = (m.name.lower(), m.etype)
        cur = merged.get(key)
        if cur is None:
            merged[key] = m
            continue
        # Keep the more informative relation (introduces > evaluates_on/…),
        # and otherwise the higher confidence surface form.
        best_rel = max(cur.relation, m.relation,
                       key=lambda r: _REL_PRIORITY.get(r, 0))
        winner = m if m.confidence > cur.confidence else cur
        winner.relation = best_rel
        merged[key] = winner
    out = sorted(merged.values(), key=lambda x: x.confidence, reverse=True)
    return out[:_MAX_ENTITIES_PER_PAPER]


# --------------------------------------------------------------------------
# Optional LLM enrichment. Uses the project's existing safe_generate so it
# inherits auto-backend selection + graceful degradation. Parsed defensively.
# --------------------------------------------------------------------------
_LLM_SYS = (
    "You extract structured entities from a research paper abstract. "
    "Return ONLY lines of the form TYPE|NAME|RELATION, one per line, where "
    "TYPE is one of method/dataset/metric/task, RELATION is one of "
    "introduces/uses/evaluates_on/addresses. No prose, no numbering."
)


def _llm_enrich(title: str, abstract: str) -> list[Mention]:
    from backend.app.services.llm.generator import get_llm

    llm = get_llm()
    if llm.name == "extractive":          # no real model -> skip, baseline covers it
        return []
    prompt = f"Title: {title}\nAbstract: {abstract}\n\nEntities:"
    raw = llm.generate(prompt, system=_LLM_SYS)
    out: list[Mention] = []
    valid_types = {"method", "dataset", "metric", "task"}
    valid_rels = {"introduces", "uses", "evaluates_on", "addresses"}
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        etype, name, relation = parts
        etype, relation = etype.lower(), relation.lower()
        name = _clean(name)
        if etype in valid_types and relation in valid_rels and not _is_noise(name):
            out.append(Mention(name, etype, relation, 0.75, "llm"))
    return out
