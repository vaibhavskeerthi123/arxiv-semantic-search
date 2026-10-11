"""Streamlit frontend for the Real-Time Semantic Search Engine.

Run the API first (uvicorn backend.app.main:app), then:
    streamlit run frontend/streamlit_app.py
Set API_BASE env var if the backend is not on localhost:8000.
"""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st
import streamlit.components.v1 as components  # noqa: F401  (ensures st.components.v1.html)

API = os.environ.get("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="arXiv Semantic Search", page_icon="🔎", layout="wide"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:16]
if "results" not in st.session_state:
    st.session_state.results = []
if "single_chat" not in st.session_state:
    st.session_state.single_chat = {}   # arxiv_id -> [turns]
if "multi_chat" not in st.session_state:
    st.session_state.multi_chat = []


def api_post(path: str, payload: dict, timeout: int = 300) -> dict | None:
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def api_get(path: str, timeout: int = 120) -> dict | list | None:
    try:
        r = requests.get(f"{API}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def api_del(path: str, timeout: int = 60) -> dict | None:
    try:
        r = requests.delete(f"{API}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def _render_graph_html(gdata: dict) -> str:
    """Self-contained interactive graph via vis-network (CDN, no pip dep).

    Physics-based layout; hover shows labels; dragging/zoom supported. Styled
    for the app's dark theme. Paper nodes are circles, concept nodes are dots
    sized by how many papers mention them.
    """
    import json as _json

    nodes = []
    for n in gdata["nodes"]:
        shape = "dot" if n["type"] != "paper" else "square"
        title = n.get("title", n["label"])
        nodes.append({
            "id": n["id"], "label": n["label"], "color": n["color"],
            "shape": shape, "value": n.get("value", 1),
            "title": f"{title} ({n['type']})",
            "font": {"color": "#e6e6e6", "size": 13},
        })
    edges = []
    for e in gdata["edges"]:
        is_cc = e.get("kind") == "concept_concept"
        edges.append({
            "from": e["source"], "to": e["target"],
            "label": "" if is_cc else e.get("relation", ""),
            "color": {"color": "#3a4a5a" if is_cc else "#55657a",
                      "opacity": 0.6},
            "dashes": is_cc,
            "font": {"color": "#8a97a8", "size": 9, "strokeWidth": 0},
            "arrows": "" if is_cc else "to",
            "width": 1 + float(e.get("weight", 1)) * 0.3 if is_cc else 1,
        })
    data = _json.dumps({"nodes": nodes, "edges": edges})
    return f"""
<div id="kg" style="width:100%;height:540px;
     background:#0e1117;border:1px solid #262730;border-radius:8px;"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/vis-network.min.js"></script>
<script>
  const g = {data};
  const container = document.getElementById('kg');
  const data = {{
    nodes: new vis.DataSet(g.nodes),
    edges: new vis.DataSet(g.edges)
  }};
  const options = {{
    nodes: {{ borderWidth: 0, scaling: {{ min: 6, max: 34 }} }},
    edges: {{ smooth: {{ type: 'continuous' }}, selectionWidth: 2 }},
    physics: {{
      barnesHut: {{ gravitationalConstant: -9000, springLength: 130,
                    springConstant: 0.03, avoidOverlap: 0.4 }},
      stabilization: {{ iterations: 180 }}
    }},
    interaction: {{ hover: true, tooltipDelay: 120, navigationButtons: false }}
  }};
  const network = new vis.Network(container, data, options);
  network.on('stabilizationIterationsDone', () => network.setOptions({{physics:false}}));
</script>
"""


st.title("🔎 Real-Time Semantic Search Engine for Research Papers")
st.caption(
    "Hybrid BM25 + dense retrieval · cross-encoder reranking · FAISS · "
    "on-demand PDF indexing (only when you pick a paper for QA) · "
    "single & multi-document RAG QA"
)

(
    tab_search, tab_single, tab_multi, tab_recs,
    tab_graph, tab_next, tab_admin,
) = st.tabs(
    ["Search", "📄 Single-paper QA", "📚 Multi-paper QA", "✨ Recommendations",
     "🕸️ Knowledge Graph", "🧭 Read Next", "⚙️ Admin"]
)

# ------------------------------------------------------------------ Search
with tab_search:
    with st.form("search_form"):
        col1, col2, col3 = st.columns([6, 1, 1])
        query = col1.text_input(
            "Search query",
            placeholder="e.g. hybrid retrieval with reranking for scientific RAG",
            label_visibility="collapsed",
        )
        top_k = col2.number_input("Top-k", 1, 30, 10, label_visibility="collapsed")
        submitted = col3.form_submit_button("Search", use_container_width=True)
    _stats = api_get("/admin/stats", timeout=10)
    if _stats:
        st.caption(
            f"Local corpus: **{_stats['papers']} papers** · "
            f"{_stats['chunk_vectors']} full-text chunks indexed · "
            f"LLM: {_stats['llm_backend']}"
        )
    fetch_live = st.toggle(
        "Fetch fresh papers from arXiv API (real-time)",
        value=True,
        help="Adds one bounded HTTP call (hard timeout, default 8s) to the "
             "arXiv API per search. If it times out or fails, search continues "
             "over the local corpus and explains what happened. When off, "
             "search runs over the locally indexed corpus only.",
    )

    if submitted and query.strip():
        with st.spinner("Hybrid retrieval → reranking… (metadata only — no PDF downloads)"):
            data = api_post(
                "/search",
                {
                    "query": query,
                    "top_k": int(top_k),
                    "fetch_arxiv": fetch_live,
                    "session_id": st.session_state.session_id,
                },
            )
        if data:
            st.session_state.results = data["results"]
            if data.get("notice"):
                st.warning(data["notice"])
            if not data["results"]:
                st.info(
                    "No papers matched. Broaden the query, or check the "
                    "Admin tab — an empty corpus means nothing has been "
                    "ingested yet."
                )
            meta = f"{data['latency_ms']:.0f} ms"
            t = data.get("timings_ms") or {}
            if t.get("arxiv_fetch", 0) > 0:
                meta += (f" (arXiv fetch {t['arxiv_fetch']:.0f} · "
                         f"retrieval+rerank {t['retrieve_rerank']:.0f})")
            if data["expansions"]:
                meta += " · expanded with: " + ", ".join(data["expansions"])
            st.caption(meta)

    for r in st.session_state.results:
        with st.container(border=True):
            head, score = st.columns([8, 1])
            head.markdown(
                f"**[{r['title']}](https://arxiv.org/abs/{r['arxiv_id']})**"
            )
            score.metric("score", f"{r.get('score', 0):.3f}", label_visibility="collapsed")
            st.caption(
                f"arXiv:{r['arxiv_id']} · {r.get('published','')} · "
                f"{r.get('categories','')} · full text: "
                f"{'indexed ✅' if r.get('pdf_status') == 'embedded' else 'not downloaded (indexed on QA)'}"
            )
            st.write(
                r["abstract"][:600] + ("…" if len(r["abstract"]) > 600 else "")
            )
            c1, c2, c3 = st.columns(3)
            if c1.button("💬 Ask this paper", key=f"ask_{r['arxiv_id']}"):
                st.session_state.active_paper = r["arxiv_id"]
                st.session_state.active_title = r["title"]
                st.toast("Opened in Single-paper QA tab")
            if c2.button("✨ Similar papers", key=f"rec_{r['arxiv_id']}"):
                st.session_state.rec_paper = r["arxiv_id"]
                st.toast("Opened in Recommendations tab")
            if c3.button("🔖 Save to reading list", key=f"save_{r['arxiv_id']}"):
                api_post("/reading-list", {
                    "session_id": st.session_state.session_id,
                    "arxiv_id": r["arxiv_id"], "action": "saved",
                })
                st.toast("Saved — see the 'Read Next' tab")

# --------------------------------------------------------------- Single QA
with tab_single:
    ids = [r["arxiv_id"] for r in st.session_state.results]
    default = st.session_state.get("active_paper")
    idx = ids.index(default) if default in ids else 0
    paper_id = st.selectbox(
        "Paper (from your last search, or type an arXiv id below)",
        options=ids or ["—"],
        index=idx if ids else 0,
    )
    manual = st.text_input("…or arXiv id", placeholder="2503.23013")
    paper_id = manual.strip() or (paper_id if paper_id != "—" else "")

    if paper_id:
        # On-demand indexing: the PDF is downloaded/parsed/embedded only
        # now that this paper was selected for QA (idempotent server-side).
        if st.session_state.get("indexed_paper") != paper_id:
            with st.status(
                f"Preparing {paper_id}: downloading PDF → parsing → "
                "chunking → embedding…", expanded=False,
            ) as status_box:
                info = api_post(f"/papers/{paper_id}/index", {}, timeout=600)
                if info:
                    st.session_state.indexed_paper = paper_id
                    status_box.update(
                        label=f"Ready — {info.get('n_chunks', '?')} chunks "
                              f"indexed ({info.get('pdf_status', '')})",
                        state="complete",
                    )
        chat = st.session_state.single_chat.setdefault(paper_id, [])
        for turn in chat:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
        q = st.chat_input("Ask about this paper…")
        if q:
            chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"), st.spinner("Retrieving relevant chunks → generating grounded answer…"):
                data = api_post(
                    "/qa/single",
                    {
                        "question": q,
                        "arxiv_id": paper_id,
                        "history": chat[:-1],
                        "session_id": st.session_state.session_id,
                    },
                )
                if data:
                    st.markdown(data["answer"])
                    with st.expander("Sources"):
                        for s_ in data["sources"]:
                            st.markdown(
                                f"**[{s_['n']}]** *{s_['section']}* — {s_['snippet']}…"
                            )
                    chat.append({"role": "assistant", "content": data["answer"]})
    else:
        st.info("Run a search first, or paste an arXiv id.")

# ---------------------------------------------------------------- Multi QA
with tab_multi:
    st.markdown(
        "Ask a question **across several papers** — the selected papers are "
        "downloaded & indexed on demand, then the system extracts evidence "
        "per paper (map) and synthesizes one answer (reduce)."
    )
    options = {
        f"{r['title'][:80]} ({r['arxiv_id']})": r["arxiv_id"]
        for r in st.session_state.results
    }
    chosen = st.multiselect(
        "Papers (leave empty to let retrieval pick the best papers)",
        options=list(options),
    )
    for turn in st.session_state.multi_chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
    mq = st.chat_input("e.g. Compare the chunking strategies used across these papers")
    if mq:
        st.session_state.multi_chat.append({"role": "user", "content": mq})
        with st.chat_message("user"):
            st.markdown(mq)
        with st.chat_message("assistant"), st.spinner("Indexing selected papers (first time only) → map-reduce synthesis…"):
            data = api_post(
                "/qa/multi",
                {
                    "question": mq,
                    "paper_ids": [options[c] for c in chosen] or None,
                    "history": st.session_state.multi_chat[:-1],
                    "session_id": st.session_state.session_id,
                },
            )
            if data:
                st.markdown(data["answer"])
                with st.expander("Per-paper evidence"):
                    for s_ in data["sources"]:
                        st.markdown(
                            f"**[{s_['n']}] {s_['title']}** (arXiv:{s_['arxiv_id']})\n\n"
                            f"{s_['snippet']}…"
                        )
                st.session_state.multi_chat.append(
                    {"role": "assistant", "content": data["answer"]}
                )

# ---------------------------------------------------------- Recommendations
with tab_recs:
    rec_default = st.session_state.get("rec_paper", "")
    rid = st.text_input("arXiv id", value=rec_default, placeholder="2503.23013")
    if st.button("Find similar papers") and rid.strip():
        data = api_get(f"/papers/{rid.strip()}/recommendations")
        if data:
            for rec in data["recommendations"]:
                with st.container(border=True):
                    st.markdown(
                        f"**[{rec['title']}](https://arxiv.org/abs/{rec['arxiv_id']})** "
                        f"· similarity {rec['score']:.3f}"
                    )
                    st.caption(f"arXiv:{rec['arxiv_id']} · {rec.get('categories','')}")
                    st.write(rec["abstract"][:400] + "…")

# ---------------------------------------------------------- Knowledge Graph
with tab_graph:
    st.subheader("🕸️ Paper → Knowledge Graph")
    st.caption(
        "Methods, datasets, metrics and tasks extracted from the corpus, and "
        "how papers connect through them — see which papers *introduce* a "
        "method vs. *build on* it."
    )

    gc1, gc2, gc3, gc4 = st.columns([2, 2, 2, 1.4])
    type_opts = ["method", "dataset", "metric", "task", "contribution"]
    sel_types = gc1.multiselect(
        "Show concept types", type_opts,
        default=["method", "dataset", "task"],
    )
    min_deg = gc2.slider("Min. papers per concept", 1, 6, 1)
    max_nodes = gc3.slider("Max nodes", 20, 250, 120, step=10)
    if gc4.button("🔄 Rebuild graph", use_container_width=True):
        with st.spinner("Extracting entities & building graph…"):
            s = api_post("/graph/build", {})
        if s:
            st.success(
                f"Built: {s['entities']} concepts · {s['paper_entity_links']} "
                f"links from {s['papers_processed']} papers"
            )

    gstats = api_get("/graph/stats")
    if gstats and gstats.get("entities", 0) == 0:
        st.info("Graph is empty — click **Rebuild graph** to extract concepts "
                "from the corpus.")
    elif gstats:
        bt = gstats.get("by_type", {})
        st.caption(
            f"Graph: **{gstats['entities']} concepts** "
            f"({', '.join(f'{v} {k}' for k, v in bt.items())}) · "
            f"{gstats['paper_entity_links']} paper–concept links · "
            f"{gstats['concept_edges']} concept–concept edges"
        )

    params = f"?min_degree={min_deg}&max_nodes={max_nodes}"
    if sel_types:
        params += "&etypes=" + ",".join(sel_types)
    gdata = api_get("/graph" + params)

    if gdata and gdata["nodes"]:
        st.components.v1.html(_render_graph_html(gdata), height=560, scrolling=False)
        # legend
        legend = " ".join(
            f"<span style='color:{c};font-weight:700'>●</span> {t}"
            for t, c in [
                ("paper", "#2F5D8A"), ("method", "#1E2761"),
                ("dataset", "#5B8A72"), ("metric", "#C08A2E"),
                ("task", "#8A4B6B"),
            ]
        )
        st.markdown(legend, unsafe_allow_html=True)

        st.divider()
        st.markdown("**🔍 Explore a concept** — which papers introduce or use it")
        concept_nodes = [n for n in gdata["nodes"] if n["type"] != "paper"]
        if concept_nodes:
            labels = {f"{n['label']} ({n['type']})": n["id"][1:]
                      for n in sorted(concept_nodes, key=lambda x: -x.get("value", 0))}
            pick = st.selectbox("Concept", list(labels.keys()))
            if pick:
                det = api_get(f"/graph/entity/{labels[pick]}")
                if det and det.get("by_relation"):
                    rel_labels = {
                        "introduces": "📌 Introduced by",
                        "uses": "🔧 Used / built on by",
                        "evaluates_on": "📊 Evaluated on by",
                        "addresses": "🎯 Addressed by",
                        "reports": "📈 Reported by",
                    }
                    for rel, papers in det["by_relation"].items():
                        st.markdown(f"**{rel_labels.get(rel, rel)}** ({len(papers)})")
                        for p in papers[:8]:
                            st.markdown(
                                f"- [{p['title']}](https://arxiv.org/abs/{p['arxiv_id']}) "
                                f"· {p.get('published','')}"
                            )
    elif gdata:
        st.info("No concepts match these filters — lower the minimum or add types.")

# ----------------------------------------------------------------- Read Next
with tab_next:
    st.subheader("🧭 What Should I Read Next?")
    st.caption(
        "Goes beyond 'similar papers': it finds concepts you've read about "
        "separately and recommends papers that **bridge** them."
    )

    rl = api_get(f"/reading-list?session_id={st.session_state.session_id}") or []
    lc, rc = st.columns([1, 2])
    with lc:
        st.markdown(f"**Your reading list ({len(rl)})**")
        if not rl:
            st.caption("Empty — add papers from the Search tab (🔖), or below.")
        for item in rl:
            cols = st.columns([5, 1])
            cols[0].markdown(f"<small>{item['title'][:48]}</small>",
                             unsafe_allow_html=True)
            if cols[1].button("✕", key=f"rm_{item['arxiv_id']}"):
                api_del(f"/reading-list?session_id={st.session_state.session_id}"
                        f"&arxiv_id={item['arxiv_id']}")
                st.rerun()
        add_id = st.text_input("Add by arXiv id", placeholder="2005.11401",
                               key="next_add")
        if st.button("➕ Add") and add_id.strip():
            api_post("/reading-list", {
                "session_id": st.session_state.session_id,
                "arxiv_id": add_id.strip(), "action": "saved",
            })
            st.rerun()

    with rc:
        if st.button("🧭 Recommend what to read next", type="primary"):
            st.session_state.next_reads = api_get(
                f"/next-reads?session_id={st.session_state.session_id}&top_k=6"
            )
        nd = st.session_state.get("next_reads")
        if nd:
            badge = {"gap_analysis": "🧠 Gap analysis",
                     "similarity_fallback": "↔️ Similarity (add more to unlock gaps)",
                     "empty": "—"}.get(nd["mode"], nd["mode"])
            st.markdown(f"*{badge}*")
            st.info(nd["message"])
            for rec in nd["recommendations"]:
                with st.container(border=True):
                    top = st.columns([7, 1])
                    top[0].markdown(
                        f"**[{rec['title']}](https://arxiv.org/abs/{rec['arxiv_id']})**"
                    )
                    if rec.get("bridges"):
                        top[1].markdown(f"🌉 **{rec['bridges']}**")
                    if rec.get("bridge_concepts"):
                        st.caption("Bridges: " + " · ".join(
                            f"`{c}`" for c in rec["bridge_concepts"]))
                    st.write(f"💡 {rec['why']}")
                    with st.expander("Abstract"):
                        st.write(rec.get("abstract", ""))

# -------------------------------------------------------------------- Admin
with tab_admin:
    c1, c2 = st.columns(2)
    if c1.button("Refresh stats"):
        st.session_state.stats = api_get("/admin/stats")
    if c2.button("Rebuild indexes", type="secondary"):
        with st.spinner("Rebuilding…"):
            st.session_state.stats = api_post("/admin/rebuild", {})
    stats = st.session_state.get("stats") or api_get("/admin/stats")
    if stats:
        cols = st.columns(4)
        cols[0].metric("Papers", stats["papers"])
        cols[1].metric("Chunks", stats["chunks"])
        cols[2].metric("Paper vectors", stats["paper_vectors"])
        cols[3].metric("Chunk vectors", stats["chunk_vectors"])
        st.caption(
            f"Embedder: {stats['embedding_backend']} (dim {stats['embedding_dim']}) · "
            f"LLM backend: {stats['llm_backend']} · queries logged: {stats['queries']}"
        )
    st.subheader("Recent queries")
    hist = api_get("/admin/history?limit=25")
    if hist:
        st.dataframe(hist, use_container_width=True)
