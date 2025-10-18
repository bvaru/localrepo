from __future__ import annotations

import os
from typing import List, Optional
import uuid

import streamlit as st

from ai_analyst.models import Document
from ai_analyst.pdf_extractor import extract_pdf_from_bytes, extract_pdf_from_path
from ai_analyst.utils import normalize_title, slugify
from ai_analyst.store import DocumentStore
from ai_analyst.chunker import ChunkingConfig
from ai_analyst.retriever import BM25Retriever
from ai_analyst.sections import extract_sections, extract_figures_and_tables, extract_references
from ai_analyst.summarizer import frequency_summarize
from ai_analyst.qa import answer_query


st.set_page_config(page_title="AI Research & Document Analyst", layout="wide")

if "store" not in st.session_state:
    st.session_state["store"] = DocumentStore()
if "retriever" not in st.session_state:
    st.session_state["retriever"] = BM25Retriever(st.session_state["store"])  # empty

store: DocumentStore = st.session_state["store"]
retriever: BM25Retriever = st.session_state["retriever"]


def add_document_from_upload(uploaded_file) -> Optional[Document]:
    if uploaded_file is None:
        return None
    data = uploaded_file.read()
    title, pages = extract_pdf_from_bytes(data)
    if not pages:
        st.error("Could not extract text from the uploaded PDF.")
        return None
    base_name = os.path.splitext(uploaded_file.name)[0]
    title_final = normalize_title(title, base_name)
    doc = Document(
        id=str(uuid.uuid4()),
        title=title_final,
        source_path=None,
        num_pages=len(pages),
        text_by_page=pages,
    )
    store.add_document(doc, config=ChunkingConfig())
    retriever.refresh()
    return doc


def add_document_from_path(path: str) -> Optional[Document]:
    title, pages = extract_pdf_from_path(path)
    if not pages:
        st.error(f"Could not extract text from: {path}")
        return None
    title_final = normalize_title(title, os.path.basename(path))
    doc = Document(
        id=str(uuid.uuid4()),
        title=title_final,
        source_path=path,
        num_pages=len(pages),
        text_by_page=pages,
    )
    store.add_document(doc, config=ChunkingConfig())
    retriever.refresh()
    return doc


st.title("AI Research & Document Analyst")

with st.sidebar:
    st.header("Upload PDFs")
    uploads = st.file_uploader("Upload research PDFs", type=["pdf"], accept_multiple_files=True)
    if uploads:
        for up in uploads:
            add_document_from_upload(up)
        st.success(f"Added {len(uploads)} document(s)")

    st.divider()
    st.caption("Optional: load a PDF from server path")
    path_input = st.text_input("PDF path on server", value="")
    if st.button("Load from path") and path_input.strip():
        add_document_from_path(path_input.strip())

# Tabs for features
overview_tab, ask_tab, sections_tab, compare_tab, refs_tab = st.tabs(
    ["Overview", "Ask", "Sections", "Compare", "References"]
)

with overview_tab:
    docs = store.list_documents()
    if not docs:
        st.info("Upload PDFs to begin.")
    else:
        for d in docs:
            with st.expander(f"{d.title} ({d.num_pages} pages)"):
                text = d.full_text()
                summary = frequency_summarize(text, max_sentences=6)
                st.markdown("**Summary**")
                st.write(summary)
                st.markdown("**First page excerpt**")
                st.text(d.text_by_page.get(1, "")[:1200])

with ask_tab:
    docs = store.list_documents()
    selected_docs = st.multiselect(
        "Restrict search to documents (optional)",
        options=[d.id for d in docs],
        format_func=lambda i: next((d.title for d in docs if d.id == i), i),
    )
    query = st.text_input("Ask a question about the documents")
    k = st.slider("Top passages", min_value=3, max_value=12, value=6)
    if st.button("Answer") and query.strip():
        restrict = selected_docs if selected_docs else None
        out = answer_query(retriever, query=query.strip(), k=k, restrict_doc_ids=restrict)
        st.markdown("**Answer**")
        st.write(out["answer"])  # type: ignore
        cits: List[str] = out.get("citations", [])  # type: ignore
        if cits:
            st.markdown("**Citations**")
            for i, c in enumerate(cits, start=1):
                st.write(f"[{i}] {c}")

with sections_tab:
    docs = store.list_documents()
    if not docs:
        st.info("Upload PDFs to extract sections.")
    else:
        doc_sel = st.selectbox("Select a document", options=[d.id for d in docs], format_func=lambda i: next((d.title for d in docs if d.id == i), i))
        doc = next(d for d in docs if d.id == doc_sel)
        secs = extract_sections(doc)
        figs_tabs = extract_figures_and_tables(doc)
        st.markdown("**Detected Sections**")
        for s in secs:
            with st.expander(f"{s.name} (pp. {s.page_start}-{s.page_end})"):
                st.text(s.text[:3000])
        st.markdown("**Figures**")
        for f in figs_tabs["figures"]:
            st.write(f)
        st.markdown("**Tables**")
        for t in figs_tabs["tables"]:
            st.write(t)

with compare_tab:
    docs = store.list_documents()
    if len(docs) < 2:
        st.info("Upload at least two PDFs to compare.")
    else:
        sel = st.multiselect("Select documents to compare", options=[d.id for d in docs], format_func=lambda i: next((d.title for d in docs if d.id == i), i))
        compare_query = st.text_input("Optional: comparison focus (e.g., Methods differences)")
        if st.button("Compare") and len(sel) >= 2:
            # Simple comparison: per doc, summarize methods/results sections then contrast
            summaries = []
            for doc_id in sel:
                d = next(d for d in docs if d.id == doc_id)
                secs = extract_sections(d)
                methods = " ".join(s.text for s in secs if s.name == "Methods")
                results = " ".join(s.text for s in secs if s.name == "Results")
                joined = (("Methods: " + methods + "\n\n") if methods else "") + (("Results: " + results) if results else "")
                if not joined:
                    joined = d.full_text()
                sumtext = frequency_summarize(joined, max_sentences=6)
                summaries.append((d.title, sumtext))
            if summaries:
                st.markdown("**Per-document summaries**")
                for title, s in summaries:
                    with st.expander(title):
                        st.write(s)
                # Optional LLM-assisted contrast
                if compare_query.strip():
                    from ai_analyst import llm
                    prompt = "Compare these documents based on: " + compare_query + "\n\n" + "\n\n".join(
                        f"Title: {title}\nSummary: {s}" for title, s in summaries
                    )
                    resp = llm.chat([
                        {"role": "system", "content": "You are a helpful research assistant. Contrast key differences clearly."},
                        {"role": "user", "content": prompt},
                    ])
                    if resp:
                        st.markdown("**LLM Comparison**")
                        st.write(resp)

with refs_tab:
    docs = store.list_documents()
    if not docs:
        st.info("Upload PDFs to extract references.")
    else:
        doc_sel = st.selectbox("Select a document", options=[d.id for d in docs], format_func=lambda i: next((d.title for d in docs if d.id == i), i))
        doc = next(d for d in docs if d.id == doc_sel)
        refs = extract_references(doc)
        if not refs:
            st.info("No references detected.")
        else:
            for r in refs:
                st.write(r.raw_text)
