# Design: Upgrade RAG FAQ — embeddings + FAISS + eval (offline)

**Date:** 2026-11-01  
**Project:** `01-rag-cardapio`  
**Status:** Approved in brainstorming; pending implementation plan  
**Goal:** Strengthen the GenAI portfolio for applied-AI / trainee roles by deepening the existing RAG project (not starting a new one).

## Context

Current portfolio GenAI trio:

| Project | Strength | Gap |
|---------|----------|-----|
| 01 RAG FAQ | Grounded FAQ + intent + Gradio | TF-IDF only; no eval |
| 02 Ops agent | Tools + JSON trace | Out of scope this cycle |
| 03 Report automation | Business impact metric | Out of scope this cycle |

**Decisions locked**

- Target: GenAI / IA aplicada roles  
- Strategy: deepen existing work  
- First upgrade: **01 RAG**  
- Timebox: **~1–2 weeks**  
- Runtime: **100% local/offline** (no API key required)  
- Approach: **FAISS + sentence-transformers + retrieval eval**, TF-IDF kept as baseline  

## Objectives

1. Add semantic retrieval via local embeddings + FAISS.  
2. Keep TF-IDF as a comparable baseline (`mode=embeddings|tfidf`).  
3. Ship a reproducible eval (Recall@3, MRR, citation match) with numbers in the README.  
4. Preserve Gradio demo and offline template generation as the default path.  

**Note on legacy OpenAI:** the existing optional `OPENAI_API_KEY` generation path may remain in code for compatibility, but it is **not** required for demo, eval, or Definition of Done. This upgrade must work with no API key.

## Non-goals (explicit)

- Chroma / other vector DBs  
- FastAPI / production HTTP API  
- Local LLM for generation  
- Wiring project 02 to call the RAG  
- Hugging Face Hub upload (separate backlog item)  
- Hybrid RRF fusion as a required deliverable (optional stretch if time remains)

## Architecture

```text
FAQ (faq.json / faq_hf.jsonl)
    ↓
Indexer → sentence-transformers → FAISS (+ metadata)
    ↓
Retriever → top-k Hits (embeddings | tfidf)
    ↓
Intent classifier (existing) → Generator (grounded templates, offline)
    ↓
Gradio UI  |  Eval CLI (golden_qa.jsonl → metrics table)
```

## Components

| Piece | Responsibility |
|-------|----------------|
| `FaqRag` (evolve) | Orchestrates retrieve → intent → generate; accepts `mode` |
| `EmbeddingIndex` (new) | Encode FAQ, build/load FAISS, top-k search |
| `TfidfIndex` (extract from current) | Existing baseline behind the same retrieve interface |
| `src/eval.py` (new) | Golden-set eval for both modes |
| `src/app.py` | Gradio: mode dropdown, answer, sources, scores |

### Shared retrieve contract

- **In:** `query: str`, `k: int`  
- **Out:** `list[Hit]` where `Hit` includes `doc_id`, `score`, text/metadata  

FAQ schema unchanged: `id`, `categoria`, `pergunta`, `resposta` (e.g. `faq-001`).

### Dependencies

- Add: `sentence-transformers`, `faiss-cpu`  
- Keep: `scikit-learn`, `gradio`, existing stack  

### Default embedding model

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (CPU-friendly, multilingual/PT).

### Persistence

- Index dir: `data/index/` (FAISS + sidecars)  
- Built on first embeddings run; documented in README  
- Prefer generate-on-first-run over committing large binaries  

### Golden set

- Path: `data/golden_qa.jsonl`  
- Size: ~25–40 items  
- Record shape: `{ "query": str, "relevant_doc_ids": [str, ...], "notes": str? }`  
- Cover real FAQ intents (entrega, pagamento, pedido, conta, promoção, etc.)

## Data flow and UX

1. User selects mode (`embeddings` default, or `tfidf`).  
2. Query normalized (existing `normalize_query`).  
3. Mode-specific retriever returns top-k hits.  
4. Intent classified (existing logic).  
5. Offline generator builds grounded answer with citations.  
6. UI shows answer, intent, sources + scores, active mode.  

**First embeddings run:** download model once → encode FAQ → save FAISS → serve (README warns ~1–3 min first time).

## Error handling

| Case | Behavior |
|------|----------|
| Empty query | UI validation |
| Invalid/empty FAQ | Clear boot error |
| Embeddings/FAISS load failure | Fallback to TF-IDF + UI warning |
| No hit / low score | Grounded refusal; do not invent (reuse existing score threshold in `FaqRag` if present; otherwise refuse when top score is below a documented constant, e.g. cosine < 0.25 for embeddings) |
| Gradio errors | User-facing message; no stack trace in UI |

## Evaluation and testing

### Eval CLI

```bash
cd 01-rag-cardapio && source .venv/bin/activate
python src/eval.py
```

- Metrics: Recall@3, MRR, citation match  
- Output: terminal table + optional `output/eval_latest.md`  
- Quality gate: embeddings Recall@3 **≥** tfidf; if tie, document why and show cases where embeddings wins  

### Manual / demo

```bash
python src/app.py   # http://127.0.0.1:7860
python -c "from src.rag import FaqRag; print(FaqRag(mode='embeddings').answer('cobraram duas vezes'))"
```

### Regression

- Update monorepo `./scripts/smoke_test.sh` for both modes  
- Optional light unit tests (retrieve k, index reload, fallback); add `pytest` only if low friction  

## How to run (target UX after implementation)

```bash
cd 01-rag-cardapio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/app.py
```

No `OPENAI_API_KEY` required for this upgrade path.

## Definition of Done

1. `embeddings` and `tfidf` modes work in CLI and Gradio  
2. FAISS index builds locally on first embeddings run  
3. `eval.py` is reproducible; README includes metrics table  
4. Zero API-key dependency for the demo/eval path  
5. `01-rag-cardapio/README.md` updated (run / test / eval)  
6. Non-goals above remain out of this delivery  

## Stretch (only if time left inside 1–2 weeks)

- Light hybrid mode (simple score fusion or RRF) as third Gradio option  
- 2–4 unit tests with pytest  

## Success criteria (portfolio)

A reviewer can clone, run offline, compare two retrievers in the UI, and see **numbers** proving semantic retrieval helps — without needing an API key.
