# RAG Embeddings + FAISS + Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `01-rag-cardapio` with local sentence-transformers + FAISS retrieval, keep TF-IDF as baseline, and ship a reproducible golden-set eval (Recall@3, MRR, citation match) — fully offline.

**Architecture:** Extract `TfidfIndex` and add `EmbeddingIndex` behind a shared `retrieve(query, k) -> list[Hit]` contract. `FaqRag(mode=...)` orchestrates retrieve → intent → grounded template generation. Gradio exposes a mode dropdown; `src/eval.py` compares both modes on `data/golden_qa.jsonl`.

**Tech Stack:** Python 3, scikit-learn (existing), `sentence-transformers`, `faiss-cpu`, Gradio, pytest.

**Spec:** `docs/superpowers/specs/2026-11-01-rag-embeddings-eval-design.md`

## Global Constraints

- 100% local/offline for demo and eval (no API key required)
- Default embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Modes: `embeddings` (default) and `tfidf`
- TF-IDF grounded threshold stays `0.08`; embeddings grounded threshold: cosine/IP score `< 0.25` → not grounded
- Index path: `01-rag-cardapio/data/index/` (generate on first run; gitignored)
- Optional legacy `OPENAI_API_KEY` path may remain but is not part of DoD
- Out of scope: Chroma, FastAPI, local LLM generation, wiring project 02, HF Hub upload, required hybrid RRF

## File Structure

| File | Role |
|------|------|
| `01-rag-cardapio/src/indexes.py` | `TfidfIndex`, `EmbeddingIndex`, shared helpers |
| `01-rag-cardapio/src/rag.py` | `Hit`, `FaqRag(mode=...)`, generate, export, normalize |
| `01-rag-cardapio/src/eval.py` | Golden-set metrics CLI |
| `01-rag-cardapio/src/app.py` | Gradio UI with mode dropdown |
| `01-rag-cardapio/data/golden_qa.jsonl` | Eval queries + relevant doc ids |
| `01-rag-cardapio/tests/test_tfidf_index.py` | Unit tests TF-IDF index |
| `01-rag-cardapio/tests/test_embedding_index.py` | Unit tests embedding index (injectable encode) |
| `01-rag-cardapio/tests/test_faq_rag_mode.py` | Mode routing, thresholds, fallback |
| `01-rag-cardapio/tests/test_eval_metrics.py` | Pure metric helpers |
| `01-rag-cardapio/requirements.txt` | Add deps + pytest |
| `01-rag-cardapio/README.md` | Run / test / eval table |
| `scripts/smoke_test.sh` | Both retrieval modes |
| `.gitignore` | `data/index/`, `output/eval_*.md` |

---

### Task 1: Dependencies, gitignore, pytest scaffold

**Files:**
- Modify: `01-rag-cardapio/requirements.txt`
- Modify: `.gitignore`
- Create: `01-rag-cardapio/tests/__init__.py` (empty)
- Create: `01-rag-cardapio/pytest.ini`

**Interfaces:**
- Consumes: none
- Produces: installable deps; pytest discovers `01-rag-cardapio/tests`

- [ ] **Step 1: Update requirements**

Replace `01-rag-cardapio/requirements.txt` with:

```text
gradio>=4.44.0
scikit-learn>=1.3.0
numpy>=1.24.0
python-dotenv>=1.0.0
sentence-transformers>=3.0.0
faiss-cpu>=1.7.4
pytest>=8.0.0
```

- [ ] **Step 2: Update gitignore**

Append to `.gitignore`:

```text
01-rag-cardapio/data/index/
01-rag-cardapio/output/
```

- [ ] **Step 3: Add pytest.ini**

Create `01-rag-cardapio/pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = src
```

- [ ] **Step 4: Create empty tests package**

Create empty file `01-rag-cardapio/tests/__init__.py`.

- [ ] **Step 5: Install and verify pytest runs (0 tests OK)**

Run:

```bash
cd 01-rag-cardapio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Expected: `no tests ran` or `0 passed` (exit 0). First install may take several minutes (torch/sentence-transformers).

- [ ] **Step 6: Commit**

```bash
git add 01-rag-cardapio/requirements.txt 01-rag-cardapio/pytest.ini 01-rag-cardapio/tests/__init__.py .gitignore
git commit -m "$(cat <<'EOF'
chore(rag): add embedding deps and pytest scaffold

EOF
)"
```

---

### Task 2: Extract `TfidfIndex` with shared `Hit` usage

**Files:**
- Create: `01-rag-cardapio/src/indexes.py`
- Modify: `01-rag-cardapio/src/rag.py`
- Create: `01-rag-cardapio/tests/test_tfidf_index.py`

**Interfaces:**
- Consumes: FAQ docs `list[dict]` with keys `id`, `categoria`, `pergunta`, `resposta`
- Produces:
  - `Hit` stays in `rag.py` (imported by indexes)
  - `class TfidfIndex:`
    - `def __init__(self, docs: list[dict]) -> None`
    - `def retrieve(self, query: str, k: int = 3) -> list[Hit]`
    - `def scores_for_query(self, query: str) -> Any`  # 1D array aligned to docs (for intent)

- [ ] **Step 1: Write failing test**

Create `01-rag-cardapio/tests/test_tfidf_index.py`:

```python
from indexes import TfidfIndex
from rag import Hit

DOCS = [
    {
        "id": "faq-009",
        "categoria": "pagamento",
        "pergunta": "Fui cobrado duas vezes",
        "resposta": "Abra o pedido e fale com o suporte para estorno.",
    },
    {
        "id": "faq-001",
        "categoria": "entrega",
        "pergunta": "Quanto tempo demora a entrega?",
        "resposta": "Em geral 20 a 50 minutos.",
    },
]


def test_tfidf_retrieve_returns_hits_sorted():
    idx = TfidfIndex(DOCS)
    hits = idx.retrieve("cobraram duas vezes no cartão", k=2)
    assert len(hits) == 2
    assert all(isinstance(h, Hit) for h in hits)
    assert hits[0].id == "faq-009"
    assert hits[0].score >= hits[1].score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 01-rag-cardapio && source .venv/bin/activate && pytest tests/test_tfidf_index.py -v`

Expected: FAIL (`ModuleNotFoundError: indexes` or `TfidfIndex` missing).

- [ ] **Step 3: Implement `TfidfIndex` in `indexes.py`**

Create `01-rag-cardapio/src/indexes.py`:

```python
"""Retrieval indexes for FAQ RAG (TF-IDF and embeddings/FAISS)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag import Hit


def _corpus(docs: list[dict[str, Any]]) -> list[str]:
    return [f"{d['pergunta']} {d['resposta']} {d['categoria']}" for d in docs]


class TfidfIndex:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        if not docs:
            raise ValueError("FAQ docs list is empty")
        self.docs = docs
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            lowercase=True,
        )
        self.matrix = self.vectorizer.fit_transform(_corpus(docs))

    def scores_for_query(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        return cosine_similarity(q, self.matrix).ravel()

    def retrieve(self, query: str, k: int = 3) -> list[Hit]:
        scores = self.scores_for_query(query)
        idxs = scores.argsort()[::-1][:k]
        hits: list[Hit] = []
        for i in idxs:
            d = self.docs[int(i)]
            hits.append(
                Hit(
                    id=d["id"],
                    categoria=d["categoria"],
                    pergunta=d["pergunta"],
                    resposta=d["resposta"],
                    score=float(scores[int(i)]),
                )
            )
        return hits
```

- [ ] **Step 4: Refactor `FaqRag` to use `TfidfIndex` (behavior unchanged for default callers)**

In `01-rag-cardapio/src/rag.py`, keep `Hit` and `CATEGORIAS`. Change `FaqRag` so TF-IDF logic uses `TfidfIndex` (still no `mode` param yet — that is Task 4). Minimal change:

```python
from indexes import TfidfIndex

class FaqRag:
    def __init__(self, docs: list[dict[str, Any]] | None = None) -> None:
        self.docs = docs or self._load_docs()
        if not self.docs:
            raise ValueError("FAQ docs list is empty")
        self._tfidf = TfidfIndex(self.docs)

    def retrieve(self, query: str, k: int = 3) -> list[Hit]:
        return self._tfidf.retrieve(query, k=k)

    def classify_intent(self, query: str) -> tuple[str, float]:
        scores = self._tfidf.scores_for_query(query)
        best: dict[str, float] = {c: 0.0 for c in CATEGORIAS}
        for i, doc in enumerate(self.docs):
            cat = doc["categoria"]
            best[cat] = max(best[cat], float(scores[i]))
        categoria = max(best, key=best.get)
        return categoria, best[categoria]
```

Remove the old inline `vectorizer` / `matrix` fields. Keep `answer`, `_generate`, `export_hf_dataset`, `normalize_query` as they are.

**Circular import note:** `indexes.py` imports `Hit` from `rag.py`, and `rag.py` imports `TfidfIndex` from `indexes.py`. If Python raises circular import at runtime, move `Hit` (+ `CATEGORIAS` optional) into `01-rag-cardapio/src/models.py` and update both imports. Prefer fixing the cycle with `models.py` rather than lazy imports.

If cycle appears, do this instead (still in Task 2):

Create `01-rag-cardapio/src/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

CATEGORIAS = [
    "entrega",
    "pagamento",
    "pedido",
    "conta",
    "promocao",
    "restaurante",
    "cardapio",
]


@dataclass
class Hit:
    id: str
    categoria: str
    pergunta: str
    resposta: str
    score: float
```

Then `from models import Hit, CATEGORIAS` in both `indexes.py` and `rag.py`, and re-export `from models import Hit, CATEGORIAS` at bottom of `rag.py` if needed for `from rag import Hit`.

- [ ] **Step 5: Run tests**

Run: `cd 01-rag-cardapio && source .venv/bin/activate && pytest tests/test_tfidf_index.py -v`

Expected: PASS

Also quick smoke:

```bash
python -c "from rag import FaqRag; print(FaqRag().answer('cobraram duas vezes')['sources'][0]['id'])"
```

Expected: `faq-009`

- [ ] **Step 6: Commit**

```bash
git add 01-rag-cardapio/src/indexes.py 01-rag-cardapio/src/rag.py 01-rag-cardapio/src/models.py 01-rag-cardapio/tests/test_tfidf_index.py
git commit -m "$(cat <<'EOF'
refactor(rag): extract TfidfIndex behind shared Hit retrieve API

EOF
)"
```

---

### Task 3: `EmbeddingIndex` (FAISS + injectable encode)

**Files:**
- Modify: `01-rag-cardapio/src/indexes.py`
- Create: `01-rag-cardapio/tests/test_embedding_index.py`

**Interfaces:**
- Produces:
  - `DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`
  - `class EmbeddingIndex:`
    - `def __init__(self, docs, *, index_dir: Path, model_name: str = DEFAULT_MODEL, encode_fn=None) -> None`
    - `def retrieve(self, query: str, k: int = 3) -> list[Hit]`
    - `def scores_for_query(self, query: str) -> np.ndarray`
    - persists `index_dir / "faiss.index"` and `index_dir / "meta.json"`

`encode_fn: Callable[[list[str]], np.ndarray] | None` — when provided, skip `sentence-transformers` (unit tests). When `None`, load SentenceTransformer and encode with `normalize_embeddings=True`. FAISS uses `IndexFlatIP` (inner product on L2-normalized vectors = cosine).

- [ ] **Step 1: Write failing tests**

Create `01-rag-cardapio/tests/test_embedding_index.py`:

```python
from pathlib import Path

import numpy as np

from indexes import EmbeddingIndex
from rag import Hit

DOCS = [
    {
        "id": "faq-009",
        "categoria": "pagamento",
        "pergunta": "Fui cobrado duas vezes",
        "resposta": "Peça estorno no suporte.",
    },
    {
        "id": "faq-001",
        "categoria": "entrega",
        "pergunta": "Quanto tempo demora a entrega?",
        "resposta": "20 a 50 minutos.",
    },
]


def _fake_encode(texts: list[str]) -> np.ndarray:
    """Deterministic 8-d unit vectors from bag-of-char hashes (test only)."""
    vecs = []
    for t in texts:
        v = np.zeros(8, dtype=np.float32)
        for i, ch in enumerate(t.lower()):
            v[i % 8] += 1.0 + (ord(ch) % 5) * 0.01
        n = np.linalg.norm(v) or 1.0
        vecs.append(v / n)
    return np.stack(vecs).astype(np.float32)


def test_embedding_retrieve_and_persist(tmp_path: Path):
    idx = EmbeddingIndex(DOCS, index_dir=tmp_path, encode_fn=_fake_encode)
    hits = idx.retrieve("cobrado duas vezes", k=1)
    assert isinstance(hits[0], Hit)
    assert hits[0].id == "faq-009"

    # reload from disk with same encode_fn
    idx2 = EmbeddingIndex(DOCS, index_dir=tmp_path, encode_fn=_fake_encode)
    hits2 = idx2.retrieve("cobrado duas vezes", k=1)
    assert hits2[0].id == "faq-009"
    assert (tmp_path / "faiss.index").exists()
    assert (tmp_path / "meta.json").exists()
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_embedding_index.py -v`

Expected: FAIL (`EmbeddingIndex` missing).

- [ ] **Step 3: Implement `EmbeddingIndex`**

Append to `01-rag-cardapio/src/indexes.py`:

```python
import json
from collections.abc import Callable
from pathlib import Path

import faiss

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EncodeFn = Callable[[list[str]], np.ndarray]


class EmbeddingIndex:
    def __init__(
        self,
        docs: list[dict[str, Any]],
        *,
        index_dir: Path,
        model_name: str = DEFAULT_MODEL,
        encode_fn: EncodeFn | None = None,
    ) -> None:
        if not docs:
            raise ValueError("FAQ docs list is empty")
        self.docs = docs
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self._encode_fn = encode_fn
        self._model = None
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss_path = self.index_dir / "faiss.index"
        meta_path = self.index_dir / "meta.json"
        if faiss_path.exists() and meta_path.exists():
            self.index = faiss.read_index(str(faiss_path))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("model_name") != model_name or meta.get("doc_ids") != [d["id"] for d in docs]:
                self._rebuild()
        else:
            self._rebuild()

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encode_fn is not None:
            vectors = self._encode_fn(texts)
        else:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("encode_fn must return 2D array")
        return arr

    def _rebuild(self) -> None:
        vectors = self._encode(_corpus(self.docs))
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        meta = {
            "model_name": self.model_name,
            "doc_ids": [d["id"] for d in self.docs],
            "dim": int(vectors.shape[1]),
        }
        (self.index_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def scores_for_query(self, query: str) -> np.ndarray:
        q = self._encode([query])
        scores, _ = self.index.search(q, len(self.docs))
        # Map FAISS order back to doc-aligned scores
        # Prefer search returning idxs:
        scores_q, idxs = self.index.search(q, len(self.docs))
        out = np.zeros(len(self.docs), dtype=np.float32)
        for score, idx in zip(scores_q[0], idxs[0], strict=True):
            if int(idx) >= 0:
                out[int(idx)] = float(score)
        return out

    def retrieve(self, query: str, k: int = 3) -> list[Hit]:
        q = self._encode([query])
        scores, idxs = self.index.search(q, min(k, len(self.docs)))
        hits: list[Hit] = []
        for score, idx in zip(scores[0], idxs[0], strict=True):
            if int(idx) < 0:
                continue
            d = self.docs[int(idx)]
            hits.append(
                Hit(
                    id=d["id"],
                    categoria=d["categoria"],
                    pergunta=d["pergunta"],
                    resposta=d["resposta"],
                    score=float(score),
                )
            )
        return hits
```

Fix `scores_for_query` to remove the duplicate unused `search` call (keep only the version that fills `out`).

- [ ] **Step 4: Run unit tests**

Run: `pytest tests/test_embedding_index.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add 01-rag-cardapio/src/indexes.py 01-rag-cardapio/tests/test_embedding_index.py
git commit -m "$(cat <<'EOF'
feat(rag): add EmbeddingIndex with FAISS persistence

EOF
)"
```

---

### Task 4: Wire `FaqRag(mode=...)` + thresholds + fallback

**Files:**
- Modify: `01-rag-cardapio/src/rag.py`
- Create: `01-rag-cardapio/tests/test_faq_rag_mode.py`

**Interfaces:**
- Produces:
  - `FaqRag.__init__(self, docs=None, *, mode: str = "embeddings", index_dir: Path | None = None, encode_fn=None)`
  - `mode` in `{"embeddings", "tfidf"}`
  - on embeddings init failure → set `self.mode = "tfidf"`, `self.fallback_warning: str | None`
  - `answer(...)` includes `"mode"` and `"warning"` keys
  - grounded: tfidf `score >= 0.08`; embeddings `score >= 0.25`

- [ ] **Step 1: Write failing tests**

Create `01-rag-cardapio/tests/test_faq_rag_mode.py`:

```python
from pathlib import Path

import numpy as np

from rag import FaqRag

DOCS = [
    {
        "id": "faq-009",
        "categoria": "pagamento",
        "pergunta": "Fui cobrado duas vezes",
        "resposta": "Peça estorno no suporte do pedido.",
    },
    {
        "id": "faq-001",
        "categoria": "entrega",
        "pergunta": "Quanto tempo demora a entrega?",
        "resposta": "20 a 50 minutos.",
    },
]


def _fake_encode(texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        v = np.zeros(8, dtype=np.float32)
        for i, ch in enumerate(t.lower()):
            v[i % 8] += 1.0 + (ord(ch) % 5) * 0.01
        n = np.linalg.norm(v) or 1.0
        vecs.append(v / n)
    return np.stack(vecs).astype(np.float32)


def test_mode_tfidf_answer_shape():
    rag = FaqRag(DOCS, mode="tfidf")
    out = rag.answer("cobraram duas vezes")
    assert out["mode"] == "tfidf"
    assert "sources" in out
    assert out["warning"] is None


def test_mode_embeddings_with_fake_encode(tmp_path: Path):
    rag = FaqRag(
        DOCS,
        mode="embeddings",
        index_dir=tmp_path,
        encode_fn=_fake_encode,
    )
    out = rag.answer("cobrado duas vezes")
    assert out["mode"] == "embeddings"
    assert out["sources"][0]["id"] == "faq-009"


def test_embeddings_init_fallback_to_tfidf(tmp_path: Path):
    def boom(_texts: list[str]) -> np.ndarray:
        raise RuntimeError("encoder down")

    rag = FaqRag(
        DOCS,
        mode="embeddings",
        index_dir=tmp_path,
        encode_fn=boom,
    )
    assert rag.mode == "tfidf"
    assert rag.fallback_warning
    out = rag.answer("cobraram duas vezes")
    assert out["mode"] == "tfidf"
    assert out["warning"]
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_faq_rag_mode.py -v`

Expected: FAIL (`mode` / unexpected kwargs).

- [ ] **Step 3: Implement mode wiring in `FaqRag`**

Update `FaqRag.__init__` and `answer` / `retrieve` / `classify_intent` roughly as:

```python
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "faq.json"
DEFAULT_INDEX_DIR = ROOT / "data" / "index"
TFIDF_GROUNDED_MIN = 0.08
EMBEDDING_GROUNDED_MIN = 0.25

class FaqRag:
    def __init__(
        self,
        docs: list[dict[str, Any]] | None = None,
        *,
        mode: str = "embeddings",
        index_dir: Path | None = None,
        encode_fn=None,
    ) -> None:
        self.docs = docs or self._load_docs()
        if not self.docs:
            raise ValueError("FAQ docs list is empty")
        if mode not in {"embeddings", "tfidf"}:
            raise ValueError(f"unsupported mode: {mode}")
        self.mode = mode
        self.fallback_warning: str | None = None
        self._tfidf = TfidfIndex(self.docs)
        self._embeddings = None
        if mode == "embeddings":
            try:
                from indexes import EmbeddingIndex

                self._embeddings = EmbeddingIndex(
                    self.docs,
                    index_dir=index_dir or DEFAULT_INDEX_DIR,
                    encode_fn=encode_fn,
                )
            except Exception as exc:  # noqa: BLE001 — demo fallback
                self.mode = "tfidf"
                self.fallback_warning = (
                    f"Embeddings indisponível ({exc}); usando TF-IDF."
                )

    def _active_index(self):
        if self.mode == "embeddings" and self._embeddings is not None:
            return self._embeddings
        return self._tfidf

    def retrieve(self, query: str, k: int = 3) -> list[Hit]:
        return self._active_index().retrieve(query, k=k)

    def classify_intent(self, query: str) -> tuple[str, float]:
        scores = self._active_index().scores_for_query(query)
        best: dict[str, float] = {c: 0.0 for c in CATEGORIAS}
        for i, doc in enumerate(self.docs):
            cat = doc["categoria"]
            best[cat] = max(best[cat], float(scores[i]))
        categoria = max(best, key=best.get)
        return categoria, best[categoria]

    def answer(self, query: str, k: int = 3) -> dict[str, Any]:
        hits = self.retrieve(query, k=k)
        intent, intent_score = self.classify_intent(query)
        top = hits[0] if hits else None
        min_score = (
            EMBEDDING_GROUNDED_MIN if self.mode == "embeddings" else TFIDF_GROUNDED_MIN
        )
        grounded = top is not None and top.score >= min_score
        if not grounded:
            texto = (
                "Não encontrei uma resposta confiante na base de FAQ. "
                "Reformule a pergunta ou fale com o suporte humano."
            )
        else:
            texto = self._generate(query, hits)
        return {
            "query": query,
            "mode": self.mode,
            "warning": self.fallback_warning,
            "intent": intent,
            "intent_score": round(intent_score, 4),
            "grounded": grounded,
            "answer": texto,
            "sources": [
                {
                    "id": h.id,
                    "categoria": h.categoria,
                    "pergunta": h.pergunta,
                    "score": round(h.score, 4),
                }
                for h in hits
            ],
        }
```

Keep `_generate` / `_openai_generate` unchanged.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_faq_rag_mode.py tests/test_tfidf_index.py tests/test_embedding_index.py -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add 01-rag-cardapio/src/rag.py 01-rag-cardapio/tests/test_faq_rag_mode.py
git commit -m "$(cat <<'EOF'
feat(rag): add embeddings/tfidf modes with grounded thresholds

EOF
)"
```

---

### Task 5: Golden set + eval metrics + CLI

**Files:**
- Create: `01-rag-cardapio/data/golden_qa.jsonl`
- Create: `01-rag-cardapio/src/eval.py`
- Create: `01-rag-cardapio/tests/test_eval_metrics.py`

**Interfaces:**
- Produces:
  - `recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float`  # 0 or 1 per query
  - `mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float`
  - `citation_match(source_id: str | None, relevant_ids: list[str], grounded: bool) -> float`
  - CLI writes `output/eval_latest.md` and prints a markdown table

- [ ] **Step 1: Write metric unit tests**

Create `01-rag-cardapio/tests/test_eval_metrics.py`:

```python
from eval import citation_match, mrr, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["c"], k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["z"], k=3) == 0.0
    assert recall_at_k(["a", "b"], ["b"], k=1) == 0.0


def test_mrr():
    assert mrr(["a", "b", "c"], ["c"]) == 1.0 / 3
    assert mrr(["c", "a"], ["c"]) == 1.0
    assert mrr(["a", "b"], ["z"]) == 0.0


def test_citation_match():
    assert citation_match("faq-009", ["faq-009"], True) == 1.0
    assert citation_match("faq-001", ["faq-009"], True) == 0.0
    assert citation_match(None, ["faq-009"], False) == 0.0
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_eval_metrics.py -v`

Expected: FAIL (`eval` missing).

- [ ] **Step 3: Implement metric helpers + CLI in `eval.py`**

Create `01-rag-cardapio/src/eval.py`:

```python
"""Offline golden-set eval: embeddings vs tfidf."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag import FaqRag

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "golden_qa.jsonl"
OUT_DIR = ROOT / "output"
OUT_MD = OUT_DIR / "eval_latest.md"


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return 1.0 if any(r in relevant for r in top) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def citation_match(
    source_id: str | None, relevant_ids: list[str], grounded: bool
) -> float:
    if not grounded or not source_id:
        return 0.0
    return 1.0 if source_id in set(relevant_ids) else 0.0


def load_golden(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_mode(mode: str, rows: list[dict[str, Any]], k: int = 3) -> dict[str, float]:
    rag = FaqRag(mode=mode)
    recalls: list[float] = []
    mrrs: list[float] = []
    cites: list[float] = []
    for row in rows:
        result = rag.answer(row["query"], k=k)
        retrieved = [s["id"] for s in result["sources"]]
        relevant = row["relevant_doc_ids"]
        recalls.append(recall_at_k(retrieved, relevant, k))
        mrrs.append(mrr(retrieved, relevant))
        top_id = retrieved[0] if retrieved else None
        cites.append(citation_match(top_id, relevant, bool(result["grounded"])))
    n = max(len(rows), 1)
    return {
        "recall_at_3": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "citation_match": sum(cites) / n,
    }


def render_table(results: dict[str, dict[str, float]]) -> str:
    lines = [
        "| mode | Recall@3 | MRR | citation match |",
        "|------|----------|-----|----------------|",
    ]
    for mode, metrics in results.items():
        lines.append(
            f"| {mode} | {metrics['recall_at_3']:.3f} | "
            f"{metrics['mrr']:.3f} | {metrics['citation_match']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    rows = load_golden()
    results = {
        "embeddings": evaluate_mode("embeddings", rows),
        "tfidf": evaluate_mode("tfidf", rows),
    }
    table = render_table(results)
    print(table)
    emb = results["embeddings"]["recall_at_3"]
    tf = results["tfidf"]["recall_at_3"]
    if emb < tf:
        print(
            "\nWARNING: embeddings Recall@3 < tfidf. "
            "Inspect golden paraphrases / model before publishing README numbers."
        )
    else:
        print("\nOK: embeddings Recall@3 >= tfidf")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(
        "# RAG eval\n\n" + table + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `data/golden_qa.jsonl` (≥25 paraphrases)**

Create `01-rag-cardapio/data/golden_qa.jsonl` with **at least 25** lines. Use paraphrases (not exact FAQ questions). Example rows (expand to full 25+ covering all 12 FAQ ids, ~2–3 each):

```jsonl
{"query": "demora quanto a entrega?", "relevant_doc_ids": ["faq-001"], "notes": "paraphrase entrega"}
{"query": "em quanto tempo o pedido chega?", "relevant_doc_ids": ["faq-001"]}
{"query": "quais pagamentos vocês aceitam?", "relevant_doc_ids": ["faq-002"]}
{"query": "dá pra pagar com pix?", "relevant_doc_ids": ["faq-002"]}
{"query": "como faço pra cancelar o pedido?", "relevant_doc_ids": ["faq-003"]}
{"query": "quero cancelar o que pedi", "relevant_doc_ids": ["faq-003"]}
{"query": "pedido veio errado com item faltando", "relevant_doc_ids": ["faq-004"]}
{"query": "mandaram o lanche errado", "relevant_doc_ids": ["faq-004"]}
{"query": "motoboy não acha minha casa", "relevant_doc_ids": ["faq-005"]}
{"query": "entregador não encontra o endereço", "relevant_doc_ids": ["faq-005"]}
{"query": "trocar email da conta", "relevant_doc_ids": ["faq-006"]}
{"query": "como atualizo o e-mail?", "relevant_doc_ids": ["faq-006"]}
{"query": "cupom não aplica no pedido", "relevant_doc_ids": ["faq-007"]}
{"query": "meu cupom não funciona", "relevant_doc_ids": ["faq-007"]}
{"query": "restaurante aparece aberto mas está fechado", "relevant_doc_ids": ["faq-008"]}
{"query": "loja fechada no app", "relevant_doc_ids": ["faq-008"]}
{"query": "cobraram duas vezes no cartão", "relevant_doc_ids": ["faq-009"]}
{"query": "fui taxado em duplicidade", "relevant_doc_ids": ["faq-009"]}
{"query": "posso mudar o endereço depois de pedir?", "relevant_doc_ids": ["faq-010"]}
{"query": "alterar endereço do pedido já feito", "relevant_doc_ids": ["faq-010"]}
{"query": "tem opção vegana no cardápio?", "relevant_doc_ids": ["faq-011"]}
{"query": "comida sem gluten", "relevant_doc_ids": ["faq-011"]}
{"query": "como ver a taxa de entrega?", "relevant_doc_ids": ["faq-012"]}
{"query": "quanto custa a frete?", "relevant_doc_ids": ["faq-012"]}
{"query": "estorno de cobrança duplicada", "relevant_doc_ids": ["faq-009"]}
{"query": "prazo médio de entrega urbana", "relevant_doc_ids": ["faq-001"]}
{"query": "cancelamento após restaurante aceitar", "relevant_doc_ids": ["faq-003"]}
```

(Ensure final file has ≥25 valid JSONL rows.)

- [ ] **Step 5: Run metric tests**

Run: `pytest tests/test_eval_metrics.py -v`

Expected: PASS

- [ ] **Step 6: Run full eval CLI (real model — first run slow)**

Run:

```bash
cd 01-rag-cardapio && source .venv/bin/activate
python src/eval.py
```

Expected: markdown table printed; `output/eval_latest.md` written; line `OK: embeddings Recall@3 >= tfidf` (if not, tighten paraphrases / check ids before continuing — do not invent metrics in README).

- [ ] **Step 7: Commit**

```bash
git add 01-rag-cardapio/src/eval.py 01-rag-cardapio/data/golden_qa.jsonl 01-rag-cardapio/tests/test_eval_metrics.py
git commit -m "$(cat <<'EOF'
feat(rag): add golden-set eval for embeddings vs tfidf

EOF
)"
```

---

### Task 6: Gradio mode dropdown + warnings

**Files:**
- Modify: `01-rag-cardapio/src/app.py`

**Interfaces:**
- Consumes: `FaqRag(mode=...).answer`
- Produces: UI with Radio/Dropdown `embeddings | tfidf`; shows mode + warning in meta

- [ ] **Step 1: Update `perguntar` and UI**

Replace mode-unaware global `rag = FaqRag()` with lazy instances or construct per call:

```python
"""Demo Gradio — RAG de FAQ delivery (TF-IDF vs embeddings offline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag import FaqRag, export_hf_dataset  # noqa: E402

export_hf_dataset()

_rag_cache: dict[str, FaqRag] = {}


def get_rag(mode: str) -> FaqRag:
    if mode not in _rag_cache:
        _rag_cache[mode] = FaqRag(mode=mode)
    return _rag_cache[mode]


EXAMPLES = [
    "demora quanto a entrega?",
    "cobraram duas vezes no cartão",
    "cupom não aplica no pedido",
    "pedido veio errado com item faltando",
    "tem opção vegana no cardápio?",
]


def perguntar(query: str, mode: str, k: int = 3) -> tuple[str, str, str]:
    if not query or not query.strip():
        return "Digite uma pergunta.", "", ""
    try:
        result = get_rag(mode).answer(query.strip(), k=int(k))
    except Exception as exc:  # noqa: BLE001 — no stack trace in UI
        return f"Erro ao responder: {exc}", "", ""
    warn = result.get("warning")
    warn_bit = f" | ⚠️ {warn}" if warn else ""
    answer = result["answer"]
    meta = (
        f"Modo: **{result['mode']}** | "
        f"Intenção: **{result['intent']}** "
        f"(score {result['intent_score']}) | "
        f"Grounded: **{result['grounded']}**"
        f"{warn_bit}"
    )
    sources = json.dumps(result["sources"], ensure_ascii=False, indent=2)
    return answer, meta, sources


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG Cardápio/FAQ — GenAI Portfolio") as demo:
        gr.Markdown(
            """
            # RAG Cardápio / FAQ Delivery
            Compare **embeddings (FAISS)** vs **TF-IDF**, com resposta grounded e fontes.

            100% offline (sem API key). Primeira carga de embeddings pode levar 1–3 min.
            Dataset: `data/faq_hf.jsonl`. Eval: `python src/eval.py`.
            """
        )
        with gr.Row():
            query = gr.Textbox(label="Pergunta do cliente", lines=2, scale=4)
            mode = gr.Radio(
                choices=["embeddings", "tfidf"],
                value="embeddings",
                label="Modo de retrieval",
            )
            k = gr.Slider(1, 5, value=3, step=1, label="Top-K fontes")
        btn = gr.Button("Responder", variant="primary")
        answer = gr.Markdown(label="Resposta")
        meta = gr.Markdown()
        sources = gr.Code(label="Fontes recuperadas (JSON)", language="json")
        gr.Examples(EXAMPLES, inputs=query)
        btn.click(perguntar, inputs=[query, mode, k], outputs=[answer, meta, sources])
        query.submit(perguntar, inputs=[query, mode, k], outputs=[answer, meta, sources])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
```

- [ ] **Step 2: Manual smoke (no automated UI test required)**

Run: `python src/app.py` → open `http://127.0.0.1:7860` → same question in both modes → sources JSON updates; empty query shows validation message.

- [ ] **Step 3: Commit**

```bash
git add 01-rag-cardapio/src/app.py
git commit -m "$(cat <<'EOF'
feat(rag): expose embeddings vs tfidf mode in Gradio

EOF
)"
```

---

### Task 7: README + monorepo smoke + DoD check

**Files:**
- Modify: `01-rag-cardapio/README.md`
- Modify: `scripts/smoke_test.sh`

- [ ] **Step 1: Update smoke test for both modes**

Replace the Projeto 01 block in `scripts/smoke_test.sh` with:

```bash
echo "==> Projeto 01 RAG"
python3 - <<'PY'
import sys
sys.path.insert(0, "01-rag-cardapio/src")
from rag import FaqRag, export_hf_dataset

r_tf = FaqRag(mode="tfidf").answer("cobraram duas vezes no cartão")
assert r_tf["grounded"], r_tf
assert r_tf["mode"] == "tfidf"

r_emb = FaqRag(mode="embeddings").answer("cobraram duas vezes no cartão")
assert r_emb["mode"] in {"embeddings", "tfidf"}  # tfidf only if fallback
assert r_emb["sources"], r_emb
export_hf_dataset()
print("OK RAG tfidf:", r_tf["intent"], r_tf["sources"][0]["id"])
print("OK RAG embeddings:", r_emb["mode"], r_emb["sources"][0]["id"])
PY
```

- [ ] **Step 2: Update README**

Rewrite `01-rag-cardapio/README.md` to include:

- What it is (embeddings + FAISS vs TF-IDF, offline)
- How to run Gradio
- How to run `python src/eval.py`
- Paste **real** numbers from `output/eval_latest.md` (never invent)
- Note first embeddings download time
- Point to golden set path

- [ ] **Step 3: Run full verification**

```bash
cd 01-rag-cardapio && source .venv/bin/activate
pytest -q
python src/eval.py
cd .. && ./scripts/smoke_test.sh
```

Expected: pytest green; eval OK line; smoke all projects pass.

- [ ] **Step 4: Commit**

```bash
git add 01-rag-cardapio/README.md scripts/smoke_test.sh
git commit -m "$(cat <<'EOF'
docs(rag): document offline eval and dual-mode smoke tests

EOF
)"
```

- [ ] **Step 5: DoD checklist (mark done only with evidence)**

1. Both modes in CLI and Gradio  
2. FAISS built under `data/index/` on first embeddings run  
3. `eval.py` reproducible + README table with real metrics  
4. No API key required  
5. README updated  
6. Non-goals untouched  

---

## Plan self-review

| Spec requirement | Task |
|------------------|------|
| Embeddings + FAISS local | Task 3–4 |
| TF-IDF baseline / mode toggle | Task 2, 4, 6 |
| Golden eval Recall@3, MRR, citation | Task 5 |
| Gradio UX + fallback warning | Task 4, 6 |
| Offline / no API key DoD | Global + Task 7 |
| README + smoke | Task 7 |
| Non-goals excluded | Global Constraints |
| Stretch hybrid / heavy pytest optional | Not scheduled (YAGNI) |

No TBD placeholders. Names consistent: `TfidfIndex`, `EmbeddingIndex`, `FaqRag(mode=...)`, `recall_at_k`, `mrr`, `citation_match`.
