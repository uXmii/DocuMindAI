<div align="center">

# 🧠 DocuMind AI
### Production-Grade Multimodal RAG System

**Hybrid Search · Vision AI · Agentic Intelligence · LLM Evaluation**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square)](https://reactjs.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-green?style=flat-square)](https://langchain-ai.github.io/langgraph)
[![Gemini](https://img.shields.io/badge/Gemini-Free_Tier-orange?style=flat-square)](https://aistudio.google.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

[Features](#-features) · [Architecture](#-architecture) · [Setup](#-quick-setup) · [How It Works](#-how-it-works) · [API](#-api-reference)

</div>

---

## ✨ Features

- 🔮 **True Multimodal RAG** — Gemini Vision understands charts, diagrams, tables, and images in PDFs — not just text
- ⚡ **Hybrid Search** — Combines semantic vector search (all-MiniLM), BM25 keyword search, and CLIP image embeddings via Reciprocal Rank Fusion
- 🤖 **Agentic Intelligence** — LangGraph agent analyzes each question, evaluates all search modes, selects the best answer, and self-refines if quality is low
- 📊 **LLM-as-Judge Evaluation** — Gemini scores answer faithfulness and completeness without needing ground truth
- 🌐 **External Factual Check** — Independent verification of answers against general knowledge
- 🔁 **Consistency Testing** — Measures how stable answers are across repeated runs
- 💡 **Winner Reasoning** — Explains *why* a particular search method won for each question type

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DocuMind AI                              │
├──────────────┬──────────────────────────────────────────────────┤
│   Frontend   │  React 18 · Geist fonts · Light theme            │
│  (port 3000) │  Chat UI · Evaluation modals · Comparison view   │
├──────────────┴──────────────────────────────────────────────────┤
│                    Flask API  (port 5000)                        │
├─────────────┬───────────────────┬───────────────────────────────┤
│  Document   │   RAG Engine      │   Agentic RAG                 │
│  Processor  │                   │   (LangGraph)                 │
│             │  ┌─────────────┐  │                               │
│ PyPDF2      │  │ ChromaDB    │  │  analyze_query                │
│  + Gemini   │  │ (vectors)   │  │    → evaluate_all_modes       │
│    Vision   │  ├─────────────┤  │    → select_best_mode         │
│  + Camelot  │  │ BM25 Index  │  │    → evaluate_quality         │
│    Tables   │  ├─────────────┤  │    → finalize_answer          │
│  + CLIP     │  │ CLIP Index  │  │                               │
│    Embeds   │  │ (512-dim)   │  │  Shared RAGEvaluator          │
│             │  └─────────────┘  │  (agent + dashboard agree)    │
├─────────────┴───────────────────┴───────────────────────────────┤
│                    Evaluation Layer                              │
│  Embedding similarity · LLM Judge · Factual Check · Consistency │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- [Gemini API key](https://aistudio.google.com/app/apikey) (free)
- Windows: [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) for vision processing

### 1. Clone
```bash
git clone https://github.com/uXmii/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Backend
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment variables
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 4. Frontend
```bash
cd frontend
npm install
```

### 5. Run
```bash
# Terminal 1 — Backend
cd backend
python app.py

# Terminal 2 — Frontend
cd frontend
npm start
```

Open **http://localhost:3000** and upload a PDF to get started.

---

## 🔍 How It Works

### 📄 Document Processing Pipeline

When you upload a PDF, it goes through three parallel processing tracks:

```
PDF Upload
    │
    ├── 1. TEXT EXTRACTION (PyPDF2)
    │       Sentence-aware chunking (1000 chars, 200 overlap)
    │       → text chunks with page metadata
    │
    ├── 2. VISION PROCESSING (Gemini Vision)
    │       Every page → image → Gemini describes it
    │       Understands: charts, diagrams, tables, equations
    │       → vision chunks with rich semantic descriptions
    │       → CLIP image embeddings (512-dim) stored separately
    │
    └── 3. TABLE EXTRACTION (Camelot)
            Lattice & stream detection
            → structured table chunks
```

**Why this matters:** A standard RAG system using PyPDF2 alone would completely miss the flowchart on page 4, the GDP timeline chart, and the Hundi comparison table. Vision processing makes these fully searchable.

---

### 🔎 Hybrid Search (3-way Fusion)

Every query runs three searches simultaneously:

| Search Type | How it works | Best for |
|-------------|-------------|---------|
| **Semantic (Vector)** | `all-MiniLM-L6-v2` embeddings, cosine similarity in ChromaDB | Conceptual questions, paraphrased queries |
| **Keyword (BM25)** | Okapi BM25 statistical ranking, exact term matching | Specific names, dates, precise facts |
| **CLIP Visual** | Text query → CLIP text encoder → searches image embedding space | Questions about diagrams, charts, visual content |

Results are merged using **Reciprocal Rank Fusion (RRF)**:
```
score(doc) = Σ 1 / (k + rank_in_method)   where k=60
```

This rewards documents that appear highly ranked in multiple search modes.

---

### 🤖 Agentic RAG (LangGraph)

The agent follows a deterministic graph:

```
analyze_query
      │
      ▼
evaluate_all_modes  ←──────────────────┐
      │                                 │
      ▼                                 │
select_best_mode                        │
      │                                 │
      ▼                                 │
evaluate_quality ──[needs_refinement]──┘
      │
      [good_enough]
      │
      ▼
finalize_answer
```

**Step by step:**

1. **analyze_query** — Classifies question as factual/conceptual/comparative/complex, extracts key terms, recommends initial search mode

2. **evaluate_all_modes** — Runs all 3 search modes in parallel, generates answers for each using Gemini (or extractive fallback)

3. **select_best_mode** — Scores each answer using the RAGEvaluator (same formula as the dashboard so they always agree). Tie-breaks by answer relevance, then speed

4. **evaluate_quality** — Checks if the winner meets the quality threshold (0.82). If not and iterations remain, refines the query and retries

5. **finalize_answer** — Returns the best answer with full metadata: confidence, search mode used, agent reasoning, workflow path

**Key design:** The agent and the evaluation dashboard share a **single RAGEvaluator instance**, so the winner the agent picks and the winner shown in the dashboard are always consistent.

---

### 📊 Evaluation System

Four layers of evaluation, each answering a different question:

#### Layer 1: Embedding Metrics (always available, no API)
| Metric | What it measures |
|--------|-----------------|
| **Answer Relevance** | Cosine similarity between question and answer embeddings |
| **Context Precision** | % of retrieved chunks that are relevant (>0.3 threshold) |
| **Context Recall** | Coverage — did we retrieve enough relevant chunks? |
| **Faithfulness** | Similarity between answer and combined context |

#### Layer 2: LLM Judge (requires Gemini API)
Gemini reads the retrieved context and the answer, scores:
- **Faithfulness (0-1)** — Is every claim in the answer supported by the context?
- **Completeness (0-1)** — Does it fully address the question?
- **Reasoning** — One sentence explaining the score

The LLM judge score is blended with embedding metrics: `0.5×embedding + 0.3×faithfulness + 0.2×completeness`

#### Layer 3: External Factual Check (requires Gemini API)
Completely independent of the document — Gemini checks: *"Based on your general knowledge, is this answer correct?"*

Returns: `correct` / `partially_correct` / `incorrect` / `unverifiable`

This is the key metric your manager asked about — it tells you whether the system is giving factually right answers, not just answers that match the document.

#### Layer 4: Consistency Testing
Runs the same question 3 times, measures:
- Score variance (std dev)
- Answer-level similarity across runs
- Consistency score: `1 - normalised_std_dev`

> **Note:** Consistency will always show 1.000 when Gemini is unavailable (rate limited) because the extractive fallback is deterministic. Real variance appears when Gemini generates answers.

#### Winner Reasoning
After selecting a winner, the system explains *why* using:
- Question type classification (factual/conceptual/comparative/complex)
- Actual score margins between methods
- The strongest metric for the winning method
- LLM judge verdict

Example: *"Your question is **factual** in nature. Semantic search (Vector) excels at factual questions because it maps your question directly to the most relevant passage by meaning, not just keywords. It scored 85% overall — 8% ahead of HYBRID (77%). Its strongest metric was **Answer Relevance** at 85%."*

---

## 🔌 API Reference

### Upload
```http
POST /upload
Content-Type: multipart/form-data
Body: file=<pdf>

Response: {
  "success": true,
  "chunks_created": 119,
  "chunk_breakdown": {"text": 84, "ocr": 0, "tables": 35},
  "vision_used": false,
  "collection_size": 119
}
```

### Query
```http
POST /query
Body: {"question": "...", "search_mode": "hybrid", "top_k": 5}

Response: {
  "answer": "...",
  "sources": [...],
  "search_mode": "hybrid",
  "clip_used": false,
  "search_time": 0.045
}
```

### Agentic Query
```http
POST /query/agentic
Body: {"question": "...", "max_iterations": 2}

Response: {
  "answer": "...",
  "confidence": 0.88,
  "quality_score": 0.877,
  "metadata": {
    "final_search_mode": "vector",
    "iterations": 1,
    "agent_thoughts": [...],
    "workflow_path": ["analyze_query", "evaluate_all_modes", ...]
  }
}
```

### Evaluate
```http
POST /evaluate/single
Body: {"question": "..."}

Response: {
  "winner": {
    "overall": {"method": "vector", "score": 0.877, "reasoning": "..."},
    "fastest": {"method": "bm25", "time": 0.007},
    "most_relevant": {"method": "vector", "score": 0.877}
  },
  "factual_check": {
    "verdict": "correct",
    "factual_accuracy": 0.9,
    "external_context": "..."
  },
  "methods": {
    "vector": {"metrics": {...}, "llm_judge": {...}},
    "bm25":   {"metrics": {...}},
    "hybrid": {"metrics": {...}}
  }
}
```

### Consistency Test
```http
POST /evaluate/consistency
Body: {"question": "...", "n_runs": 3, "mode": "hybrid"}

Response: {
  "consistency_score": 0.98,
  "is_consistent": true,
  "std_score": 0.012,
  "scores": [0.87, 0.86, 0.88]
}
```

---

## 📁 Project Structure

```
DocuMind-AI/
├── backend/
│   ├── app.py                  # Flask API (agentic endpoints)
│   ├── rag_engine.py           # Core RAG: ChromaDB + BM25 + CLIP + Gemini
│   ├── multimodal_processor.py # PDF → Vision AI + Tables + CLIP embeddings
│   ├── document_processor.py   # Text chunking pipeline
│   ├── evaluation_metrics.py   # LLM Judge + Factual Check + Consistency
│   ├── agentic_rag.py          # LangGraph agent workflow
│   ├── agent_state.py          # TypedDict state definition
│   ├── agent_tools.py          # Query analyzer + Answer evaluator
│   ├── test_evaluation.py      # CLI batch evaluation runner
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.js              # Full React UI
│       └── index.js
├── .env.example                # Template — copy to .env
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | **Recommended** | Free at [aistudio.google.com](https://aistudio.google.com). Powers vision, generation, and LLM judge |
| `HF_API_KEY` | Optional | Hugging Face — fallback generation if Gemini unavailable |
| `ANTHROPIC_API_KEY` | Optional | Claude fallback |
| `OPENAI_API_KEY` | Optional | GPT-4o fallback |

**Gemini Free Tier Limits:**
- 15 requests/minute
- 1,500 requests/day
- Resets daily at midnight Pacific Time

---

## 🪟 Windows: Poppler Setup (for Vision Processing)

Vision processing requires Poppler to convert PDF pages to images:

1. Download from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
2. Extract to `C:\Program Files\poppler\`
3. Add `C:\Program Files\poppler\Library\bin` to your system PATH
4. Restart your terminal and the Flask server

Without Poppler, the system still works — it just uses text extraction only (no chart/diagram understanding).

---

## 🧩 Integrating Into Your Own Project

The core components are modular and can be used independently:

```python
# Use just the RAG engine
from rag_engine import RAGEngine
engine = RAGEngine(collection_name="my_docs")
engine.add_documents(chunks)
result = engine.query("your question", search_mode="hybrid")

# Use just the evaluator
from evaluation_metrics import RAGEvaluator
evaluator = RAGEvaluator(rag_engine=engine)
result = evaluator.evaluate_query("your question")
print(result["winner"]["overall"]["reasoning"])

# Use just the multimodal processor
from multimodal_processor import MultimodalProcessor
processor = MultimodalProcessor()
result = processor.process_multimodal_document("doc.pdf")
# result["ocr_chunks"] — vision-described page chunks
# result["table_chunks"] — structured table chunks

# Use the agent standalone
from agentic_rag import AgenticRAG
agent = AgenticRAG(rag_engine=engine, evaluator=evaluator)
result = agent.query("complex question requiring multi-step reasoning")
```

---

## 📚 Key Concepts to Learn From

| Concept | File | What to study |
|---------|------|--------------|
| RAG basics | `rag_engine.py` | `_vector_search`, `_bm25_search`, `_reciprocal_rank_fusion` |
| Multimodal embeddings | `multimodal_processor.py` | `VisionLLMClient`, `CLIPEncoder` |
| LangGraph agents | `agentic_rag.py` | `_build_graph`, node functions |
| LLM evaluation | `evaluation_metrics.py` | `LLMJudge`, `_determine_winner`, `_generate_winner_reasoning` |
| Query analysis | `agent_tools.py` | `QueryAnalyzer.analyze_query` |

---

## 🙏 Built With

- [ChromaDB](https://www.trychroma.com/) — Vector database
- [sentence-transformers](https://www.sbert.net/) — Embeddings + CLIP
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent orchestration
- [Google Gemini](https://ai.google.dev/) — Vision, generation, evaluation
- [Camelot](https://camelot-py.readthedocs.io/) — PDF table extraction
- [Flask](https://flask.palletsprojects.com/) — API server
- [React](https://reactjs.org/) — Frontend

---

<div align="center">
Made by <a href="https://github.com/uXmii">uXmii</a> · If this helped you, give it a ⭐
</div>
