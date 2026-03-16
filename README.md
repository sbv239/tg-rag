# TG-RAG

A RAG system that indexes Telegram channel posts and answers questions about them via a Telegram bot.

Built over 5 sprints as a portfolio project: ingestion pipeline, hybrid retrieval, reranking, user feedback collection, and evaluation.

---

## What it does

Users interact with a Telegram bot. They ask a question in natural language — the system finds the most relevant posts across 24 wine-focused channels (~38,000 posts total) and returns a grounded answer with links to the source posts.

Each answer includes inline source citations in the format `[channel](url)`. Only sources that Claude actually cited in its response are shown to the user.

---

## Architecture

```
Telegram Channels
      │
      ▼
telegram_loader.py   ←  Telethon, incremental (state per channel)
      │
      ▼
embedder.py          ←  Voyage AI voyage-4-lite, local embedding cache
      │
      ▼
ChromaDB             ←  Persistent vector store (cosine similarity)


Query (Telegram Bot)
      │
      ├──► Vector Search   (ChromaDB, top-20)  ─┐
      │                                          ├──► RRF Fusion → top-20
      └──► BM25 Search     (in-memory, top-20)  ─┘
                                                        │
                                                        ▼
                                               Voyage rerank-2 → top-5
                                                        │
                                                        ▼
                                                 Claude Sonnet
                                                        │
                                                        ▼
                                           Answer + cited sources
                                                        │
                                                        ▼
                                           Feedback buttons (1–5 ⭐)
                                                        │
                                                        ▼
                                                   SQLite DB
```

---

## Stack

| Layer | Tool |
|---|---|
| Telegram parsing | Telethon |
| Embeddings | Voyage AI `voyage-4-lite` |
| Vector DB | ChromaDB (local, persistent) |
| Keyword search | rank-bm25 (in-memory) |
| Reranker | Voyage AI `rerank-2` |
| LLM | Claude Sonnet (Anthropic) |
| Bot | python-telegram-bot 22.6 |
| Feedback storage | SQLite (WAL mode) |

---

## Retrieval pipeline

The system uses hybrid search to handle both semantic and exact-match queries.

**Why hybrid?** Wine content includes many proper nouns — producer names, appellations, grape varieties — where BM25 outperforms vector search. Vector search handles paraphrasing and conceptual queries. RRF fusion combines both without requiring score normalization.

**Step by step:**

1. Query preprocessing — lowercase, stopword removal, preserve `$` tickers
2. Vector search — embed query with `voyage-4-lite` (`input_type="query"`), retrieve top-20 from ChromaDB
3. BM25 search — whitespace tokenization, top-20 by score (zero-score results dropped)
4. RRF fusion — `score = Σ 1/(60 + rank)`, deduplication by URL, top-20 candidates
5. Reranking — Voyage `rerank-2` scores each (query, document) pair jointly, returns top-5
6. Generation — Claude receives top-5 posts as numbered context blocks with channel + date + URL headers

**Source filtering:** after generation, only URLs that appear in Claude's markdown links `[channel](url)` are shown to the user. This eliminates irrelevant context from the sources list and reflects the actual reasoning in the answer.

---

## Evaluation

User ratings (1–5 ⭐) are collected after each answered question and stored in SQLite.

```bash
# All-time stats
python -m src.evaluation.analyze_feedback

# Inspect low-rated responses
python -m src.evaluation.analyze_feedback --low-rated 10
```

Metrics reported: MUR (Mean User Rating), Satisfaction Rate (≥4⭐), Dissatisfaction Rate (≤2⭐), rating distribution, median and mean response time.

### Results: vector-only vs hybrid search

Measured on real user traffic before and after deploying hybrid search (BM25 + RRF + reranking).

| Metric | Vector only | Hybrid search | Δ |
|---|---|---|---|
| Ratings collected | 29 | 28 | — |
| MUR (mean rating) | 3.17 | **4.04** | +0.87 |
| Satisfaction Rate (≥4⭐) | 44.8% | **75.0%** | +30.2pp |
| Dissatisfaction Rate (≤2⭐) | 37.9% | **10.7%** | −27.2pp |

Hybrid search nearly doubled the satisfaction rate and reduced dissatisfied responses by 3.5×. The improvement is most pronounced on queries with proper nouns (producer names, appellations) where BM25 exact matching outperforms vector similarity.

---

## Project structure

```
tg-rag/
├── src/
│   ├── ingestion/
│   │   ├── telegram_loader.py   # Telethon: incremental channel parsing
│   │   └── embedder.py          # Embeddings → ChromaDB + local cache
│   ├── retrieval/
│   │   └── retriever.py         # Hybrid search: vector + BM25 + RRF + reranking
│   ├── generation/
│   │   ├── prompt.py            # Prompt templates (system + RAG user message)
│   │   └── chain.py             # RAG chain + query preprocessing
│   ├── evaluation/
│   │   └── analyze_feedback.py  # Feedback analysis: MUR, satisfaction rate, distribution
│   └── ui/
│       ├── bot.py               # Telegram bot (async polling)
│       └── feedback.py          # SQLite feedback storage
├── data/
│   ├── chroma_db/               # Vector store (not committed)
│   ├── embedding_cache/         # Embedding cache (not committed)
│   └── feedback.db              # User ratings (not committed)
├── config.py                    # All parameters in one place
├── run_ingestion.py             # Entry point: load posts → embed → index
└── requirements.txt
```

---

## Setup

**Prerequisites:** Python 3.11+, a Telegram API app, Voyage AI key, Anthropic key, Telegram bot token.

```bash
git clone <repo>
cd tg-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
```

**.env variables:**
```
TELEGRAM_API_ID=
TELEGRAM_API_HASH=
ANTHROPIC_API_KEY=
VOYAGE_API_KEY=
TELEGRAM_BOT_TOKEN=
ANONYMIZED_TELEMETRY=False
```

---

## Usage

```bash
# First run: authenticate with Telegram (creates tg_session.session)
python run_ingestion.py

# Start the bot
python -m src.ui.bot

# Console chat (for testing without Telegram)
python -m src.generation.chain
```

On subsequent runs, `run_ingestion.py` only fetches posts newer than the last known post ID per channel.

---

## Design decisions

**No chunking.** Telegram posts are capped at ~2000 characters — smaller than a typical chunk size. Posts are indexed whole. This avoids split-context problems and simplifies the pipeline.

**Stateless RAG.** No conversation history is passed to Claude. Each question is independent. This reduces token cost, eliminates context overflow risk, and keeps the codebase simple.

**BM25 in memory.** The index (~38k posts) is built at startup (~3–5s) and kept in RAM (~450MB). Acceptable for the server (956MB total). No persistence needed — rebuilt on each restart from ChromaDB documents.

**RRF over linear fusion.** Reciprocal Rank Fusion doesn't require score normalization across vector and BM25 results, is robust to outliers, and matches the original Cormack et al. 2009 formulation with k=60.

**Single Voyage AI provider** for both embeddings and reranking — reduces dependencies and keeps the retrieval stack coherent.