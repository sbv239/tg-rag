"""
retriever.py — Hybrid search (vector + BM25) с RRF fusion и Voyage reranking.

Спринт 2: базовый vector search (top-k cosine similarity).
Спринт 4: hybrid search (vector + BM25) + RRF fusion + reranking.

Пайплайн:
    query
      ├── embed → ChromaDB vector search → top-K (vector results)
      └── tokenize → BM25 index → top-K          (bm25 results)
                        ↓
                  RRF fusion → единый top-K
                        ↓
                  Voyage reranker → top-N → в LLM
"""

import logging
import time
from collections import defaultdict

import chromadb
import voyageai
from rank_bm25 import BM25Okapi

from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    VOYAGE_API_KEY,
    EMBEDDING_MODEL,
    TOP_K,
    RERANKER_TOP_N,
)

logger = logging.getLogger(__name__)

# Константа для RRF: сглаживает влияние топовых позиций.
# k=60 — стандартное значение из оригинальной статьи Cormack et al. 2009.
_RRF_K = 60


class Retriever:
    """
    Hybrid ретривер: vector search + BM25 + RRF fusion + reranking.

    При инициализации загружает все тексты из ChromaDB и строит BM25-индекс.
    38k постов × ~500 символов ≈ ~450MB RAM — допустимо для нашего сервера (956MB).

    Public API:
        retrieve(query, top_k, rerank_top_n) → list[dict]
    """

    def __init__(self) -> None:
        logger.info("Initializing Retriever...")

        self._voyage = voyageai.Client(api_key=VOYAGE_API_KEY)

        self._chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self._collection = self._chroma.get_collection(name=COLLECTION_NAME)

        total = self._collection.count()
        logger.info(
            "Collection '%s' has %d documents. Building BM25 index...",
            COLLECTION_NAME,
            total,
        )

        # Загружаем все документы для BM25-индекса
        t0 = time.perf_counter()
        self._all_docs, self._all_ids = self._load_all_documents()
        self._bm25 = self._build_bm25_index(self._all_docs)
        logger.info(
            "BM25 index built over %d documents in %.2fs",
            len(self._all_docs),
            time.perf_counter() - t0,
        )

        logger.info("Retriever ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = TOP_K,
        rerank_top_n: int = RERANKER_TOP_N,
    ) -> list[dict]:
        """
        Найти rerank_top_n наиболее релевантных постов.

        Args:
            query:        Поисковый запрос (предобработанный в chain.py).
            top_k:        Кол-во кандидатов из каждого поиска (vector и BM25).
            rerank_top_n: Кол-во постов после reranking — идут в LLM.

        Returns:
            Список словарей (до rerank_top_n штук) с ключами:
                text, channel, post_id, date, url, score
        """
        logger.info(
            "Retriever.retrieve() | query=%r | top_k=%d | rerank_top_n=%d",
            query, top_k, rerank_top_n,
        )

        # 1. Embed query (нужен и для vector search)
        t0 = time.perf_counter()
        query_embedding = self._embed_query(query)
        logger.debug("Query embedded in %.3fs", time.perf_counter() - t0)

        # 2. Vector search
        t1 = time.perf_counter()
        vector_results = self._vector_search(query_embedding, top_k=top_k)
        logger.info(
            "Vector search: %d results in %.3fs",
            len(vector_results), time.perf_counter() - t1,
        )

        # 3. BM25 search
        t2 = time.perf_counter()
        bm25_results = self._bm25_search(query, top_k=top_k)
        logger.info(
            "BM25 search: %d results in %.3fs",
            len(bm25_results), time.perf_counter() - t2,
        )

        # 4. RRF fusion
        t3 = time.perf_counter()
        fused = self._rrf_fusion(vector_results, bm25_results, top_k=top_k)
        logger.info(
            "RRF fusion: %d candidates in %.3fs",
            len(fused), time.perf_counter() - t3,
        )

        # 5. Voyage reranking
        t4 = time.perf_counter()
        reranked = self._rerank(query, fused, top_n=rerank_top_n)
        logger.info(
            "Reranking: %d results in %.3fs",
            len(reranked), time.perf_counter() - t4,
        )

        return reranked

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def _load_all_documents(self) -> tuple[list[str], list[str]]:
        """
        Загрузить все тексты и их ChromaDB-id одним запросом.

        ChromaDB не поддерживает пагинацию через offset в старых версиях,
        поэтому тянем всё через get() с большим limit.
        """
        total = self._collection.count()
        raw = self._collection.get(
            limit=total,
            include=["documents"],
        )
        docs = raw["documents"]   # list[str]
        ids = raw["ids"]          # list[str]
        return docs, ids

    def _build_bm25_index(self, documents: list[str]) -> BM25Okapi:
        """
        Построить BM25Okapi индекс по всем документам.

        Токенизация: lowercase + split по пробелам.
        Для русского/смешанного текста простой whitespace split работает лучше,
        чем NLTK (не требует словарей, сохраняет слова целиком).
        """
        tokenized = [doc.lower().split() for doc in documents]
        return BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        """Получить эмбеддинг запроса через Voyage AI."""
        response = self._voyage.embed(
            texts=[query],
            model=EMBEDDING_MODEL,
            input_type="query",
        )
        return response.embeddings[0]

    def _vector_search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict]:
        """
        Векторный поиск в ChromaDB.

        Возвращает список словарей, отсортированных по убыванию score.
        score = 1 - cosine_distance/2  →  [0..1], выше = лучше.
        """
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            score = 1.0 - dist / 2.0
            chunks.append(self._make_chunk(doc, meta, score))

        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """
        BM25 поиск по всем документам.

        Возвращает top_k документов по BM25-скору, обогащённых метаданными
        из ChromaDB (channel, date, url и т.д.).
        """
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Берём индексы top_k документов по убыванию скора
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        # Фильтруем нулевые скоры — документы без совпадений не нужны
        top_indices = [i for i in top_indices if scores[i] > 0]

        if not top_indices:
            logger.debug("BM25: no non-zero scores for query %r", query)
            return []

        # Получаем метаданные для найденных документов из ChromaDB
        chroma_ids = [self._all_ids[i] for i in top_indices]
        raw = self._collection.get(
            ids=chroma_ids,
            include=["documents", "metadatas"],
        )

        # ChromaDB.get() не гарантирует порядок — строим маппинг id → данные
        id_to_data = {
            chroma_id: (doc, meta)
            for chroma_id, doc, meta in zip(
                raw["ids"], raw["documents"], raw["metadatas"]
            )
        }

        chunks = []
        for i in top_indices:
            chroma_id = self._all_ids[i]
            if chroma_id not in id_to_data:
                continue
            doc, meta = id_to_data[chroma_id]
            # Нормализуем BM25-скор в [0..1] для единообразия
            max_score = scores[top_indices[0]] or 1.0
            normalized_score = round(scores[i] / max_score, 4)
            chunks.append(self._make_chunk(doc, meta, normalized_score))

        return chunks

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF) двух списков результатов.

        RRF_score(doc) = Σ 1 / (k + rank)
        где rank — позиция документа в списке (1-based), k=60.

        Документы идентифицируются по url (уникален для каждого поста).
        После fusion возвращаем top_k по убыванию RRF-скора.
        """
        rrf_scores: dict[str, float] = defaultdict(float)

        # url → chunk для восстановления данных после слияния
        url_to_chunk: dict[str, dict] = {}

        for rank, chunk in enumerate(vector_results, start=1):
            url = chunk["url"]
            rrf_scores[url] += 1.0 / (_RRF_K + rank)
            url_to_chunk[url] = chunk

        for rank, chunk in enumerate(bm25_results, start=1):
            url = chunk["url"]
            rrf_scores[url] += 1.0 / (_RRF_K + rank)
            url_to_chunk[url] = chunk

        # Сортируем по убыванию RRF-скора, берём top_k
        sorted_urls = sorted(rrf_scores, key=lambda u: rrf_scores[u], reverse=True)[:top_k]

        fused = []
        for url in sorted_urls:
            chunk = url_to_chunk[url].copy()
            chunk["score"] = round(rrf_scores[url], 6)
            fused.append(chunk)

        return fused

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int,
    ) -> list[dict]:
        """
        Переранжировать кандидатов через Voyage AI rerank-2.

        Voyage принимает query + список текстов, возвращает их в новом порядке
        с relevance_score для каждого. Берём top_n лучших.

        Если кандидатов меньше top_n — возвращаем всё что есть без вызова API.
        """
        if not candidates:
            return []

        if len(candidates) <= top_n:
            logger.debug("Reranker skipped: %d candidates ≤ top_n=%d", len(candidates), top_n)
            return candidates

        texts = [c["text"] for c in candidates]

        response = self._voyage.rerank(
            query=query,
            documents=texts,
            model="rerank-2",
            top_k=top_n,
        )

        reranked = []
        for result in response.results:
            chunk = candidates[result.index].copy()
            chunk["score"] = round(result.relevance_score, 4)
            reranked.append(chunk)

        logger.debug(
            "Reranker scores: %s",
            [c["score"] for c in reranked],
        )

        return reranked

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk(doc: str, meta: dict, score: float) -> dict:
        """Собрать стандартный словарь чанка из сырых данных ChromaDB."""
        return {
            "text": doc,
            "channel": meta.get("channel", ""),
            "post_id": meta.get("post_id", ""),
            "date": meta.get("date", ""),
            "url": meta.get("url", ""),
            "score": round(score, 4),
        }