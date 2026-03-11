"""
retriever.py — Векторный поиск через ChromaDB с metadata filters.

Спринт 2: базовый vector search (top-k cosine similarity).
Спринт 4: будет расширен до hybrid search (vector + BM25) + reranking.
"""

import logging
import time
from typing import Optional

import chromadb
import voyageai

from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    VOYAGE_API_KEY,
    EMBEDDING_MODEL,
    TOP_K,
)

logger = logging.getLogger(__name__)


class Retriever:
    """
    Базовый векторный ретривер.

    Пайплайн (Спринт 2):
        query → embed → ChromaDB vector search → top-k chunks

    Пайплайн (Спринт 4, будет добавлено):
        query → embed → vector search
                      → BM25 search
                      → score fusion
                      → reranking
                      → top-n chunks
    """

    def __init__(self) -> None:
        logger.info("Initializing Retriever...")

        self._voyage = voyageai.Client(api_key=VOYAGE_API_KEY)

        self._chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self._collection = self._chroma.get_collection(name=COLLECTION_NAME)

        logger.info(
            "Retriever ready. Collection '%s' has %d documents.",
            COLLECTION_NAME,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Найти top_k чанков, релевантных запросу.

        Args:
            query:     Поисковый запрос (уже предобработанный в chain.py).
            channel:   Фильтр по названию канала (точное совпадение).
            date_from: Включительная нижняя граница даты публикации.
            date_to:   Включительная верхняя граница даты публикации.
            top_k:     Количество возвращаемых чанков.

        Returns:
            Список словарей с ключами:
                - text      : str   — текст чанка
                - channel   : str   — название канала
                - post_id   : str   — ID поста
                - date      : str   — дата публикации
                - url       : str   — ссылка на пост
                - score     : float — cosine similarity (0..1, чем выше — тем лучше)
        """
        logger.info("Retriever.retrieve() | query=%r | channel=%s | date_from=%s | date_to=%s | top_k=%d",
                    query, top_k)

        # 1. Embed query
        t0 = time.perf_counter()
        query_embedding = self._embed_query(query)
        logger.debug("Query embedded in %.3fs", time.perf_counter() - t0)

        # 2. Build metadata filter
        where_filter = self._build_where_filter(channel, date_from, date_to)

        # 3. Vector search in ChromaDB
        t1 = time.perf_counter()
        results = self._vector_search(query_embedding, top_k=top_k, where=where_filter)
        logger.info("Vector search returned %d chunks in %.3fs", len(results), time.perf_counter() - t1)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        """Получить эмбеддинг запроса через Voyage AI."""
        response = self._voyage.embed(
            texts=[query],
            model=EMBEDDING_MODEL,
            input_type="query",  # query vs document — важно для voyage
        )
        return response.embeddings[0]

    def _vector_search(
        self,
        query_embedding: list[float],
        top_k: int,
        where: Optional[dict],
    ) -> list[dict]:
        """
        Запрос к ChromaDB, возвращает нормализованные словари.

        ChromaDB возвращает distances (L2 по умолчанию или cosine если задан
        при создании коллекции). Конвертируем в score: чем выше — тем лучше.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        chunks = []
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Конвертируем в similarity [0..1]: score = 1 - dist/2
            score = 1.0 - dist / 2.0

            chunks.append({
                "text": doc,
                "channel": meta.get("channel", ""),
                "post_id": meta.get("post_id", ""),
                "date": meta.get("date", ""),
                "url": meta.get("url", ""),
                "score": round(score, 4),
            })

        # Сортируем по убыванию score (ChromaDB обычно уже сортирует, но явно надёжнее)
        chunks.sort(key=lambda x: x["score"], reverse=True)

        return chunks