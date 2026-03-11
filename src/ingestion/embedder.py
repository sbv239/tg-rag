import hashlib
import json
import logging
import os
import time
from pathlib import Path

import chromadb
import voyageai

import config

logger = logging.getLogger(__name__)

# --- Voyage AI клиент (новый API) ---
_voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)

# --- ChromaDB ---
_chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
_collection = _chroma_client.get_or_create_collection(
    name=config.COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

# --- Embedding Cache ---
Path(config.EMBEDDING_CACHE_PATH).mkdir(parents=True, exist_ok=True)
_cache_file = Path(config.EMBEDDING_CACHE_PATH) / "embeddings.json"


def _load_cache() -> dict:
    if _cache_file.exists():
        with open(_cache_file, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(_cache_file, "w") as f:
        json.dump(cache, f)


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Возвращает эмбеддинги для списка текстов.
    Использует локальный кэш: если эмбеддинг уже считался — берём из кэша.
    """
    cache = _load_cache()
    embeddings = []
    texts_to_embed = []
    indices_to_embed = []

    for i, text in enumerate(texts):
        key = _text_hash(text)
        if key in cache:
            embeddings.append(cache[key])
        else:
            embeddings.append(None)
            texts_to_embed.append(text)
            indices_to_embed.append(i)

    cache_hits = len(texts) - len(texts_to_embed)
    logger.info(f"Эмбеддинги: {cache_hits} cache hit, {len(texts_to_embed)} новых")

    if texts_to_embed:
        t0 = time.time()
        result = _voyage_client.embed(
            texts_to_embed,
            model=config.EMBEDDING_MODEL,
            input_type="document",
        )
        elapsed = time.time() - t0
        logger.info(f"Voyage AI: получено {len(texts_to_embed)} эмбеддингов за {elapsed:.2f}с")

        for idx, embedding in zip(indices_to_embed, result.embeddings):
            embeddings[idx] = embedding
            key = _text_hash(texts[idx])
            cache[key] = embedding

        _save_cache(cache)

    return embeddings


def index_chunks(chunks: list[dict], batch_size: int = 100) -> None:
    """
    Индексирует чанки в ChromaDB.
    Пропускает уже существующие chunk_id (идемпотентность).
    """
    if not chunks:
        logger.warning("Нет чанков для индексации")
        return

    # Проверяем какие chunk_id уже есть в коллекции
    existing = set(_collection.get(ids=[c["chunk_id"] for c in chunks])["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

    if not new_chunks:
        logger.info("Все чанки уже проиндексированы, пропускаем")
        return

    logger.info(f"Индексируем {len(new_chunks)} новых чанков (батч={batch_size})")

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]

        texts = [c["text"] for c in batch]
        embeddings = get_embeddings(texts)

        _collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "channel": c["channel"],
                    "post_id": c["post_id"],
                    "date": c["date"],
                    "url": c["url"],
                }
                for c in batch
            ],
        )

        logger.info(f"  Проиндексировано батч {i // batch_size + 1}: {len(batch)} чанков")

    logger.info(f"Индексация завершена. Всего в коллекции: {_collection.count()} чанков")


def get_collection_stats() -> dict:
    return {
        "total_chunks": _collection.count(),
        "collection_name": config.COLLECTION_NAME,
    }


if __name__ == "__main__":
    stats = get_collection_stats()
    print(f"ChromaDB: {stats}")