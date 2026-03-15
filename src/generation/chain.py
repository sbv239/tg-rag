"""
chain.py — RAG-цепочка: query preprocessing → retrieval → generation.

Точка входа для всей RAG-логики. Используется как из UI (bot.py),
так и из консоли (python -m src.generation.chain).

Query preprocessing живёт здесь же — ~25 строк, отдельный модуль избыточен.
История диалога не поддерживается — каждый вопрос обрабатывается независимо.
"""

import logging
import re
import time

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    TOP_K,
)
from src.retrieval.retriever import Retriever
from src.generation.prompt import build_system_prompt, build_rag_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Русские стоп-слова для query preprocessing
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "а", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "но", "по", "его", "её", "их", "то", "все", "она", "так", "её",
    "был", "к", "из", "у", "же", "под", "за", "от", "до", "бы",
    "о", "об", "это", "да", "ну", "и", "или", "ли", "уж",
    "мне", "вы", "мы", "вам", "нам", "её", "им",
}


# ---------------------------------------------------------------------------
# Query Preprocessing
# ---------------------------------------------------------------------------

def preprocess_query(query: str) -> str:
    """
    Предобработка запроса перед эмбеддингом.

    Шаги:
        1. Нижний регистр
        2. Удаление лишних пробелов и символов (кроме кириллицы, латиницы, цифр, $)
        3. Удаление русских стоп-слов
        4. Trim

    Сохраняем $ для тикеров ($AAPL, $BTC) — важно для финансового контента.
    """
    original = query
    query = query.lower()

    # Оставляем: кириллицу, латиницу, цифры, пробел, дефис, точку, $
    query = re.sub(r"[^\w\s\-\.$]", " ", query, flags=re.UNICODE)

    # Убираем стоп-слова (только целые слова)
    tokens = query.split()
    tokens = [t for t in tokens if t not in _STOPWORDS]
    query = " ".join(tokens)

    # Нормализуем пробелы
    query = re.sub(r"\s+", " ", query).strip()

    if query != original.lower():
        logger.debug("Query preprocessed: %r → %r", original, query)

    return query


# ---------------------------------------------------------------------------
# RAGChain
# ---------------------------------------------------------------------------

class RAGChain:
    """
    Основная RAG-цепочка. Один экземпляр на всё приложение.

    Каждый вопрос обрабатывается независимо — история диалога не хранится.
    Это упрощает кодовую базу и снижает стоимость запросов к Claude.

    Пример использования:
        chain = RAGChain()
        result = chain.ask("Что писали про Бургундию?")
        print(result["answer"])
        for src in result["sources"]:
            print(src["url"])
    """

    def __init__(self) -> None:
        logger.info("Initializing RAGChain...")
        self._retriever = Retriever()
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("RAGChain ready. Model: %s", LLM_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        *,
        top_k: int = TOP_K,
    ) -> dict:
        """
        Задать вопрос RAG-цепочке.

        Args:
            query:  Вопрос пользователя (сырой текст).
            top_k:  Количество чанков для retrieval.

        Returns:
            {
                "answer":          str        — ответ Claude,
                "sources":         list[dict] — [{channel, url, date, post_id}],
                "chunks":          list[dict] — полные чанки (для отладки),
                "query_processed": str        — запрос после preprocessing,
            }
        """
        t_start = time.perf_counter()
        logger.info("RAGChain.ask() | query=%r", query)

        # 1. Query preprocessing
        processed_query = preprocess_query(query)

        # 2. Retrieval
        t_ret = time.perf_counter()
        chunks = self._retriever.retrieve(processed_query, top_k=top_k)
        logger.info("Retrieved %d chunks in %.3fs", len(chunks), time.perf_counter() - t_ret)

        # 3. Build prompt
        messages = build_rag_prompt(query=query, chunks=chunks)

        # 4. Generate answer
        t_gen = time.perf_counter()
        answer = self._generate(messages)
        logger.info("Generation completed in %.3fs", time.perf_counter() - t_gen)

        # 5. Extract sources (дедупликация по url)
        sources = self._extract_sources(chunks)

        total_time = time.perf_counter() - t_start
        logger.info(
            "RAGChain.ask() done in %.3fs | chunks=%d | sources=%d",
            total_time, len(chunks), len(sources),
        )

        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
            "query_processed": processed_query,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate(self, messages: list[dict]) -> str:
        """Вызов Claude через Anthropic Messages API."""
        response = self._client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=build_system_prompt(),
            messages=messages,
        )
        return response.content[0].text

    @staticmethod
    def _extract_sources(chunks: list[dict]) -> list[dict]:
        """
        Дедуплицировать источники по url и вернуть список для отображения.

        Возвращает: [{channel, url, date, post_id}, ...]
        Сортировка: по дате убывания (самые свежие — первыми).
        """
        seen_urls = set()
        sources = []

        for chunk in chunks:
            url = chunk.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "channel": chunk["channel"],
                    "url": url,
                    "date": chunk["date"],
                    "post_id": chunk["post_id"],
                })

        sources.sort(key=lambda x: x["date"], reverse=True)
        return sources


# ---------------------------------------------------------------------------
# CLI: python -m src.generation.chain
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    chain = RAGChain()

    print("\n=== TG-RAG Консольный чат ===")
    print("Введите вопрос. 'exit' для выхода.\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Выход.")
            break

        result = chain.ask(user_input)

        print(f"\nАссистент: {result['answer']}\n")

        if result["sources"]:
            print("Источники:")
            for src in result["sources"]:
                print(f"  • [{src['channel']}] {src['date']}  {src['url']}")
        print()