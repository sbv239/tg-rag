"""
prompt.py — Шаблоны промптов для RAG-цепочки.

Два шаблона:
  - build_rag_prompt()     : основной QA-промпт с контекстом и citation
  - build_system_prompt()  : системный промпт для Claude
"""

from datetime import date


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — ИИ-ассистент, который отвечает на вопросы на основе постов из Telegram-каналов.

Правила:
1. Отвечай ТОЛЬКО на основе предоставленного контекста. Не используй знания из других источников.
2. Если в контексте нет информации для ответа — честно скажи об этом.
3. Каждое утверждение в ответе должно подкрепляться ссылкой на источник в формате [канал](url).
4. Если несколько источников говорят об одном — упомяни все.
5. Отвечай на том языке, на котором задан вопрос.
6. Будь лаконичен: не пересказывай контекст целиком, выдели главное."""


def build_system_prompt() -> str:
    """Вернуть системный промпт."""
    return SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# RAG prompt builder
# ---------------------------------------------------------------------------

def build_rag_prompt(
    query: str,
    chunks: list[dict],
) -> list[dict]:
    """
    Собрать список сообщений для Anthropic Messages API.

    Args:
        query:   Вопрос пользователя.
        chunks:  Список чанков из retriever'а.
                 Каждый чанк: {text, channel, post_id, date, url, score}

    Returns:
        Список сообщений для передачи в `client.messages.create(messages=...)`.
    """
    context_block = _format_context(chunks)
    user_message = _format_user_message(query, context_block)
    return [{"role": "user", "content": user_message}]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _format_context(chunks: list[dict]) -> str:
    """
    Форматирует список чанков в текстовый блок контекста.

    Каждый источник пронумерован для удобства цитирования Claude.
    Пример:
        [1] Канал: investing_channel | Дата: 2024-11-15 | https://t.me/investing_channel/42
        Текст поста...

        [2] Канал: crypto_news | Дата: 2024-11-14 | https://t.me/crypto_news/99
        Текст поста...
    """
    if not chunks:
        return "Релевантные источники не найдены."

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[{i}] Канал: {chunk['channel']} | "
            f"Дата: {chunk['date']} | "
            f"{chunk['url']}"
        )
        lines.append(header)
        lines.append(chunk["text"].strip())
        lines.append("")  # пустая строка между источниками

    return "\n".join(lines).strip()


def _format_user_message(query: str, context_block: str) -> str:
    """
    Сборка финального user-сообщения с контекстом и вопросом.

    Структура:
        <context>
        ...источники...
        </context>

        Вопрос: ...
    """
    today = date.today().isoformat()

    return (
        f"Сегодняшняя дата: {today}\n\n"
        f"<context>\n"
        f"{context_block}\n"
        f"</context>\n\n"
        f"Вопрос: {query}\n\n"
        f"Дай ответ, ссылаясь на источники из контекста в формате [название канала](ссылка)."
    )