import logging
import tiktoken

import config

logger = logging.getLogger(__name__)

# Используем cl100k_base — стандартный токенизатор для большинства моделей
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def split_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """
    Разбивает текст на чанки по токенам с overlap.
    Если текст короче chunk_size — возвращает как один чанк.
    """
    tokens = _tokenizer.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end >= len(tokens):
            break

        start += chunk_size - chunk_overlap

    return chunks


def chunk_posts(posts: list[dict]) -> list[dict]:
    """
    Принимает список постов, возвращает список чанков с метаданными.

    Каждый чанк:
    {
        "chunk_id":  "<channel>_<post_id>_<idx>",
        "text":      "...",
        "channel":   "channel_name",
        "post_id":   123,
        "date":      "2024-01-15T10:30:00+00:00",
        "url":       "https://t.me/channel/123",
    }
    """
    all_chunks = []
    skipped = 0

    for post in posts:
        text = post["text"].strip()
        if not text:
            skipped += 1
            continue

        parts = split_text(text)

        for idx, part in enumerate(parts):
            chunk = {
                "chunk_id": f"{post['channel']}_{post['post_id']}_{idx}",
                "text": part,
                "channel": post["channel"],
                "post_id": post["post_id"],
                "date": post["date"],
                "url": post["url"],
            }
            all_chunks.append(chunk)

    logger.info(
        f"Чанкинг завершён: {len(posts)} постов → {len(all_chunks)} чанков "
        f"(пропущено: {skipped})"
    )
    return all_chunks


if __name__ == "__main__":
    # Быстрый тест без Telegram
    sample_posts = [
        {
            "post_id": 1,
            "channel": "test_channel",
            "text": "Короткий пост.",
            "date": "2024-01-15T10:00:00+00:00",
            "url": "https://t.me/test_channel/1",
        },
        {
            "post_id": 2,
            "channel": "test_channel",
            "text": " ".join(["слово"] * 600),  # длинный пост > 500 токенов
            "date": "2024-01-15T11:00:00+00:00",
            "url": "https://t.me/test_channel/2",
        },
    ]

    chunks = chunk_posts(sample_posts)
    print(f"Получено чанков: {len(chunks)}")
    for c in chunks:
        print(f"  {c['chunk_id']}: {count_tokens(c['text'])} токенов")