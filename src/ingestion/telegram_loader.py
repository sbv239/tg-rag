import asyncio
import json
import logging
from pathlib import Path

from telethon import TelegramClient
from telethon.tl.types import Message

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Файл для хранения последнего загруженного post_id по каждому каналу
_STATE_FILE = Path("data/loader_state.json")


def _load_state() -> dict:
    """Читает сохранённое состояние (last_post_id по каналам)."""
    if _STATE_FILE.exists():
        with open(_STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_state(state: dict) -> None:
    """Сохраняет состояние на диск."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def build_post_url(channel: str, post_id: int) -> str:
    return f"https://t.me/{channel}/{post_id}"


def format_post(message: Message, channel: str) -> dict:
    """Преобразует Telethon Message в наш формат."""
    return {
        "post_id": message.id,
        "channel": channel,
        "text": message.text or "",
        "date": message.date.isoformat(),
        "url": build_post_url(channel, message.id),
    }


async def load_channel_posts(
    client: TelegramClient,
    channel: str,
    last_post_id: int = 0,
    limit: int = config.POSTS_LIMIT,
    min_length: int = config.MIN_POST_LENGTH,
) -> tuple[list[dict], int]:
    """
    Загружает посты из одного канала начиная после last_post_id.
    Возвращает (список постов, максимальный post_id из загруженных).

    Telethon параметр min_id означает: загружать только посты с id > min_id.
    Посты приходят от новых к старым, поэтому первый пост — самый новый.
    """
    posts = []
    max_post_id = last_post_id

    if last_post_id > 0:
        logger.info(f"Загружаю новые посты из @{channel} (после post_id={last_post_id})")
    else:
        logger.info(f"Загружаю посты из @{channel} впервые (limit={limit})")

    try:
        async for message in client.iter_messages(
            channel,
            limit=limit,
            min_id=last_post_id,  # только посты новее last_post_id
        ):
            if not message.text:
                continue
            if len(message.text) < min_length:
                continue

            posts.append(format_post(message, channel))

            if message.id > max_post_id:
                max_post_id = message.id

        if last_post_id > 0:
            logger.info(f"@{channel}: найдено {len(posts)} новых постов")
        else:
            logger.info(f"@{channel}: загружено {len(posts)} постов")

    except Exception as e:
        logger.error(f"@{channel}: ошибка загрузки — {e}")

    return posts, max_post_id


async def load_all_channels(channels: list[str] = config.TELEGRAM_CHANNELS) -> list[dict]:
    """
    Загружает только новые посты из всех каналов.
    Состояние (последний post_id) сохраняется между запусками.
    """
    state = _load_state()
    all_posts = []

    async with TelegramClient(
        "tg_session",
        config.TELEGRAM_API_ID,
        config.TELEGRAM_API_HASH,
    ) as client:
        for channel in channels:
            last_post_id = state.get(channel, 0)
            posts, max_post_id = await load_channel_posts(
                client, channel, last_post_id=last_post_id
            )
            all_posts.extend(posts)

            # Обновляем состояние только если загрузили новые посты
            if max_post_id > last_post_id:
                state[channel] = max_post_id

    # Сохраняем обновлённое состояние
    _save_state(state)
    logger.info(f"Итого загружено постов: {len(all_posts)} из {len(channels)} каналов")

    return all_posts


def load_posts(channels: list[str] = config.TELEGRAM_CHANNELS) -> list[dict]:
    """Синхронная обёртка для удобного вызова из других модулей."""
    return asyncio.run(load_all_channels(channels))


if __name__ == "__main__":
    posts = load_posts()
    print(f"\nПримеры постов ({min(3, len(posts))}):")
    for post in posts[:3]:
        print(f"  [{post['channel']}] {post['date'][:10]} — {post['text'][:80]}...")
        print(f"  {post['url']}\n")