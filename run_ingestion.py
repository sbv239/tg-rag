import logging

import chromadb
chromadb.config.Settings(anonymized_telemetry=False)

from src.ingestion.telegram_loader import load_posts
from src.ingestion.embedder import index_posts, get_collection_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run():
    logger.info("=== Ingestion Pipeline START ===")

    # 1. Загружаем посты из Telegram
    logger.info("Шаг 1/2: Загрузка постов из Telegram...")
    posts = load_posts()
    logger.info(f"Загружено постов: {len(posts)}")

    if not posts:
        logger.error("Посты не загружены. Проверь TELEGRAM_CHANNELS в config.py")
        return

    # 2. Индексируем в ChromaDB
    logger.info("Шаг 2/2: Индексация в ChromaDB...")
    index_posts(posts)

    stats = get_collection_stats()
    logger.info(f"=== Ingestion Pipeline DONE ===")
    logger.info(f"Коллекция '{stats['collection_name']}': {stats['total_chunks']} постов")


if __name__ == "__main__":
    run()