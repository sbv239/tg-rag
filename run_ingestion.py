import logging

from src.ingestion.telegram_loader import load_posts
from src.ingestion.chunker import chunk_posts
from src.ingestion.embedder import index_chunks, get_collection_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run():
    logger.info("=== Ingestion Pipeline START ===")

    # 1. Загружаем посты из Telegram
    logger.info("Шаг 1/3: Загрузка постов из Telegram...")
    posts = load_posts()
    logger.info(f"Загружено постов: {len(posts)}")

    if not posts:
        logger.error("Посты не загружены. Проверь TELEGRAM_CHANNELS в config.py")
        return

    # 2. Разбиваем на чанки
    logger.info("Шаг 2/3: Чанкинг...")
    chunks = chunk_posts(posts)
    logger.info(f"Получено чанков: {len(chunks)}")

    # 3. Индексируем в ChromaDB
    logger.info("Шаг 3/3: Индексация в ChromaDB...")
    index_chunks(chunks)

    # Итог
    stats = get_collection_stats()
    logger.info(f"=== Ingestion Pipeline DONE ===")
    logger.info(f"Коллекция '{stats['collection_name']}': {stats['total_chunks']} чанков")


if __name__ == "__main__":
    run()