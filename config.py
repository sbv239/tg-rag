import os
from dotenv import load_dotenv

load_dotenv()

# --- Telegram ---
TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")

# Список каналов для индексации (без @)
TELEGRAM_CHANNELS = [
    "ASWineGuide",
    "ZytZyr",
    "the_daily_winegraph",
    "takoe_vino",
    "winegeekspb",
    "drunk_monday"
]

# Сколько последних постов грузить с каждого канала
POSTS_LIMIT = None

# Минимальная длина поста (символов) — короткие посты не индексируем
MIN_POST_LENGTH = 50

# --- API Keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

# --- Модели ---
EMBEDDING_MODEL = "voyage-4-lite"
LLM_MODEL = "claude-sonnet-4-5"
LLM_MAX_TOKENS = 2500     # максимум токенов в ответе
LLM_TEMPERATURE = 0.2      # низкая температура = фактические ответы, меньше галлюцинаций

# --- Чанкинг ---
CHUNK_SIZE = 500       # токенов
CHUNK_OVERLAP = 50     # токенов

# --- ChromaDB ---
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "tg_posts"

# --- Embedding Cache ---
EMBEDDING_CACHE_PATH = "data/embedding_cache"

# --- Retrieval ---
TOP_K = 10                    # чанков из векторного поиска
HYBRID_VECTOR_WEIGHT = 0.7    # вес векторного поиска в fusion
HYBRID_BM25_WEIGHT = 0.3      # вес BM25 в fusion
RERANKER_TOP_N = 5            # чанков после reranking подаём в LLM