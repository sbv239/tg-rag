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
    "drunk_monday",
    "inmyglass",
    "winestate",
    "drunken_masta",
    "vraskovOFF",
    "okolovina",
    "simple_wine_news",
    "RamonSosnovskiy",
    "SommIlya",
    "dp_trade_telegram",
    "thewineology",
    "drinkitaly",
    "murashko_anna_wine",
    "nikasomm",
    "wineretail",
    "Wineandme_AltoAdige",
    "sawwwaa",
    "caxapandwine",
    "nevvino"
]

# Сколько последних постов грузить с каждого канала
POSTS_LIMIT = None

# Минимальная длина поста (символов) — короткие посты не индексируем
MIN_POST_LENGTH = 250

# --- API Keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

# --- Модели ---
EMBEDDING_MODEL = "voyage-4-lite"
RERANKER_MODEL = "rerank-2"          # Voyage AI reranker
LLM_MODEL = "claude-sonnet-4-5"
LLM_MAX_TOKENS = 2500     # максимум токенов в ответе
LLM_TEMPERATURE = 0.2      # низкая температура = фактические ответы, меньше галлюцинаций

# --- ChromaDB ---
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "tg_posts"

# --- Embedding Cache ---
EMBEDDING_CACHE_PATH = "data/embedding_cache"

# --- Retrieval ---
TOP_K = 20              # кандидатов из каждого поиска (vector и BM25) — больше кандидатов для reranker
RERANKER_TOP_N = 5      # постов после reranking подаём в LLM

# Веса для возможной линейной fusion (не используются при RRF, оставлены для экспериментов)
HYBRID_VECTOR_WEIGHT = 0.7
HYBRID_BM25_WEIGHT = 0.3