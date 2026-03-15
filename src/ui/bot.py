"""
bot.py — Telegram-бот поверх RAG-цепочки.

Функциональность:
    - /start  — приветствие и инструкция
    - Любое сообщение → RAG-ответ + источники
    - Отдельное сообщение с оценкой (только если найдены источники)

Один экземпляр RAGChain на всё приложение — история диалога не хранится,
каждый вопрос обрабатывается независимо.

Привязка оценки к ответу:
    Данные каждого ответа хранятся по message_id сообщения с кнопками.
    Это гарантирует что оценка всегда идёт к правильному вопросу,
    даже если пользователь задал несколько вопросов подряд.

Запуск:
    python -m src.ui.bot
"""

import logging
import os
import time

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.generation.chain import RAGChain
from src.ui.feedback import FeedbackDB

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

WELCOME_MESSAGE = (
    "👋 Привет! Я бот для поиска по постам телеграм-каналов о вине.\n\n"
    "Задавай любые вопросы — я найду релевантные посты и отвечу на их основе.\n\n"
    "Например:\n"
    "• Какие вина из Бургундии стоит попробовать?\n"
    "• Что писали про Barolo в последнее время?\n"
    "• Чем отличается Шабли от других Шардоне?\n"
)

RATING_LABELS = {1: "1 ⭐", 2: "2 ⭐", 3: "3 ⭐", 4: "4 ⭐", 5: "5 ⭐"}

# Хранилище данных ответов: message_id → result dict
# Позволяет правильно привязать оценку к конкретному ответу
_pending_ratings: dict[int, dict] = {}

# ---------------------------------------------------------------------------
# Глобальные объекты (инициализируются один раз при старте)
# ---------------------------------------------------------------------------

_chain: RAGChain | None = None
_feedback_db: FeedbackDB | None = None


# ---------------------------------------------------------------------------
# Форматирование ответа
# ---------------------------------------------------------------------------

def _format_sources(sources: list[dict]) -> str:
    """Форматировать список источников в строку с Markdown-ссылками."""
    if not sources:
        return ""
    lines = []
    for src in sources:
        channel = src["channel"]
        url = src["url"]
        date = src["date"][:10]  # YYYY-MM-DD
        lines.append(f"• [{channel}]({url}) — {date}")
    return "\n".join(lines)


def _make_rating_keyboard(user_id: int, message_id: int) -> InlineKeyboardMarkup:
    """Inline-клавиатура с оценками 1–5. message_id привязывает кнопки к конкретному ответу."""
    buttons = [
        InlineKeyboardButton(
            text=label,
            callback_data=f"rate:{user_id}:{message_id}:{score}",
        )
        for score, label in RATING_LABELS.items()
    ]
    return InlineKeyboardMarkup([buttons])


# ---------------------------------------------------------------------------
# Хэндлеры
# ---------------------------------------------------------------------------

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start — приветствие."""
    await update.message.reply_text(WELCOME_MESSAGE, parse_mode=ParseMode.MARKDOWN)
    logger.info("/start | user_id=%d", update.effective_user.id)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка входящего вопроса → RAG-ответ."""
    user_id = update.effective_user.id
    query = update.message.text.strip()

    if not query:
        return

    logger.info("Incoming query | user_id=%d | query=%r", user_id, query)

    await update.message.chat.send_action("typing")

    t_start = time.perf_counter()

    try:
        result = _chain.ask(query)
    except Exception as exc:
        logger.exception("RAGChain error | user_id=%d | error=%s", user_id, exc)
        await update.message.reply_text(
            "⚠️ Произошла ошибка при обработке запроса. Попробуй ещё раз."
        )
        return

    response_time = time.perf_counter() - t_start
    answer = result["answer"]
    sources = result["sources"]

    logger.info(
        "Answer ready | user_id=%d | sources=%d | time=%.2fs",
        user_id, len(sources), response_time,
    )

    # --- Формируем текст ответа ---
    reply_text = answer
    if sources:
        sources_block = _format_sources(sources)
        reply_text = f"{answer}\n\n📎 *Источники:*\n{sources_block}"

    # --- Отправляем ответ (fallback на plain text если Markdown сломан) ---
    try:
        await update.message.reply_text(
            reply_text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    except Exception:
        logger.warning("Markdown parse failed, retrying as plain text")
        await update.message.reply_text(
            reply_text,
            disable_web_page_preview=True,
        )

    # --- Отдельное сообщение с кнопками оценки (только если есть источники) ---
    if sources:
        rating_msg = await update.message.reply_text(
            "Понравился ли вам ответ? Поставьте оценку по шкале от 1 до 5:",
            reply_markup=_make_rating_keyboard(user_id, 0),
        )
        keyboard = _make_rating_keyboard(user_id, rating_msg.message_id)
        await rating_msg.edit_reply_markup(reply_markup=keyboard)

        _pending_ratings[rating_msg.message_id] = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "response_time_s": response_time,
        }


async def rating_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка нажатия на кнопку оценки."""
    cb = update.callback_query
    await cb.answer()

    # Формат callback_data: "rate:{user_id}:{message_id}:{score}"
    try:
        _, target_user_id_str, message_id_str, score_str = cb.data.split(":")
        score = int(score_str)
        target_user_id = int(target_user_id_str)
        message_id = int(message_id_str)
    except (ValueError, AttributeError):
        logger.warning("Invalid callback data: %r", cb.data)
        return

    actual_user_id = update.effective_user.id

    if actual_user_id != target_user_id:
        await cb.answer("Это не твой вопрос 😊", show_alert=True)
        return

    result_data = _pending_ratings.pop(message_id, None)
    if not result_data:
        await cb.edit_message_reply_markup(reply_markup=None)
        return

    if _feedback_db is not None:
        try:
            _feedback_db.save(
                user_id=actual_user_id,
                query=result_data["query"],
                answer=result_data["answer"],
                sources=result_data["sources"],
                rating=score,
                response_time_s=result_data["response_time_s"],
            )
        except Exception as exc:
            logger.exception("Failed to save feedback: %s", exc)

    try:
        await cb.edit_message_text("Спасибо за обратную связь!")
    except Exception as exc:
        logger.warning("Could not edit message: %s", exc)

    logger.info(
        "Rating saved | user_id=%d | score=%d | query=%r",
        actual_user_id, score, result_data["query"],
    )


# ---------------------------------------------------------------------------
# Запуск бота
# ---------------------------------------------------------------------------

def main() -> None:
    global _chain, _feedback_db

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан в .env")

    _chain = RAGChain()
    _feedback_db = FeedbackDB()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CallbackQueryHandler(rating_callback, pattern=r"^rate:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("Bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()