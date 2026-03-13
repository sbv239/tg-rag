"""
bot.py — Telegram-бот поверх RAG-цепочки.

Функциональность:
    - /start  — приветствие и инструкция
    - /clear  — сброс истории диалога
    - Любое сообщение → RAG-ответ + источники
    - Inline-кнопки оценки ⭐–⭐⭐⭐⭐⭐ (только если найдены источники)

История диалога:
    Каждый пользователь получает свой экземпляр RAGChain.
    Хранится in-memory: dict[user_id → RAGChain].
    Сбрасывается командой /clear или перезапуском бота.

Запуск:
    python -m src.ui.bot
"""

import logging
import os
import time

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
    "• Чем отличается Шабли от других Шардоне?\n\n"
    "Команды:\n"
    "/clear — сбросить историю диалога"
)

RATING_LABELS = {1: "1 ⭐", 2: "2 ⭐", 3: "3 ⭐", 4: "4 ⭐", 5: "5 ⭐"}

# Ключ для хранения данных последнего ответа в user_data (для привязки оценки)
LAST_RESULT_KEY = "last_result"

# ---------------------------------------------------------------------------
# Глобальные объекты (инициализируются один раз при старте)
# ---------------------------------------------------------------------------

_chains: dict[int, RAGChain] = {}   # user_id → RAGChain
_feedback_db: FeedbackDB | None = None


def _get_chain(user_id: int) -> RAGChain:
    """Вернуть (или создать) RAGChain для пользователя."""
    if user_id not in _chains:
        logger.info("Creating new RAGChain for user_id=%d", user_id)
        _chains[user_id] = RAGChain()
    return _chains[user_id]


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


def _make_rating_keyboard(user_id: int) -> InlineKeyboardMarkup:
    """Inline-клавиатура с оценками 1–5."""
    buttons = [
        InlineKeyboardButton(
            text=label,
            callback_data=f"rate:{user_id}:{score}",
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


async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/clear — сброс истории диалога."""
    user_id = update.effective_user.id
    if user_id in _chains:
        _chains[user_id].clear_history()
    # Очищаем также сохранённый последний результат
    context.user_data.pop(LAST_RESULT_KEY, None)
    await update.message.reply_text("🗑 История диалога очищена.")
    logger.info("/clear | user_id=%d", user_id)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка входящего вопроса → RAG-ответ."""
    user_id = update.effective_user.id
    query = update.message.text.strip()

    if not query:
        return

    logger.info("Incoming query | user_id=%d | query=%r", user_id, query)

    # Индикатор печатания пока думаем
    await update.message.chat.send_action("typing")

    chain = _get_chain(user_id)
    t_start = time.perf_counter()

    try:
        result = chain.ask(query)
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

    # --- Отправляем ответ ---
    if sources:
        # С кнопками оценки
        keyboard = _make_rating_keyboard(user_id)
        await update.message.reply_text(
            reply_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard,
            disable_web_page_preview=True,
        )
        # Сохраняем данные для привязки оценки
        context.user_data[LAST_RESULT_KEY] = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "response_time_s": response_time,
        }
    else:
        # Без кнопок — источники не найдены
        await update.message.reply_text(
            reply_text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )


async def rating_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка нажатия на кнопку оценки."""
    query = update.callback_query
    await query.answer()  # убирает "часики" у кнопки

    # Формат callback_data: "rate:{user_id}:{score}"
    try:
        _, target_user_id_str, score_str = query.data.split(":")
        score = int(score_str)
        target_user_id = int(target_user_id_str)
    except (ValueError, AttributeError):
        logger.warning("Invalid callback data: %r", query.data)
        return

    actual_user_id = update.effective_user.id

    # Защита: оценивать может только тот, кто задал вопрос
    if actual_user_id != target_user_id:
        await query.answer("Это не твой вопрос 😊", show_alert=True)
        return

    # Берём данные последнего ответа
    last_result = context.user_data.get(LAST_RESULT_KEY)
    if not last_result:
        await query.edit_message_reply_markup(reply_markup=None)
        return

    # Сохраняем оценку
    if _feedback_db is not None:
        try:
            _feedback_db.save(
                user_id=actual_user_id,
                query=last_result["query"],
                answer=last_result["answer"],
                sources=last_result["sources"],
                rating=score,
                response_time_s=last_result["response_time_s"],
            )
        except Exception as exc:
            logger.exception("Failed to save feedback: %s", exc)

    # Убираем кнопки и ставим подтверждение
    stars = "⭐" * score
    try:
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(f"Спасибо за оценку! {stars}")
    except Exception as exc:
        logger.warning("Could not edit message: %s", exc)

    # Очищаем last_result чтобы повторное нажатие не сохраняло дубли
    context.user_data.pop(LAST_RESULT_KEY, None)

    logger.info("Rating saved | user_id=%d | score=%d", actual_user_id, score)


# ---------------------------------------------------------------------------
# Запуск бота
# ---------------------------------------------------------------------------

def main() -> None:
    global _feedback_db

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан в .env")

    _feedback_db = FeedbackDB()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("clear", clear_handler))
    app.add_handler(CallbackQueryHandler(rating_callback, pattern=r"^rate:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("Bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()