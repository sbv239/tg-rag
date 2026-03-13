"""
feedback.py — хранение оценок пользователей в SQLite.

Схема:
    feedback(id, user_id, query, answer, sources_json, rating, response_time_s, ts)

Использование:
    from src.ui.feedback import FeedbackDB
    db = FeedbackDB()
    db.save(user_id=123, query="...", answer="...", sources=[...], rating=5, response_time_s=2.3)
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/feedback.db"


class FeedbackDB:
    """
    Тонкая обёртка над SQLite для записи и чтения пользовательских оценок.

    Thread-safety: sqlite3 в Python поддерживает многопоточность при
    check_same_thread=False + WAL-режиме — достаточно для одного бота.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()
        logger.info("FeedbackDB initialized: %s", self._db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        *,
        user_id: int,
        query: str,
        answer: str,
        sources: list[dict],
        rating: int,
        response_time_s: float,
    ) -> int:
        """
        Сохранить оценку пользователя.

        Args:
            user_id:          Telegram user ID.
            query:            Вопрос пользователя.
            answer:           Ответ бота.
            sources:          Список источников [{channel, url, date, post_id}].
            rating:           Оценка 1–5.
            response_time_s:  Время генерации ответа в секундах.

        Returns:
            id новой записи.
        """
        sources_json = json.dumps(sources, ensure_ascii=False)
        ts = datetime.now(timezone.utc).isoformat()

        cursor = self._conn.execute(
            """
            INSERT INTO feedback (user_id, query, answer, sources_json, rating, response_time_s, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, query, answer, sources_json, rating, round(response_time_s, 3), ts),
        )
        self._conn.commit()
        row_id = cursor.lastrowid
        logger.info(
            "Feedback saved | user=%d rating=%d response_time=%.2fs id=%d",
            user_id, rating, response_time_s, row_id,
        )
        return row_id

    def get_stats(self) -> dict:
        """
        Базовая статистика для мониторинга.

        Returns:
            {total, avg_rating, ratings_distribution: {1: n, 2: n, ...}}
        """
        row = self._conn.execute(
            "SELECT COUNT(*), AVG(rating) FROM feedback"
        ).fetchone()
        total = row[0] or 0
        avg_rating = round(row[1], 2) if row[1] else None

        dist_rows = self._conn.execute(
            "SELECT rating, COUNT(*) FROM feedback GROUP BY rating ORDER BY rating"
        ).fetchall()
        distribution = {r: c for r, c in dist_rows}

        return {
            "total": total,
            "avg_rating": avg_rating,
            "ratings_distribution": distribution,
        }

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id          INTEGER NOT NULL,
                query            TEXT    NOT NULL,
                answer           TEXT    NOT NULL,
                sources_json     TEXT    NOT NULL,
                rating           INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                response_time_s  REAL    NOT NULL,
                ts               TEXT    NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)"
        )
        self._conn.commit()