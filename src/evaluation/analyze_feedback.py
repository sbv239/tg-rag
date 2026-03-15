"""
analyze_feedback.py — анализ пользовательских оценок из feedback.db.

Метрики:
    - MUR (Mean User Rating)       — среднее по всем оценкам
    - Satisfaction Rate            — доля оценок >= 4
    - Dissatisfaction Rate         — доля оценок <= 2
    - Rating Distribution          — гистограмма по оценкам
    - Median Response Time         — медианное время ответа
    - Low-rated Queries            — примеры плохих ответов (рейтинг <= 2)

Использование:
    python -m src.evaluation.analyze_feedback
    python -m src.evaluation.analyze_feedback --db data/feedback.db --low-rated 10
"""

import argparse
import json
import sqlite3
import statistics
from pathlib import Path

DEFAULT_DB_PATH = "data/feedback.db"


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_feedback(db_path: str) -> list[dict]:
    """Загрузить все записи из feedback.db."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, user_id, query, rating, response_time_s, ts FROM feedback ORDER BY ts"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(rows: list[dict]) -> dict:
    """Посчитать все метрики по списку записей."""
    if not rows:
        return {}

    ratings = [r["rating"] for r in rows]
    times = [r["response_time_s"] for r in rows]

    total = len(ratings)
    mur = round(statistics.mean(ratings), 2)
    median_rating = statistics.median(ratings)
    satisfaction_rate = round(sum(1 for r in ratings if r >= 4) / total * 100, 1)
    dissatisfaction_rate = round(sum(1 for r in ratings if r <= 2) / total * 100, 1)

    distribution = {}
    for star in range(1, 6):
        count = ratings.count(star)
        distribution[star] = {"count": count, "pct": round(count / total * 100, 1)}

    median_time = round(statistics.median(times), 2)
    avg_time = round(statistics.mean(times), 2)

    return {
        "total": total,
        "mur": mur,
        "median_rating": median_rating,
        "satisfaction_rate": satisfaction_rate,
        "dissatisfaction_rate": dissatisfaction_rate,
        "distribution": distribution,
        "median_response_time_s": median_time,
        "avg_response_time_s": avg_time,
    }


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

BAR_WIDTH = 30  # символов в баре гистограммы


def print_report(metrics: dict, rows: list[dict], low_rated_n: int) -> None:
    """Вывести читаемый отчёт в stdout."""
    if not metrics:
        print("❌  Нет данных в feedback.db")
        return

    print()
    print("=" * 52)
    print("  FEEDBACK REPORT")
    print("=" * 52)

    print(f"\n  Всего оценок        : {metrics['total']}")
    print(f"  MUR                 : {metrics['mur']} / 5.0")
    print(f"  Медиана             : {metrics['median_rating']}")
    print(f"  Satisfaction Rate   : {metrics['satisfaction_rate']}%  (оценки ≥ 4)")
    print(f"  Dissatisfaction Rate: {metrics['dissatisfaction_rate']}%  (оценки ≤ 2)")
    print(f"  Медиана времени     : {metrics['median_response_time_s']}s")
    print(f"  Среднее время       : {metrics['avg_response_time_s']}s")

    print("\n  РАСПРЕДЕЛЕНИЕ ОЦЕНОК\n")
    dist = metrics["distribution"]
    max_count = max(d["count"] for d in dist.values()) or 1
    for star in range(5, 0, -1):
        d = dist[star]
        bar_len = int(d["count"] / max_count * BAR_WIDTH)
        bar = "█" * bar_len
        print(f"  {'⭐' * star:<12} {bar:<{BAR_WIDTH}}  {d['count']:>4}  ({d['pct']}%)")

    # Примеры плохих ответов
    if low_rated_n > 0:
        bad = [r for r in rows if r["rating"] <= 2]
        if bad:
            print(f"\n  НИЗКИЕ ОЦЕНКИ (≤ 2) — последние {min(low_rated_n, len(bad))} из {len(bad)}\n")
            for r in bad[-low_rated_n:]:
                ts = r["ts"][:16].replace("T", " ")
                print(f"  [{r['rating']}⭐] {ts}  |  {r['response_time_s']}s")
                print(f"  Вопрос: {r['query'][:120]}")
                print()

    print("=" * 52)
    print()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ feedback.db")
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Путь к feedback.db (по умолчанию: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--low-rated",
        type=int,
        default=5,
        metavar="N",
        help="Показать N последних вопросов с оценкой ≤ 2 (0 — не показывать)",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"❌  Файл не найден: {args.db}")
        return

    rows = load_feedback(args.db)
    metrics = compute_metrics(rows)
    print_report(metrics, rows, low_rated_n=args.low_rated)


if __name__ == "__main__":
    main()
