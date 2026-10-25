"""Архив новостей Benzinga 2009–2020 (1.4 млн заголовков, 6200 тикеров).

У 96% новостей есть только дата, без времени. Поэтому вход считается
осторожно: в первый торговый день ПОСЛЕ даты новости, по открытию. Так мы
никогда не покупаем раньше, чем новость могла выйти, — цена дня самой
новости (если она вышла утром) пропадает, и это честная плата за неизвестное
время. Если время известно и новость вышла до открытия (9:30 по Нью-Йорку),
входим в тот же день.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(r"A:\news_ai_data")
RAW = DATA / "benzinga" / "raw_analyst_ratings.jsonl"
CACHE = DATA / "benzinga" / "news.pkl"


def load() -> pd.DataFrame:
    """Колонки: ticker, day (дата новости), premarket (вышла до открытия), headline."""
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    raw = pd.read_json(RAW, lines=True)
    ts = raw["date"].astype(str)
    has_time = ~ts.str.endswith(" 00:00:00")
    stamp = pd.to_datetime(ts.str[:19])          # время уже нью-йоркское (-04:00/-05:00)
    minutes = stamp.dt.hour * 60 + stamp.dt.minute
    df = pd.DataFrame({
        # Yahoo пишет классы акций через дефис: BRK.B → BRK-B
        "ticker": raw["stock"].astype(str).str.upper().str.replace(".", "-", regex=False),
        "day": stamp.dt.normalize(),
        "premarket": has_time & (minutes < 9 * 60 + 30),
        "headline": raw["headline"].astype(str),
    })
    df = df.drop_duplicates(subset=["ticker", "day", "headline"]).reset_index(drop=True)
    df.to_pickle(CACHE)
    return df
