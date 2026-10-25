"""Дневные цены акций с Yahoo с кэшем на диске.

Кэш — по CSV на тикер в data/prices/. Повторный запуск читает с диска и
дозапрашивает только недостающий хвост, а не всю историю.
"""
from __future__ import annotations

import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# yfinance сыплет предупреждениями о будущих изменениях pandas на каждый запрос.
warnings.filterwarnings("ignore", module="yfinance")

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "prices"
START = "2006-01-01"


def load(ticker: str) -> pd.DataFrame:
    """Колонки: open, high, low, close (с учётом сплитов и дивидендов), volume.
    Индекс — дата торгового дня."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker}.csv"
    cached = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None

    start = START if cached is None or cached.empty else (cached.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
    if cached is not None and pd.Timestamp(start).date() > date.today():
        return cached

    fresh = yf.download(ticker, start=start, auto_adjust=True, progress=False, multi_level_index=False)
    if not fresh.empty:
        fresh = fresh.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        fresh.index = pd.to_datetime(fresh.index).tz_localize(None)
        # Сегодняшняя свеча может быть ещё не закрыта — в кэш её не кладём,
        # иначе недорисованная цена осталась бы там навсегда.
        fresh = fresh[fresh.index.date < date.today()]
    if not fresh.empty:
        df = fresh if cached is None else pd.concat([cached, fresh])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(path)
        return df
    if cached is None:
        raise RuntimeError(f"Yahoo не отдал данные по {ticker}")
    return cached
