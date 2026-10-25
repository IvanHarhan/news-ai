"""FNSPID: статьи с nasdaq.com (Zacks, Motley Fool, RTTNews и др.) — источник
для экзамена 2021–2023. Из 23 ГБ (там полные тексты статей) берём только
дату, заголовок, тикер и издание.

Время в файле — UTC. Ровно 00:00:00 UTC считаем «время неизвестно»: тогда
вход в первый торговый день ПОСЛЕ указанной даты, как у Benzinga. Если время
есть — переводим в нью-йоркское, и новость до 9:30 даёт вход в тот же день.
"""
from __future__ import annotations

import sys

import pandas as pd

from news import DATA

DIR = DATA / "fnspid"
RAW = DIR / "nasdaq_exteral_data.csv"
CACHE = DIR / "news.pkl"


def load() -> pd.DataFrame:
    """Те же колонки, что у news.load(): ticker, day, premarket, headline (+ publisher)."""
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    parts = []
    reader = pd.read_csv(RAW, usecols=["Date", "Article_title", "Stock_symbol", "Publisher"],
                         dtype=str, chunksize=200_000, on_bad_lines="skip")
    for i, ch in enumerate(reader):
        ch = ch.dropna(subset=["Date", "Article_title", "Stock_symbol"])
        utc = pd.to_datetime(ch["Date"].str.replace(" UTC", "", regex=False), errors="coerce", utc=True)
        ok = utc.notna()
        ch, utc = ch[ok], utc[ok]
        has_time = (utc.dt.hour + utc.dt.minute + utc.dt.second) > 0
        ny = utc.dt.tz_convert("America/New_York")
        minutes = ny.dt.hour * 60 + ny.dt.minute
        day = ny.dt.tz_localize(None).dt.normalize().where(has_time, utc.dt.tz_localize(None).dt.normalize())
        parts.append(pd.DataFrame({
            "ticker": ch["Stock_symbol"].str.upper().str.replace(".", "-", regex=False),
            "day": day,
            "premarket": (has_time & (minutes < 9 * 60 + 30)).to_numpy(),
            "headline": ch["Article_title"].astype(str),
            "publisher": ch["Publisher"].fillna(""),
        }))
        print(f"  разобрано пачек: {i + 1} (~{(i + 1) * 200_000:,} строк)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["ticker", "day", "headline"]).reset_index(drop=True)
    df.to_pickle(CACHE)
    return df


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    df = load()
    print(df.shape)
    print(df["day"].dt.year.value_counts().sort_index().to_string())
    print(df["publisher"].value_counts().head(15).to_string())
