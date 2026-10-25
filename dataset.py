"""Новость → обучающий пример: что сделала акция СВЕРХ рынка после входа.

Вход — открытие первого торгового дня после новости (или того же дня, если
новость вышла до открытия, см. news.py). Выход:
  ex1 — закрытие того же дня (держим один день),
  ex5 — закрытие 5-го торгового дня.
Из доходности акции вычитается доходность SPY за то же время.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PRICE = 1.0       # копеечные акции — шум и огромные спреды
MAX_ABS_MOVE = 1.0    # ±100% за день почти всегда ошибка данных Yahoo


def build(news_df: pd.DataFrame, panel: dict) -> pd.DataFrame:
    close, open_ = panel["close"], panel["open"]
    cal = close.index[close["SPY"].notna()]
    close, open_ = close.loc[cal], open_.loc[cal]

    df = news_df[news_df["ticker"].isin(close.columns)].copy()
    side = np.where(df["premarket"].to_numpy(), "left", "right")
    days = df["day"].to_numpy()
    e = np.where(side == "left", cal.searchsorted(days, side="left"), cal.searchsorted(days, side="right"))
    df["e"] = e
    df = df[df["e"] + 4 < len(cal)]
    e = df["e"].to_numpy()
    col = close.columns.get_indexer(df["ticker"])

    O, C = open_.to_numpy(), close.to_numpy()
    so, sc = open_["SPY"].to_numpy(), close["SPY"].to_numpy()
    entry = O[e, col]
    df["entry_price"] = entry
    df["ex1"] = C[e, col] / entry - sc[e] / so[e]
    df["ex5"] = C[e + 4, col] / entry - sc[e + 4] / so[e]
    df["entry"] = cal[e]

    ok = (entry >= MIN_PRICE) & np.isfinite(df["ex1"]) & np.isfinite(df["ex5"])
    ok &= (df["ex1"].abs() < MAX_ABS_MOVE) & (df["ex5"].abs() < MAX_ABS_MOVE)
    df = df[ok].drop(columns=["e"]).reset_index(drop=True)
    df["year"] = df["entry"].dt.year
    return df
