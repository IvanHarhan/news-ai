"""Растёт ли Apple перед презентацией айфона — и что после.

Как меряем, чтобы не обмануться:
- Доходность считается СВЕРХ рынка (AAPL минус SPY), иначе рост всего рынка
  засчитался бы Apple.
- Apple с 2007 года обгоняла рынок почти всегда, поэтому «в среднем выросла
  перед презентацией» ничего не значит само по себе. Сравниваем с тем же
  окном в случайные дни (плацебо): если перед айфоном не лучше, чем в любые
  другие 10 дней, — эффекта нет.
- 0-й день — день презентации (она идёт днём по Нью-Йорку, реакция внутри
  этого дня).

Запуск: .venv\\Scripts\\python.exe iphone_effect.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import prices
from stats import mean_ci, placebo_pct

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Даты презентаций, где показывали новый айфон.
EVENTS = [
    ("2007-01-09", "iPhone"), ("2008-06-09", "iPhone 3G"), ("2009-06-08", "iPhone 3GS"),
    ("2010-06-07", "iPhone 4"), ("2011-10-04", "iPhone 4S"), ("2012-09-12", "iPhone 5"),
    ("2013-09-10", "iPhone 5s"), ("2014-09-09", "iPhone 6"), ("2015-09-09", "iPhone 6s"),
    ("2016-09-07", "iPhone 7"), ("2017-09-12", "iPhone 8/X"), ("2018-09-12", "iPhone XS"),
    ("2019-09-10", "iPhone 11"), ("2020-10-13", "iPhone 12"), ("2021-09-14", "iPhone 13"),
    ("2022-09-07", "iPhone 14"), ("2023-09-12", "iPhone 15"), ("2024-09-09", "iPhone 16"),
    ("2025-09-09", "iPhone 17"),
]

# (название, от, до) — в торговых днях относительно дня презентации.
WINDOWS = [
    ("за 30 дней до", -30, -1), ("за 10 дней до", -10, -1), ("за 5 дней до", -5, -1),
    ("день презентации", 0, 0), ("5 дней после", 1, 5), ("20 дней после", 1, 20),
]


def excess(aapl: pd.Series, spy: pd.Series, i: int, a: int, b: int) -> float:
    """Доходность AAPL сверх SPY за дни [i+a, i+b] (по закрытиям)."""
    s, e = i + a - 1, i + b
    return aapl.iloc[e] / aapl.iloc[s] - spy.iloc[e] / spy.iloc[s]


def main() -> None:
    a = prices.load("AAPL")["close"]
    s = prices.load("SPY")["close"].reindex(a.index).ffill()
    idx = a.index

    ev_i = [int(idx.searchsorted(pd.Timestamp(d))) for d, _ in EVENTS]
    lo = max(-w[1] for w in WINDOWS) + 1
    hi = max(w[2] for w in WINDOWS)
    near = {j for i in ev_i for j in range(i - 40, i + 41)}
    pool_days = [i for i in range(lo, len(idx) - hi) if i not in near]

    lines = [
        f"Айфон-эффект — {date.today()}",
        f"Презентаций: {len(EVENTS)} ({EVENTS[0][0][:4]}–{EVENTS[-1][0][:4]}), доходность AAPL СВЕРХ рынка (SPY).",
        "Плацебо — то же окно в случайные дни без презентаций рядом. 95%+ = необычно, ~50% = как обычно.",
        "",
    ]
    for name, w0, w1 in WINDOWS:
        real = np.array([excess(a, s, i, w0, w1) for i in ev_i])
        pool = np.array([excess(a, s, i, w0, w1) for i in pool_days])
        m, l, h = mean_ci(real)
        pct = placebo_pct(m, pool, len(real))
        lines.append(
            f"{name:<17} среднее {m*100:+5.2f}% [{l*100:+5.2f}; {h*100:+5.2f}]  "
            f"в плюсе {int((real > 0).sum())}/{len(real)}  обычно {pool.mean()*100:+5.2f}%  "
            f"лучше {pct:3.0f}% случайных"
        )

    lines += ["", "По годам (сверх рынка):", f"{'дата':<11} {'модель':<11} {'10д до':>7} {'день':>7} {'5д после':>9}"]
    for (d, model), i in zip(EVENTS, ev_i):
        lines.append(
            f"{d:<11} {model:<11} {excess(a, s, i, -10, -1)*100:+6.1f}% "
            f"{excess(a, s, i, 0, 0)*100:+6.1f}% {excess(a, s, i, 1, 5)*100:+8.1f}%"
        )

    report = "\n".join(lines)
    print(report)
    out = Path(__file__).resolve().parent / "reports" / f"iphone_{date.today()}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    main()
