"""Цены открытия и закрытия по тысячам тикеров 2009–2020 — для разметки новостей.

Отдельно от prices.py: там кэш по одному CSV на тикер с 2006 года по сей
день, для 3000 тикеров это полгигабайта лишнего. Здесь — две таблицы
(дата × тикер) в одном файле на диске A.

Тикеры, которых Yahoo не знает (делистинг, переименование), запоминаются и
больше не запрашиваются. Новости по ним выпадают — это искажение (пропадают
в основном умершие компании), о нём сказано в отчёте.
"""
from __future__ import annotations

import time
import warnings

import pandas as pd
import yfinance as yf
from yfinance import shared as yf_shared

from news import DATA

warnings.filterwarnings("ignore", module="yfinance")

PANEL = DATA / "prices_2009_2020.pkl"
START, END = "2008-12-01", "2020-07-31"
# Экзамен 2021–2023 — отдельный файл, чтобы не смешивать с обучающей панелью.
EXAM_PANEL = DATA / "prices_2021_2023.pkl"
EXAM_START, EXAM_END = "2020-11-01", "2024-02-15"
CHUNK = 50
PAUSE = 8                  # между пачками — Yahoo режет частые запросы
RATE_LIMIT_WAIT = 300      # после блокировки Yahoo «остывает» минутами
RETRIES = 4


# Тикеры, которые точно торгуются весь период. Кладутся в КАЖДУЮ пачку: если
# хоть один не пришёл — пачку заблокировали, и пустые ответы по остальным
# ничего не значат. Отдельный контрольный запрос не годится: Yahoo пропускает
# одиночный запрос и в ту же секунду режет пачку.
CANARIES = ["SPY", "AAPL", "IBM"]


def _got(raw, todo) -> list[str]:
    if raw.empty:
        return []
    c = raw["Close"]
    return [t for t in todo if t in c.columns and c[t].notna().sum() > 20]


def _download(part: list[str], start: str, end: str):
    """Скачать пачку. Возвращает (кадры, тикеры, которые не удалось проверить).

    Yahoo блокирует двумя способами: явно (ошибка «rate limit») и молча — после
    блокировки отвечает «possibly delisted» на всё подряд, даже на AMZN. Поэтому
    тикер записывается в «не существует», только если в том же запросе пришли
    все контрольные тикеры."""
    frames, todo = [], list(part)
    for attempt in range(RETRIES + 1):
        req = todo + [c for c in CANARIES if c not in todo]
        raw = yf.download(req, start=start, end=end, auto_adjust=True, progress=False,
                          group_by="column", threads=True)
        got = _got(raw, req)
        frames.append(raw)
        if all(c in got for c in CANARIES):
            limited = [t for t, err in yf_shared._ERRORS.items()
                       if t in todo and t not in got and "rate" in str(err).lower()]
        else:
            limited = [t for t in todo if t not in got]
        if not limited or attempt == RETRIES:
            return frames, limited
        print(f"  Yahoo заблокировал пачку ({len(limited)} тикеров), жду {RATE_LIMIT_WAIT} с", flush=True)
        time.sleep(RATE_LIMIT_WAIT)
        todo = limited


def load(tickers: list[str], start: str = START, end: str = END, path=PANEL) -> dict[str, pd.DataFrame]:
    """{'open': DataFrame, 'close': DataFrame, 'missing': [...]}, колонки — тикеры."""
    have = pd.read_pickle(path) if path.exists() else {"open": pd.DataFrame(), "close": pd.DataFrame(), "missing": []}
    todo = [t for t in dict.fromkeys(tickers + ["SPY"])
            if t not in have["close"].columns and t not in have["missing"]]
    if not todo:
        return have

    opens, closes, missing = [have["open"]], [have["close"]], list(have["missing"])
    still_limited: list[str] = []
    out = have
    for i in range(0, len(todo), CHUNK):
        part = todo[i:i + CHUNK]
        frames, limited = _download(part, start, end)
        got: set[str] = set()
        for raw in frames:
            if raw.empty:
                continue
            c = raw["Close"]
            # Только запрошенные: контрольные тикеры уже лежат в панели.
            ok = [t for t in part if t in c.columns and t not in got and c[t].notna().sum() > 20]
            got.update(ok)
            opens.append(raw["Open"][ok])
            closes.append(c[ok])
        still_limited += limited
        missing += [t for t in part if t not in got and t not in limited]
        # Сохраняем после каждой пачки: с паузами на лимит загрузка идёт долго,
        # и оборванный запуск продолжится с того же места.
        out = _save(opens, closes, missing, path)
        print(f"  цены: {min(i + CHUNK, len(todo))}/{len(todo)}, нет у Yahoo {len(missing)}, "
              f"не докачалось из-за лимита {len(still_limited)}", flush=True)
        time.sleep(PAUSE)
    return out


def _save(opens, closes, missing, path) -> dict:
    out = {
        "open": pd.concat(opens, axis=1).sort_index(),
        "close": pd.concat(closes, axis=1).sort_index(),
        "missing": list(missing),
    }
    for k in ("open", "close"):
        out[k].index = pd.to_datetime(out[k].index).tz_localize(None)
        out[k] = out[k].astype("float32")
    pd.to_pickle(out, path)
    return out


def news_tickers(min_news: int = 100) -> list[str]:
    import news
    vc = news.load()["ticker"].value_counts()
    return vc[vc >= min_news].index.tolist()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    p = load(news_tickers())
    print(f"Готово: цены есть у {p['close'].shape[1]} акций, у Yahoo нет {len(p['missing'])}")
