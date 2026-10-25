"""Экзамен 2021–2023 на новостях nasdaq.com (FNSPID) — годы, которых не видели
ни модель, ни её настройщик.

Правила записаны ДО прогона (2026-10-24) и не меняются после:
  СДАН, если 5-дневный вариант после издержек 0.05%/сторона уверенно выше нуля
  (нижняя граница 95% интервала > 0), до издержек в плюсе минимум 2 года из 3,
  и выбор модели лучше 95% случайных выборов.

Две проверки, обе с теми же настройками, что в train.py:
  A. Модель, обученная на Benzinga 2009–2020, без изменений читает новости
     Nasdaq 2021–2023 (понимает ли она чужой источник).
  B. Тот же метод, но обучение на новостях Nasdaq до 2021, экзамен на 2021–2023.
Каждая — на всех акциях и только на крупных (как train.py --liquid).

Запуск (после download_fnspid.ps1): .venv\\Scripts\\python.exe exam.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import dataset
import fnspid
import news
import panel
import textmodel as tm
from stats import block_mean_ci
from train import (BATCH, BLOCK, DECILE, EMBARGO_DAYS, EPOCHS, HORIZONS, LIQUID_MIN_PRICE, LIQUID_TOP,
                   PLACEBO_RUNS, REPORT_EVERY, ROUND_TRIP, SAMPLE, DayBook)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
EXAM_YEARS = [2021, 2022, 2023]
MIN_NEWS_FOR_PRICES = 30
EXAM_CACHE = fnspid.DIR / "exam_dataset.pkl"
NASDAQ_TRAIN_CACHE = fnspid.DIR / "train_dataset.pkl"


def tickers_with_news(df: pd.DataFrame, years) -> list[str]:
    vc = df[df["day"].dt.year.isin(years)]["ticker"].value_counts()
    return vc[vc >= MIN_NEWS_FOR_PRICES].index.tolist()


def exam_dataset(nd: pd.DataFrame) -> pd.DataFrame:
    if EXAM_CACHE.exists():
        return pd.read_pickle(EXAM_CACHE)
    t = tickers_with_news(nd, EXAM_YEARS)
    print(f"Цены 2021–2023 для {len(t)} акций", flush=True)
    pe = panel.load(t, panel.EXAM_START, panel.EXAM_END, panel.EXAM_PANEL)
    exam = dataset.build(nd[nd["day"].dt.year.isin(EXAM_YEARS)], pe)
    exam = exam[exam["year"].isin(EXAM_YEARS)].reset_index(drop=True)
    exam.to_pickle(EXAM_CACHE)
    return exam


def nasdaq_train_dataset(nd: pd.DataFrame) -> pd.DataFrame:
    if NASDAQ_TRAIN_CACHE.exists():
        return pd.read_pickle(NASDAQ_TRAIN_CACHE)
    years = range(2009, 2021)
    t = tickers_with_news(nd, years)
    print(f"Цены 2009–2020 для {len(t)} акций Nasdaq (обучение B)", flush=True)
    pn = panel.load(t)
    ntrain = dataset.build(nd[nd["day"].dt.year.isin(years)], pn)
    ntrain.to_pickle(NASDAQ_TRAIN_CACHE)
    return ntrain


def train_and_score(train_df: pd.DataFrame, test_df: pd.DataFrame, rng, first_test: int = 2021) -> np.ndarray:
    """Метод train.py без изменений: этап 1 ищет момент остановки на последнем
    обучающем годе (first_test-1), этап 2 учится на всём до first_test и
    оценивает тест."""
    both = pd.concat([train_df[["headline", "ticker", "entry", "ex1"]],
                      test_df[["headline", "ticker", "entry", "ex1"]]], ignore_index=True)
    ids, offsets, names = tm.tokenize_all(both["headline"].tolist())
    codes = pd.factorize(both["ticker"])[0].astype(np.int64)
    y = (both["ex1"].to_numpy() > 0).astype(np.float64)
    entry = both["entry"].to_numpy()
    n_tr = len(train_df)
    test_rows = np.arange(n_tr, len(both))

    def before(year):
        cut = np.datetime64(f"{year}-01-01") - np.timedelta64(EMBARGO_DAYS, "D")
        return np.where((entry < cut) & (np.arange(len(both)) < n_tr))[0]

    def fit(tr_rows, ck_rows, steps):
        remap = tm.build_vocab(tr_rows, ids, offsets, codes, len(names))
        tr, ck = tm.Encoded(tr_rows, ids, offsets, remap), tm.Encoded(ck_rows, ids, offsets, remap)
        model = tm.Neuron(int((remap >= 0).sum()))
        s_ck = rng.choice(ck.n, min(SAMPLE, ck.n), replace=False)
        best, best_step, step = np.inf, steps, 0
        while step < steps:
            perm = rng.permutation(tr.n)
            for b in range(0, tr.n, BATCH):
                if step >= steps:
                    break
                idx = perm[b:b + BATCH]
                model.step(tr, idx, y[tr_rows][idx])
                step += 1
                if step % REPORT_EVERY == 0 or step == steps:
                    loss = tm.logloss(model.predict(ck, s_ck), y[ck_rows][s_ck])
                    if loss < best:
                        best, best_step = loss, step
        return model, ck, best_step

    years = both["entry"].dt.year.to_numpy()
    rows_a = before(first_test - 1)
    spe_a = -(-len(rows_a) // BATCH)
    val = np.where((years == first_test - 1) & (np.arange(len(both)) < n_tr))[0]
    _, _, best_step = fit(rows_a, val, EPOCHS * spe_a)
    rows_b = before(first_test)
    steps_b = max(REPORT_EVERY, round(best_step / spe_a * -(-len(rows_b) // BATCH)))
    print(f"  остановка на {best_step / spe_a:.2f} прохода по данным, словарь строится по {len(rows_b)} новостям", flush=True)
    model, te, _ = fit(rows_b, test_rows, steps_b)
    return model.predict(te)


def evaluate(name: str, test: pd.DataFrame, score: np.ndarray, prev_counts: dict, rng) -> list[str]:
    out = [f"── {name} ──"]
    for liquid in (False, True):
        books, days = [], {h: [] for h in HORIZONS}
        for y in EXAM_YEARS:
            m = (test["year"] == y).to_numpy()
            if liquid:
                big = prev_counts[y].index[:LIQUID_TOP]
                m = m & test["ticker"].isin(big).to_numpy() & (test["entry_price"].to_numpy() >= LIQUID_MIN_PRICE)
            if m.sum() == 0:
                continue
            b = DayBook(test[m])
            books.append(b)
            row = [f"  {y}{' крупные' if liquid else ''}: новостей {int(m.sum())}"]
            for h in HORIZONS:
                d = b.run(score[m], col=h)
                days[h].append(d)
                row.append(f"{HORIZONS[h]} {d.mean()*100:+.3f}% → {(d.mean()-ROUND_TRIP)*100:+.3f}% ({len(d)} дн.)"
                           if len(d) else f"{HORIZONS[h]}: нет дней с {DECILE:.0%}-выборкой")
            out.append("  ".join(row))
        for h in HORIZONS:
            g = pd.concat(days[h]).sort_index() if days[h] else pd.Series(dtype=float)
            if len(g) < 30:
                out.append(f"  ИТОГ {HORIZONS[h]}{' крупные' if liquid else ''}: мало дней ({len(g)})")
                continue
            mu, lo, hi = block_mean_ci((g - ROUND_TRIP).to_numpy(), BLOCK[h])
            pl = np.asarray([pd.concat([b.run(rng=rng, col=h) for b in books]).mean() for _ in range(PLACEBO_RUNS)])
            pct = float((pl < g.mean()).mean() * 100)
            years_pos = sum(1 for d in days[h] if len(d) and d.mean() > 0)
            verdict = ""
            if h == "ex5":
                ok = lo > 0 and years_pos >= 2 and pct >= 95
                verdict = "  → СДАН" if ok else "  → НЕ СДАН"
            out.append(f"  ИТОГ {HORIZONS[h]}{' крупные' if liquid else ''}: до {g.mean()*100:+.3f}%, после "
                       f"{mu*100:+.3f}% [{lo*100:+.3f}; {hi*100:+.3f}], лет в плюсе {years_pos}/{len(days[h])}, "
                       f"лучше {pct:.0f}% случайных{verdict}")
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    nd = fnspid.load()
    exam = exam_dataset(nd)
    print(f"Экзамен: {len(exam)} новостей Nasdaq 2021–2023 с ценами", flush=True)
    counts = {y: nd[nd["day"].dt.year == y - 1]["ticker"].value_counts() for y in EXAM_YEARS}
    out = ROOT / "reports" / f"exam_{date.today()}.txt"

    lines = [f"Экзамен 2021–2023 — {date.today()}",
             "Правила записаны до прогона: сдан, если 5 дней после издержек 0.05%/сторона имеет нижнюю границу "
             "интервала > 0, до издержек ≥2 из 3 лет в плюсе, лучше ≥95% случайных.", ""]

    def save(final: bool) -> None:
        report = "\n".join(lines + ([] if final else ["", "(проверка B ещё идёт — качаются цены 2009–2020)"]))
        out.write_text(report, encoding="utf-8")
        print("\n" + report + f"\n\nОтчёт: {out}", flush=True)

    # A первой: ей нужны только цены экзамена, B — ещё и цены 2009–2020.
    print("A: учусь на Benzinga 2009–2020", flush=True)
    bz = pd.read_pickle(news.DATA / "dataset.pkl")
    lines += evaluate("A. Модель Benzinga → новости Nasdaq", exam, train_and_score(bz, exam, rng), counts, rng)
    save(final=False)

    ntrain = nasdaq_train_dataset(nd)
    print(f"B: учусь на Nasdaq 2009–2020 ({len(ntrain)} новостей)", flush=True)
    lines.append("")
    lines += evaluate("B. Метод заново на Nasdaq → Nasdaq", exam, train_and_score(ntrain, exam, rng), counts, rng)
    save(final=True)


if __name__ == "__main__":
    main()
