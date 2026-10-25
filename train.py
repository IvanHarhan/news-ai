"""Обучение модели новостей с проверкой «только на прошлом» (walk-forward).

Фолды: учимся на всех годах ДО тестового, проверяем на тестовом.
  2009–2013 → 2014, 2009–2014 → 2015, …, 2009–2019 → 2020.
Между обучением и тестом — зазор в 10 дней (5-дневная доходность новости из
конца декабря иначе залезала бы в январь теста).

Как оцениваем в деньгах. Каждый торговый день тестового года берём акции,
по которым вышли новости: 10% с самой высокой оценкой модели — в лонг, 10% с
самой низкой — в шорт, на 1 день (открытие → закрытие) или на 5 дней.
Издержки — COST_PER_SIDE на каждую сторону каждой сделки. Плацебо — то же
самое, но акции выбраны случайно.

Прогресс пишется в ui/progress.json — его рисует ui/index.html.
Запуск: .venv\\Scripts\\python.exe train.py   (или start.bat — сразу с интерфейсом)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import dataset
import deepmodel as deep
import news
import panel
import textmodel as tm
from stats import block_mean_ci

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
PROGRESS = ROOT / "ui" / "progress.json"
DATASET_CACHE = news.DATA / "dataset.pkl"

TEST_YEARS = list(range(2014, 2021))
EMBARGO_DAYS = 10
EPOCHS = 3
BATCH = 4096
REPORT_EVERY = 40          # шагов между замерами для интерфейса
SAMPLE = 40000             # на скольких примерах мерить ошибку в замере
MIN_NEWS_PER_DAY = 20
# --liquid: проверка только на крупных акциях, где издержки 0.05% реальны.
# Объёмов торгов в данных нет, поэтому «крупная» = в топ-300 по числу новостей
# за ПРОШЛЫЙ год (аналитики пишут о крупных компаниях) и цена входа от $10.
# Учится модель всё равно на всех новостях.
LIQUID = "--liquid" in sys.argv
LIQUID_TOP = 300
LIQUID_MIN_PRICE = 10.0
# --finbert: уровень 2 — вместо слов отпечаток FinBERT (сначала embed.py).
FINBERT = "--finbert" in sys.argv
SUFFIX = ("_finbert" if FINBERT else "") + ("_liquid" if LIQUID else "")
DECILE = 0.10
COST_PER_SIDE = 0.0005     # 0.05%: дешёвый брокер + спред ликвидной акции
ROUND_TRIP = 4 * COST_PER_SIDE   # лонг и шорт, каждый — вход и выход
# Горизонты оценки, заданы ДО полного прогона: держать день или 5 дней.
# Учится модель на знаке дневного движения.
HORIZONS = {"ex1": "1 день", "ex5": "5 дней"}
BLOCK = {"ex1": 5, "ex5": 10}    # длина куска для интервала (см. stats.block_mean_ci)
PLACEBO_RUNS = 200
NEURON_WORDS = 14          # слов с каждой стороны на схеме нейрона


class Progress:
    def __init__(self):
        self.state = {"status": "подготовка данных", "folds": [], "current": None, "final": None}

    def save(self) -> None:
        PROGRESS.parent.mkdir(exist_ok=True)
        tmp = PROGRESS.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, PROGRESS)


def load_dataset(prog: Progress) -> pd.DataFrame:
    if DATASET_CACHE.exists():
        return pd.read_pickle(DATASET_CACHE)
    prog.state["status"] = "читаю новости"; prog.save()
    nd = news.load()
    tickers = panel.news_tickers()
    prog.state["status"] = f"качаю цены ({len(tickers)} акций)"; prog.save()
    pn = panel.load(tickers)
    prog.state["status"] = "размечаю новости ценами"; prog.save()
    df = dataset.build(nd, pn)
    df.to_pickle(DATASET_CACHE)
    return df


class DayBook:
    """Новости тестового года, сгруппированные «день входа × акция».

    Несколько новостей про одну акцию в один день — это одна сделка, а не
    несколько: оценки модели по ним усредняются. Группировка считается один
    раз, а прогоны (модель, сотни плацебо) только пересортировывают."""

    def __init__(self, test: pd.DataFrame):
        t = test[["entry", "ticker", "ex1", "ex5"]].reset_index(drop=True)
        grp = t.groupby(["entry", "ticker"], sort=True)
        self.code = grp.ngroup().to_numpy()
        first = grp[["ex1", "ex5"]].first()
        self.ex = {c: first[c].to_numpy() for c in ("ex1", "ex5")}
        days = first.index.get_level_values(0).to_numpy()
        self.days, starts = np.unique(days, return_index=True)
        self.bounds = np.r_[starts, len(days)]
        self.counts = np.bincount(self.code)

    def run(self, score: np.ndarray | None = None, rng=None, col: str = "ex1") -> pd.Series:
        """Лонг-шорт по дням входа до издержек. col — горизонт (ex1 день, ex5 пять дней).
        rng вместо score — случайный выбор (плацебо)."""
        s_all = None if score is None else np.bincount(self.code, weights=score) / self.counts
        vals, when = [], []
        for i, day in enumerate(self.days):
            a, b = self.bounds[i], self.bounds[i + 1]
            if b - a < MIN_NEWS_PER_DAY:
                continue
            s = rng.random(b - a) if rng is not None else s_all[a:b]
            k = max(1, int((b - a) * DECILE))
            order = np.argsort(s)
            ex = self.ex[col][a:b]
            vals.append(ex[order[-k:]].mean() - ex[order[:k]].mean())
            when.append(day)
        return pd.Series(vals, index=pd.to_datetime(when))


def neuron_snapshot(model: tm.Neuron, names: np.ndarray) -> list:
    w = model.w
    top = np.argsort(-w)[:NEURON_WORDS]
    bot = np.argsort(w)[:NEURON_WORDS]
    return [[str(names[i]), round(float(w[i]), 4)] for i in np.r_[top, bot]]


def main() -> None:
    prog = Progress()
    prog.save()
    df = load_dataset(prog)
    print(f"Примеров: {len(df)}, {df['entry'].min().date()} → {df['entry'].max().date()}", flush=True)

    if not FINBERT:
        prog.state["status"] = "разбираю заголовки на слова"; prog.save()
        ids, offsets, names = tm.tokenize_all(df["headline"].tolist())
        names = np.asarray(names, dtype=object)
        ticker_codes = pd.factorize(df["ticker"])[0].astype(np.int64)
    y_all = (df["ex1"].to_numpy() > 0).astype(np.float64)
    entry = df["entry"].to_numpy()
    rng = np.random.default_rng(0)
    all_days, books = {h: [] for h in HORIZONS}, []

    years = df["year"].to_numpy()

    def rows_before(year: int) -> np.ndarray:
        t0 = np.datetime64(f"{year}-01-01") - np.timedelta64(EMBARGO_DAYS, "D")
        return np.where(entry < t0)[0]

    if FINBERT:
        prog.state["status"] = "загружаю то, что прочитал FinBERT"; prog.save()
        heads = pd.read_pickle(deep.FEATURES / "headlines.pkl")
        head_idx = pd.Index(heads).get_indexer(df["headline"])
        if (head_idx < 0).any():
            raise SystemExit("FinBERT прочитал не все заголовки — запусти embed.py заново")
        emb = np.load(deep.FEATURES / "emb.npy")
        probs = np.load(deep.FEATURES / "probs.npy")
        sent_labels = {"positive": "позитив", "negative": "негатив", "neutral": "нейтрально"}
        labels = [sent_labels.get(x, x) for x in pd.read_pickle(deep.FEATURES / "labels.pkl")]

    class WordsBackend:
        """Уровень 1: один нейрон, вес на каждое слово."""
        kind = "words"

        def __init__(self, train_rows, check_rows):
            remap = tm.build_vocab(train_rows, ids, offsets, ticker_codes, len(names))
            self.vocab = np.empty(int((remap >= 0).sum()), dtype=object)
            self.vocab[remap[remap >= 0]] = names[remap >= 0]
            self.tr = tm.Encoded(train_rows, ids, offsets, remap)
            self.ck = tm.Encoded(check_rows, ids, offsets, remap)
            self.model = tm.Neuron(len(self.vocab))
            self.size = len(self.vocab)

        def step(self, idx, y):
            self.model.step(self.tr, idx, y)

        def predict(self, which, idx=None):
            return self.model.predict(self.tr if which == "train" else self.ck, idx)

        def snapshot(self):
            return {"neuron": neuron_snapshot(self.model, self.vocab)}

    class FinbertBackend:
        """Уровень 2: отпечаток FinBERT → скрытый слой → прогноз."""
        kind = "finbert"

        def __init__(self, train_rows, check_rows):
            self.tr = deep.Features(train_rows, head_idx, emb, probs)
            self.ck = deep.Features(check_rows, head_idx, emb, probs, self.tr.mean, self.tr.std)
            self.size = emb.shape[1] + probs.shape[1]
            self.model = deep.MLP(self.size)

        def step(self, idx, y):
            self.model.step(self.tr.batch(idx), y.astype(np.float32))

        def predict(self, which, idx=None):
            f = self.tr if which == "train" else self.ck
            idx = np.arange(f.n) if idx is None else idx
            return np.concatenate([self.model.predict(f.batch(idx[i:i + 50000]))
                                   for i in range(0, len(idx), 50000)])

        def snapshot(self):
            return {"net": self.model.snapshot(labels)}

    Backend = FinbertBackend if FINBERT else WordsBackend

    def fit(train_rows, check_rows, steps, cur, phase):
        """Учит модель steps шагов, по ходу меряет ошибку на check_rows.
        Возвращает (модель, шаг с наименьшей ошибкой на check_rows)."""
        m = Backend(train_rows, check_rows)
        ytr, yck = y_all[train_rows], y_all[check_rows]
        s_tr = rng.choice(len(train_rows), min(SAMPLE, len(train_rows)), replace=False)
        s_ck = rng.choice(len(check_rows), min(SAMPLE, len(check_rows)), replace=False)
        cur.update(phase=phase, model=m.kind, n_train=int(len(train_rows)), vocab=int(m.size),
                   steps=[], train_loss=[], test_loss=[], test_auc=[], total_steps=int(steps))
        best, best_step, step = np.inf, steps, 0
        while step < steps:
            perm = rng.permutation(len(train_rows))
            for b in range(0, len(train_rows), BATCH):
                if step >= steps:
                    break
                idx = perm[b:b + BATCH]
                m.step(idx, ytr[idx])
                step += 1
                if step % REPORT_EVERY == 0 or step == steps:
                    p_tr, p_ck = m.predict("train", s_tr), m.predict("check", s_ck)
                    loss = tm.logloss(p_ck, yck[s_ck])
                    if loss < best:
                        best, best_step = loss, step
                    cur["step"] = step
                    cur["steps"].append(step)
                    cur["train_loss"].append(round(tm.logloss(p_tr, ytr[s_tr]), 5))
                    cur["test_loss"].append(round(loss, 5))
                    cur["test_auc"].append(round(tm.auc(p_ck, yck[s_ck]), 4))
                    cur.update(m.snapshot())
                    prog.save()
        return m, best_step

    for fi, ty in enumerate(TEST_YEARS):
        test_rows = np.where(years == ty)[0]
        if LIQUID:
            big = df["ticker"][years == ty - 1].value_counts().index[:LIQUID_TOP]
            keep = (df["ticker"].iloc[test_rows].isin(big).to_numpy()
                    & (df["entry_price"].iloc[test_rows].to_numpy() >= LIQUID_MIN_PRICE))
            test_rows = test_rows[keep]
        cur = {"fold": fi + 1, "folds_total": len(TEST_YEARS), "train": f"2009–{ty - 1}",
               "test": str(ty), "val": str(ty - 1), "n_test": int(len(test_rows)), "neuron": []}
        prog.state.update(status="обучение", current=cur)

        # Этап 1: учимся на годах до ty-1, проверяем на ty-1 — ищем шаг, после
        # которого модель начинает зубрить. Тестовый год здесь не участвует.
        rows_a = rows_before(ty - 1)
        spe_a = -(-len(rows_a) // BATCH)
        _, best_step = fit(rows_a, np.where(years == ty - 1)[0], EPOCHS * spe_a, cur, "stop")

        # Этап 2: учимся на всех годах до тестового столько же «проходов по
        # данным», сколько оказалось лучшим на этапе 1, и только теперь смотрим тест.
        rows_b = rows_before(ty)
        steps_b = max(REPORT_EVERY, round(best_step / spe_a * -(-len(rows_b) // BATCH)))
        cur["best_epochs"] = round(best_step / spe_a, 2)
        m, _ = fit(rows_b, test_rows, steps_b, cur, "final")
        yte = y_all[test_rows]

        score = m.predict("check")
        book = DayBook(df.iloc[test_rows])
        books.append(book)
        res = {"test": str(ty), "auc": round(tm.auc(score, yte), 4), **m.snapshot()}
        for h in HORIZONS:
            d = book.run(score, col=h)
            all_days[h].append(d)
            res[h] = {"n": int(len(d)), "gross": round(float(d.mean()) * 100, 3),
                      "net": round(float(d.mean() - ROUND_TRIP) * 100, 3)}
        prog.state["folds"].append(res)
        prog.save()
        print(f"{ty}: AUC {res['auc']:.4f}  " + "  ".join(
            f"{HORIZONS[h]}: {res[h]['gross']:+.3f}% → {res[h]['net']:+.3f}%" for h in HORIZONS), flush=True)

    # ── Итог по всем тестовым годам ──
    # Плацебо: те же дни, но акции из тех же новостей выбраны случайно. Модель
    # не пересчитывается — меряем, насколько её выбор лучше случайного.
    prog.state["status"] = "плацебо"; prog.save()
    final, lines_h = {}, []
    pd.to_pickle(all_days, news.DATA / f"last_run_days{SUFFIX}.pkl")
    for h in HORIZONS:
        gross = pd.concat(all_days[h]).sort_index()
        # Соседние дни зависимы (5-дневные сделки перекрываются на 4/5, и у
        # рынка бывают полосы) — интервал считается кусками подряд, иначе он
        # получается обманчиво узким.
        m, lo, hi = block_mean_ci((gross - ROUND_TRIP).to_numpy(), BLOCK[h])
        placebo = np.asarray([pd.concat([b.run(rng=rng, col=h) for b in books]).mean()
                              for _ in range(PLACEBO_RUNS)])
        pct = float((placebo < gross.mean()).mean() * 100)
        final[h] = {
            "n": int(len(gross)), "gross_mean": round(float(gross.mean()) * 100, 4),
            "net_mean": round(m * 100, 4), "net_lo": round(lo * 100, 4), "net_hi": round(hi * 100, 4),
            "placebo_pct": round(pct, 1),
        }
        lines_h.append(
            f"{HORIZONS[h]}: {len(gross)} входов, за сделку до издержек {gross.mean()*100:+.3f}%, "
            f"после {m*100:+.3f}% [{lo*100:+.3f}; {hi*100:+.3f}], лучше {pct:.0f}% случайных выборов")
        lines_h.append("   при издержках " + ", ".join(
            f"{c*100:.2f}%/сторона: {(gross.mean() - 4 * c)*100:+.3f}%" for c in (0.001, 0.002)))
    # Кривая счёта — дневной вариант: 50% капитала на лонг и 50% на шорт.
    net1 = pd.concat(all_days["ex1"]).sort_index() - ROUND_TRIP
    equity = np.cumprod(1 + net1.to_numpy() * 0.5)
    final["equity_dates"] = [d.strftime("%Y-%m-%d") for d in net1.index[::5]]
    final["equity"] = [round(float(x), 4) for x in equity[::5]]
    prog.state.update(status="готово", final=final)
    prog.save()

    lines = [
        f"Модель новостей, {'уровень 2 (FinBERT + скрытый слой)' if FINBERT else 'уровень 1 (один нейрон)'} — {date.today()}",
        f"Примеров: {len(df)}; тестовые годы {TEST_YEARS[0]}–{TEST_YEARS[-1]}, обучение только на прошлом.",
        (f"ПРОВЕРКА ТОЛЬКО НА КРУПНЫХ: топ-{LIQUID_TOP} по числу новостей за прошлый год, цена от ${LIQUID_MIN_PRICE:.0f}."
         if LIQUID else "Проверка на всех акциях с новостями."),
        f"Сделки: в каждый день входа лонг 10% лучших оценок / шорт 10% худших. "
        f"Издержки {COST_PER_SIDE*100:.2f}% на сторону, {ROUND_TRIP*100:.2f}% на лонг+шорт туда-обратно.",
        "",
    ] + [f"{r['test']}: AUC {r['auc']:.4f}  " + "  ".join(
        f"{HORIZONS[h]} {r[h]['gross']:+.3f}% → {r[h]['net']:+.3f}%" for h in HORIZONS)
        for r in prog.state["folds"]] + ["", "Итог:"] + lines_h
    report = "\n".join(lines)
    print("\n" + report)
    out = ROOT / "reports" / f"news_model_{date.today()}{SUFFIX}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"Время: {(time.time() - t) / 60:.1f} мин")
