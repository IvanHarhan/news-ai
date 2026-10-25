"""Текст → числа и сама модель (уровень 1).

Модель — один нейрон (логистическая регрессия): у каждого слова и пары слов
свой вес, сумма весов слов заголовка → вероятность «акция обгонит рынок».
Учится градиентным спуском (Adam) мини-пакетами. Всё на numpy, без готовых
библиотек ML — чтобы интерфейс мог показывать каждый шаг.

Словарь строится ТОЛЬКО по годам обучения. И в него не попадают слова,
которые встречаются у малого числа компаний (названия фирм, имена): иначе
модель выучила бы «Apple = росла в 2009–2015», а не смысл новости.
"""
from __future__ import annotations

import re

import numpy as np

_WORD = re.compile(r"[a-z]+")
MIN_DOCS = 30
# 25 пропускало названия, которые попадают в «сборные» заголовки про многие
# акции (huawei, transdigm). 50 — слово должно встречаться хотя бы у полусотни
# разных компаний.
MIN_TICKERS = 50
MAX_VOCAB = 40000


def tokenize_all(headlines) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Все заголовки → (ids, offsets, имена токенов). Токены — слова и пары
    соседних слов. Повтор токена в одном заголовке считается один раз."""
    index: dict[str, int] = {}
    ids: list[int] = []
    offsets = [0]
    for h in headlines:
        words = _WORD.findall(h.lower())
        toks = set(words)
        toks.update(f"{a} {b}" for a, b in zip(words, words[1:]))
        for t in toks:
            i = index.get(t)
            if i is None:
                i = index[t] = len(index)
            ids.append(i)
        offsets.append(len(ids))
    names = [""] * len(index)
    for t, i in index.items():
        names[i] = t
    return np.asarray(ids, dtype=np.int32), np.asarray(offsets, dtype=np.int64), names


def rows_to_entries(rows: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Для выбранных строк: (позиции их токенов в общем массиве, номер строки внутри выборки)."""
    starts, ends = offsets[rows], offsets[rows + 1]
    lens = ends - starts
    pos = np.repeat(starts - np.cumsum(np.r_[0, lens[:-1]]), lens) + np.arange(lens.sum())
    return pos, np.repeat(np.arange(len(rows)), lens)


def build_vocab(train_rows, ids, offsets, ticker_codes, n_tokens) -> np.ndarray:
    """Глобальный id токена → id в словаре фолда (−1 = не берём)."""
    pos, r = rows_to_entries(train_rows, offsets)
    tok = ids[pos]
    docs = np.bincount(tok, minlength=n_tokens)
    pair = np.unique(tok.astype(np.int64) * (ticker_codes.max() + 1) + ticker_codes[train_rows][r])
    n_tick = np.bincount((pair // (ticker_codes.max() + 1)).astype(np.int64), minlength=n_tokens)
    keep = np.where((docs >= MIN_DOCS) & (n_tick >= MIN_TICKERS))[0]
    keep = keep[np.argsort(-docs[keep])][:MAX_VOCAB]
    remap = np.full(n_tokens, -1, dtype=np.int32)
    remap[keep] = np.arange(len(keep), dtype=np.int32)
    return remap


class Encoded:
    """Строки выборки в виде «плоских» индексов словаря — готово для модели."""

    def __init__(self, rows, ids, offsets, remap):
        pos, r = rows_to_entries(rows, offsets)
        v = remap[ids[pos]]
        keep = v >= 0
        self.tok, self.row = v[keep], r[keep]
        self.n = len(rows)
        # rows_to_entries отдаёт токены уже по порядку строк — сортировать не нужно.
        cnt = np.bincount(self.row, minlength=self.n)
        self.scale = (1.0 / np.sqrt(np.maximum(cnt, 1))).astype(np.float32)
        self.start = np.r_[0, np.cumsum(cnt)]

    def batch(self, idx: np.ndarray):
        """Подвыборка строк idx: (токены, номер строки в пакете, масштаб)."""
        s, e = self.start[idx], self.start[idx + 1]
        lens = e - s
        pos = np.repeat(s - np.cumsum(np.r_[0, lens[:-1]]), lens) + np.arange(lens.sum())
        return self.tok[pos], np.repeat(np.arange(len(idx)), lens), self.scale[idx]


class Neuron:
    def __init__(self, vocab_size: int, l2: float = 1e-6, lr: float = 0.02):
        self.w = np.zeros(vocab_size, dtype=np.float64)
        self.b = 0.0
        self.l2, self.lr = l2, lr
        self._m = np.zeros_like(self.w)
        self._v = np.zeros_like(self.w)
        self._mb = self._vb = 0.0
        self._t = 0

    def logits(self, tok, row, scale, n):
        return self.b + np.bincount(row, weights=self.w[tok], minlength=n) * scale

    def predict(self, enc: Encoded, idx=None) -> np.ndarray:
        idx = np.arange(enc.n) if idx is None else idx
        tok, row, scale = enc.batch(idx)
        return 1.0 / (1.0 + np.exp(-self.logits(tok, row, scale, len(idx))))

    def step(self, enc: Encoded, idx: np.ndarray, y: np.ndarray) -> None:
        tok, row, scale = enc.batch(idx)
        p = 1.0 / (1.0 + np.exp(-self.logits(tok, row, scale, len(idx))))
        err = (p - y) / len(idx)
        g = np.bincount(tok, weights=(err * scale)[row], minlength=len(self.w)) + self.l2 * self.w
        gb = err.sum()
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        self._m = b1 * self._m + (1 - b1) * g
        self._v = b2 * self._v + (1 - b2) * g * g
        self._mb = b1 * self._mb + (1 - b1) * gb
        self._vb = b2 * self._vb + (1 - b2) * gb * gb
        c1, c2 = 1 - b1 ** self._t, 1 - b2 ** self._t
        self.w -= self.lr * (self._m / c1) / (np.sqrt(self._v / c2) + eps)
        self.b -= self.lr * (self._mb / c1) / (np.sqrt(self._vb / c2) + eps)


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Вероятность, что случайный «вырос» получил оценку выше случайного «упал». 0.5 = монетка."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
