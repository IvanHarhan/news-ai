"""Общая статистика для проверок: среднее с доверительным интервалом и
сравнение с плацебо. Подход взят из unibot (modules/paper/metrics.py):
смотрим не «в скольких случаях угадали», а средний результат и его разброс."""
from __future__ import annotations

import numpy as np


def mean_ci(x: np.ndarray, runs: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Среднее и 95% бутстрап-интервал."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(runs, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def block_mean_ci(x: np.ndarray, block: int, runs: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Как mean_ci, но для рядов, где соседние значения зависимы (например,
    5-дневные сделки, открываемые каждый день: соседние на 4/5 одно и то же).
    Тянем не отдельные точки, а куски подряд длиной block."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n_blocks = -(-len(x) // block)
    starts = rng.integers(0, len(x) - block + 1, size=(runs, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(runs, -1)[:, :len(x)]
    boots = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def placebo_pct(real_mean: float, pool: np.ndarray, n: int, runs: int = 10000, seed: int = 1) -> float:
    """В скольких % случайных выборок по n штук из pool среднее НИЖЕ настоящего.
    95+ — настоящий результат необычен; около 50 — ничем не отличается от случая."""
    rng = np.random.default_rng(seed)
    means = rng.choice(np.asarray(pool, dtype=float), size=(runs, n), replace=True).mean(axis=1)
    return float((means < real_mean).mean() * 100)
