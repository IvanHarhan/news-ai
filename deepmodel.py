"""Уровень 2: маленькая нейросеть поверх того, что прочитал FinBERT.

Вход — 768 «смысловых» чисел заголовка + 3 вероятности настроения (embed.py).
Скрытый слой — HIDDEN нейронов, каждый смотрит на все входы и учится узнавать
свой «тип новости». Выход — вероятность, что акция обгонит рынок.

На numpy, как и уровень 1: так интерфейс может показать каждый вес.
"""
from __future__ import annotations

import numpy as np

import news

FEATURES = news.DATA / "finbert"
HIDDEN = 32
UI_HIDDEN = 12          # сколько скрытых нейронов рисовать (самые сильные)


class Features:
    """Признаки для строк датасета: отпечаток FinBERT + настроение, нормированные
    по средним/разбросу ОБУЧАЮЩИХ строк (тест в нормировке не участвует)."""

    def __init__(self, rows: np.ndarray, head_idx: np.ndarray, emb: np.ndarray, probs: np.ndarray,
                 mean=None, std=None):
        self.rows, self.head_idx, self.emb, self.probs = rows, head_idx, emb, probs
        self.n = len(rows)
        if mean is None:
            s = self.raw(np.random.default_rng(0).choice(self.n, min(50000, self.n), replace=False))
            mean, std = s.mean(axis=0), s.std(axis=0) + 1e-6
        self.mean, self.std = mean.astype(np.float32), std.astype(np.float32)

    def raw(self, idx: np.ndarray) -> np.ndarray:
        h = self.head_idx[self.rows[idx]]
        return np.hstack([self.emb[h].astype(np.float32), self.probs[h].astype(np.float32)])

    def batch(self, idx: np.ndarray) -> np.ndarray:
        return (self.raw(idx) - self.mean) / self.std


class MLP:
    def __init__(self, n_in: int, n_hidden: int = HIDDEN, l2: float = 1e-4, lr: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.p = {
            "W1": (rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))).astype(np.float32),
            "b1": np.zeros(n_hidden, dtype=np.float32),
            "w2": (rng.normal(0, np.sqrt(1 / n_hidden), n_hidden)).astype(np.float32),
            "b2": np.zeros(1, dtype=np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.l2, self.lr, self.t = l2, lr, 0

    def _forward(self, X):
        pre = X @ self.p["W1"] + self.p["b1"]
        h = np.maximum(pre, 0)
        z = h @ self.p["w2"] + self.p["b2"][0]
        return pre, h, 1.0 / (1.0 + np.exp(-z))

    def predict(self, X) -> np.ndarray:
        return self._forward(X)[2].astype(np.float64)

    def step(self, X, y) -> None:
        pre, h, p = self._forward(X)
        err = ((p - y) / len(y)).astype(np.float32)
        dh = np.outer(err, self.p["w2"]) * (pre > 0)
        g = {
            "w2": h.T @ err + self.l2 * self.p["w2"],
            "b2": np.array([err.sum()], dtype=np.float32),
            "W1": X.T @ dh + self.l2 * self.p["W1"],
            "b1": dh.sum(axis=0),
        }
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] * g[k]
            self.p[k] -= self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + eps)

    def snapshot(self, labels: list[str]) -> dict:
        """Для интерфейса: входы (3 настроения + блок смысла) → сильнейшие скрытые → выход."""
        W1, w2 = self.p["W1"], self.p["w2"]
        top = np.argsort(-np.abs(w2))[:UI_HIDDEN]
        emb_dim = W1.shape[0] - len(labels)
        hidden = []
        for j in top:
            w_in = [round(float(W1[emb_dim + i, j]), 4) for i in range(len(labels))]
            w_in.append(round(float(np.linalg.norm(W1[:emb_dim, j]) / np.sqrt(emb_dim)), 4))
            hidden.append({"w_in": w_in, "w_out": round(float(w2[j]), 4)})
        return {"inputs": [*labels, f"смысл ({emb_dim} чисел)"], "hidden": hidden}
