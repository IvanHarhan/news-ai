"""FinBERT читает все уникальные заголовки и сохраняет для каждого:
  - «смысловой отпечаток» — 768 чисел (выход pooler, то, на что смотрит его
    собственный классификатор настроения);
  - вероятности настроения: позитив / негатив / нейтрально.

FinBERT здесь только читает, его веса не меняются — учится потом маленькая
сеть поверх (deepmodel.py). Базовый BERT учился на Википедии и книгах до 2018
года, поэтому честнее всего смотреть на тестовые 2019–2020.

Результат: A:\\news_ai_data\\finbert\\emb.npy (float16, строка = уникальный
заголовок) + probs.npy + headlines.pkl. Считается кусками, оборванный запуск
продолжается с места остановки.

Запуск: .venv\\Scripts\\python.exe embed.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

import news

OUT = news.DATA / "finbert"
os.environ.setdefault("HF_HOME", str(news.DATA / "hf"))

import torch  # noqa: E402  (после HF_HOME: модель кладётся на диск A)
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

MODEL = "ProsusAI/finbert"
BATCH = 256
MAX_LEN = 48
CHUNK = 50_000


def unique_headlines() -> pd.Series:
    path = OUT / "headlines.pkl"
    if path.exists():
        return pd.read_pickle(path)
    ds = pd.read_pickle(news.DATA / "dataset.pkl")
    uniq = pd.Series(pd.unique(ds["headline"]))
    OUT.mkdir(parents=True, exist_ok=True)
    uniq.to_pickle(path)
    return uniq


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    heads = unique_headlines()
    n = len(heads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Заголовков: {n}, устройство: {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    t0 = time.time()
    for c0 in range(0, n, CHUNK):
        part_path = OUT / f"part_{c0 // CHUNK:03d}.npz"
        if part_path.exists():
            continue
        texts = heads.iloc[c0:c0 + CHUNK].tolist()
        # Сортировка по длине: в пачке строки похожей длины, меньше пустого добивания.
        order = np.argsort([len(t) for t in texts])
        emb = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
        probs = np.zeros((len(texts), len(labels)), dtype=np.float16)
        with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            for b in range(0, len(texts), BATCH):
                idx = order[b:b + BATCH]
                enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                          max_length=MAX_LEN, return_tensors="pt").to(dev)
                pooled = model.bert(**enc).pooler_output
                logits = model.classifier(pooled)
                emb[idx] = pooled.float().cpu().numpy()
                probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        np.savez(part_path, emb=emb, probs=probs)
        done = min(c0 + CHUNK, n)
        rate = done / (time.time() - t0 + 1e-9)
        print(f"  прочитано {done}/{n}  (~{(n - done) / max(rate, 1):.0f} с осталось)", flush=True)

    parts = sorted(OUT.glob("part_*.npz"))
    emb = np.concatenate([np.load(p)["emb"] for p in parts])
    probs = np.concatenate([np.load(p)["probs"] for p in parts])
    np.save(OUT / "emb.npy", emb)
    np.save(OUT / "probs.npy", probs)
    pd.Series(labels).to_pickle(OUT / "labels.pkl")
    print(f"Готово: {emb.shape}, метки настроения {labels}", flush=True)


if __name__ == "__main__":
    main()
