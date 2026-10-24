"""
Compute and cache MiniLM vectors for train/dev so head experiments (train.py)
run in seconds instead of re-encoding text every time.

Saves, under artifacts/cache/:
    {split}_ticket.npy   (N, 384)  mean-pooled "subject . body"
    {split}_draft.npy    (N, 384)  mean-pooled draft_reply
    {split}_kb.npy       (N, 384)  mean-pooled kb_article

Run once after export_encoder.py:
    python embed.py
"""

import json
import time
from pathlib import Path

import numpy as np

from encoder import Encoder
from features import ticket_text

CACHE = Path("artifacts/cache")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def embed_split(enc: Encoder, rows, split: str, batch_size: int = 64):
    tickets = [ticket_text(r["state"]) for r in rows]
    drafts = [r["state"].get("draft_reply") or "" for r in rows]
    kbs = [r["state"].get("kb_article") or "" for r in rows]

    out = {}
    for name, texts in [("ticket", tickets), ("draft", drafts), ("kb", kbs)]:
        vecs = []
        t0 = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            vecs.append(enc.encode_batch(texts[i:i + batch_size]))
        vecs = np.concatenate(vecs, axis=0)
        out[name] = vecs
        print("  %-6s %-6s %s  %.1fs" % (split, name, vecs.shape, time.perf_counter() - t0))

    CACHE.mkdir(parents=True, exist_ok=True)
    for name, vecs in out.items():
        np.save(CACHE / ("%s_%s.npy" % (split, name)), vecs)
    return out


def main():
    enc = Encoder()
    for split in ("train", "dev"):
        rows = load("data/%s.jsonl" % split)
        embed_split(enc, rows, split)
    print("\nsaved caches to %s" % CACHE)


if __name__ == "__main__":
    main()
