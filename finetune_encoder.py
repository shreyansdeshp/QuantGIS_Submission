"""
Fine-tune the shared MiniLM encoder jointly on route + approve + grounded,
instead of keeping it frozen. The frozen encoder underperformed plain TF-IDF
on route and had a weak abstention signal on grounded (see model_card.md in
the parent repo) - both are representation problems a frozen, generic
sentence-similarity model can't fix no matter what head sits on top.

Trains on data/train.jsonl ONLY, with an internal validation split carved
from it (data/dev.jsonl is never touched here, so it stays a valid held-out
set for the final selfcheck comparison).

Disposable torch heads (route/approve/grounded) exist only to shape the
encoder's gradients during this script; the real served heads are the
sklearn ones from train.py, refit afterward on vectors extracted with the
fine-tuned encoder.

Usage:
    python finetune_encoder.py --epochs 4 --lr 2e-5
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUEUES = ["billing", "technical", "account", "shipping", "general"]
SEED = 42
MAX_LENGTH = 128
OUT_DIR = Path("artifacts/encoder_ft_hf")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class TicketPairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        s = r["state"]
        ticket = "%s . %s" % (s.get("subject") or "", s.get("body") or "")
        return {
            "ticket": ticket,
            "draft": s.get("draft_reply") or "",
            "kb": s.get("kb_article") or "",
            "route": QUEUES.index(r["labels"]["route"]),
            "approve": 1.0 if r["labels"]["approve"] else 0.0,
            "grounded": 1.0 if r["labels"]["grounded"] else 0.0,
        }


def collate(batch, tokenizer, device):
    def tok(key):
        enc = tokenizer([b[key] for b in batch], return_tensors="pt",
                         padding=True, truncation=True, max_length=MAX_LENGTH)
        return {k: v.to(device) for k, v in enc.items()}

    return {
        "ticket_enc": tok("ticket"),
        "draft_enc": tok("draft"),
        "kb_enc": tok("kb"),
        "route": torch.tensor([b["route"] for b in batch], device=device),
        "approve": torch.tensor([b["approve"] for b in batch], device=device),
        "grounded": torch.tensor([b["grounded"] for b in batch], device=device),
    }


class MultiTaskModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        dim = encoder.config.hidden_size  # 384
        self.route_head = nn.Linear(dim, 5)
        self.approve_head = nn.Linear(dim, 1)
        self.grounded_head = nn.Sequential(
            nn.Linear(dim * 4, 128), nn.ReLU(), nn.Linear(128, 1))

    def encode(self, enc):
        out = self.encoder(**enc).last_hidden_state
        return mean_pool(out, enc["attention_mask"])

    def forward(self, batch):
        ticket_vec = self.encode(batch["ticket_enc"])
        draft_vec = self.encode(batch["draft_enc"])
        kb_vec = self.encode(batch["kb_enc"])

        route_logits = self.route_head(ticket_vec)
        approve_logits = self.approve_head(ticket_vec).squeeze(-1)

        pair = torch.cat([draft_vec, kb_vec, (draft_vec - kb_vec).abs(),
                           draft_vec * kb_vec], dim=-1)
        grounded_logits = self.grounded_head(pair).squeeze(-1)
        return route_logits, approve_logits, grounded_logits


def run_epoch(model, loader, opt, device, train: bool):
    model.train(train)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    route_correct = route_n = 0
    approve_correct = approve_n = 0
    grounded_correct = grounded_n = 0

    for batch in loader:
        with torch.set_grad_enabled(train):
            route_logits, approve_logits, grounded_logits = model(batch)
            loss = (ce(route_logits, batch["route"])
                    + bce(approve_logits, batch["approve"])
                    + bce(grounded_logits, batch["grounded"]))
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

        total_loss += loss.item() * len(batch["route"])
        route_correct += (route_logits.argmax(-1) == batch["route"]).sum().item()
        route_n += len(batch["route"])
        approve_correct += ((approve_logits > 0).float() == batch["approve"]).sum().item()
        approve_n += len(batch["approve"])
        grounded_correct += ((grounded_logits > 0).float() == batch["grounded"]).sum().item()
        grounded_n += len(batch["grounded"])

    return {
        "loss": total_loss / route_n,
        "route_acc": route_correct / route_n,
        "approve_acc": approve_correct / approve_n,
        "grounded_acc": grounded_correct / grounded_n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=12,
                     help="small by default - this box has a 4GB laptop GPU; "
                          "raise it if you have more VRAM or are on CPU")
    ap.add_argument("--no_grad_checkpoint", action="store_true",
                     help="disable gradient checkpointing (faster, more VRAM)")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.12)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("device:", device)

    random.seed(SEED)
    rows = load("data/train.jsonl")
    random.Random(SEED).shuffle(rows)
    n_val = int(len(rows) * args.val_frac)
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    # Persist which request_ids the encoder NEVER trained on, so a later,
    # honest OOF check (train.py --heldout-split) can restrict itself to
    # exactly those rows. Without this, train.py's own 5-fold CV is
    # optimistic: the encoder already saw every row's label during this
    # fine-tuning pass, so "held out from the head" isn't "held out from
    # the encoder" - see the chat thread that led to this flag.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "val_split.json", "w", encoding="utf-8") as fh:
        json.dump({"val_request_ids": [r["request_id"] for r in val_rows],
                    "val_frac": args.val_frac, "seed": SEED}, fh, indent=2)

    print("train rows: %d   internal-val rows: %d   (data/dev.jsonl untouched)"
          % (len(train_rows), len(val_rows)))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    if device == "cuda" and not args.no_grad_checkpoint:
        encoder.gradient_checkpointing_enable()
        encoder.config.use_cache = False
    model = MultiTaskModel(encoder).to(device)

    train_ds = TicketPairDataset(train_rows)
    val_ds = TicketPairDataset(val_rows)
    collate_fn = lambda b: collate(b, tokenizer, device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                             collate_fn=collate_fn)

    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": list(model.route_head.parameters())
                    + list(model.approve_head.parameters())
                    + list(model.grounded_head.parameters()), "lr": args.head_lr},
    ], weight_decay=0.01)

    best_val_loss, best_state, patience_left = float("inf"), None, args.patience
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, train_loader, opt, device, train=True)
        va = run_epoch(model, val_loader, opt, device, train=False)
        dt = time.perf_counter() - t0
        print("epoch %d (%.1fs)  train loss %.4f  route_acc %.3f approve_acc %.3f grounded_acc %.3f"
              "  |  val loss %.4f  route_acc %.3f approve_acc %.3f grounded_acc %.3f"
              % (epoch, dt, tr["loss"], tr["route_acc"], tr["approve_acc"], tr["grounded_acc"],
                 va["loss"], va["route_acc"], va["approve_acc"], va["grounded_acc"]))

        if va["loss"] < best_val_loss - 1e-4:
            best_val_loss = va["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("early stopping (patience exhausted)")
                break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.encoder.load_state_dict(best_state)
    model.encoder.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("saved fine-tuned encoder to %s (best val loss %.4f)" % (OUT_DIR, best_val_loss))


if __name__ == "__main__":
    main()
