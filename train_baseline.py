"""
Baseline trainer (TF-IDF + logistic regression + HistGradientBoosting).

This is the fallback kept alongside V2. It trains three independent heads
(no shared encoder) and is intentionally simple: it exists to set a bar V2
has to clear, and to have something that works if V2 runs out of runway.

Usage:
    python train_baseline.py            # train on train.jsonl, report OOF + dev
    python train_baseline.py --all      # refit on train+dev for the final artifact
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from features import (DiffWordsScorer, PhraseScorer, numeric_features,
                       overlap_features, ticket_text)

SEED = 42
N_FOLDS = 5
QUEUES = ["billing", "technical", "account", "shipping", "general"]
FP_COST = 3.0
ABSTAIN_FRAC = 0.17
ARTIFACTS = Path("artifacts/baseline")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def macro_f1(y_true, y_pred, classes):
    out = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(out) / len(out)


def cost_error(y_true, y_pred, fp_cost=FP_COST):
    c = 0.0
    for y, p in zip(y_true, y_pred):
        if p and not y:
            c += fp_cost
        elif y and not p:
            c += 1.0
    return c / len(y_true)


# --------------------------------------------------------------------------

def train_route(rows):
    texts = [ticket_text(r["state"]) for r in rows]
    y = np.array([QUEUES.index(r["labels"]["route"]) for r in rows])
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                   min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=4.0)),
    ])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_proba = cross_val_predict(pipe, texts, y, cv=skf, method="predict_proba")

    # temperature scaling on OOF log-probs
    log_oof = np.log(np.clip(oof_proba, 1e-9, 1.0))
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.3, 5.0, 95):
        scaled = log_oof / T
        scaled -= scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled)
        p /= p.sum(axis=1, keepdims=True)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)))
        if nll < best_nll:
            best_nll, best_T = nll, T

    scaled = log_oof / best_T
    scaled -= scaled.max(axis=1, keepdims=True)
    cal_oof = np.exp(scaled)
    cal_oof /= cal_oof.sum(axis=1, keepdims=True)
    conf_oof = cal_oof.max(axis=1)
    abstain_cutoff = float(np.quantile(conf_oof, ABSTAIN_FRAC))

    pipe.fit(texts, y)
    pred_oof = oof_proba.argmax(axis=1)
    f1 = macro_f1(y.tolist(), pred_oof.tolist(), list(range(5)))
    print("  [route]    OOF macro-F1 %.4f   T=%.3f   abstain_cutoff=%.4f"
          % (f1, best_T, abstain_cutoff))
    return {"pipe": pipe, "temperature": best_T, "abstain_cutoff": abstain_cutoff}


def train_approve(rows):
    texts = [r["state"].get("body") or "" for r in rows]
    y = np.array([1 if r["labels"]["approve"] else 0 for r in rows])
    num = np.stack([numeric_features(r["state"]) for r in rows])

    phrase = PhraseScorer(max_features=3000)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    phrase_pipe = Pipeline([
        ("tfidf", phrase.vectorizer), ("clf", phrase.clf)])
    phrase_oof = cross_val_predict(phrase_pipe, texts, y, cv=skf,
                                    method="predict_proba")[:, 1]
    phrase.fit(texts, y)

    X = np.column_stack([phrase_oof, num])
    head = LogisticRegression(max_iter=2000, C=1.0)
    head_oof = cross_val_predict(head, X, y, cv=skf, method="predict_proba")[:, 1]
    head.fit(X, y)

    # Platt recalibration on OOF.
    platt = LogisticRegression(max_iter=1000)
    platt.fit(head_oof.reshape(-1, 1), y)
    cal_oof = platt.predict_proba(head_oof.reshape(-1, 1))[:, 1]

    # threshold minimising 3:1 cost on OOF
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 181):
        pred = cal_oof >= t
        c = cost_error(y.tolist(), pred.tolist())
        if c < best_cost:
            best_cost, best_t = c, t

    pred_oof = cal_oof >= best_t
    cost = cost_error(y.tolist(), pred_oof.tolist())
    f1 = macro_f1(y.tolist(), pred_oof.astype(int).tolist(), [0, 1])

    # abstain on highest expected cost: 3*(1-p) if predicted yes else p
    exp_cost = np.where(pred_oof, 3.0 * (1 - cal_oof), cal_oof)
    abstain_cutoff = float(np.quantile(exp_cost, 1 - ABSTAIN_FRAC))

    print("  [approve]  OOF cost %.4f   macro-F1 %.4f   threshold=%.3f   abstain_cutoff=%.4f"
          % (cost, f1, best_t, abstain_cutoff))
    return {"phrase": phrase, "head": head, "platt": platt,
            "threshold": best_t, "abstain_cutoff": abstain_cutoff}


def train_grounded(rows):
    drafts = [r["state"].get("draft_reply") or "" for r in rows]
    kbs = [r["state"].get("kb_article") or "" for r in rows]
    y = np.array([1 if r["labels"]["grounded"] else 0 for r in rows])

    diff = DiffWordsScorer(max_features=3000)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    diff_docs = diff._docs(drafts, kbs)
    diff_pipe = Pipeline([("tfidf", diff.vectorizer), ("clf", diff.clf)])
    diff_oof = cross_val_predict(diff_pipe, diff_docs, y, cv=skf,
                                  method="predict_proba")[:, 1]
    diff.fit(drafts, kbs, y)

    overlap = np.stack([overlap_features(d, k) for d, k in zip(drafts, kbs)])
    X = np.column_stack([diff_oof, overlap])
    head = HistGradientBoostingClassifier(max_depth=4, max_iter=150,
                                           random_state=SEED)
    head_oof = cross_val_predict(head, X, y, cv=skf, method="predict_proba")[:, 1]
    head.fit(X, y)

    platt = LogisticRegression(max_iter=1000)
    platt.fit(head_oof.reshape(-1, 1), y)
    cal_oof = platt.predict_proba(head_oof.reshape(-1, 1))[:, 1]

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (cal_oof >= t).astype(int)
        f1 = macro_f1(y.tolist(), pred.tolist(), [0, 1])
        if f1 > best_f1:
            best_f1, best_t = f1, t

    conf_oof = np.maximum(cal_oof, 1 - cal_oof)
    abstain_cutoff = float(np.quantile(conf_oof, ABSTAIN_FRAC))

    print("  [grounded] OOF macro-F1 %.4f   threshold=%.3f   abstain_cutoff=%.4f"
          % (best_f1, best_t, abstain_cutoff))
    return {"diff": diff, "head": head, "platt": platt,
            "threshold": best_t, "abstain_cutoff": abstain_cutoff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                     help="fit on train+dev for the final artifact")
    args = ap.parse_args()

    train_rows = load("data/train.jsonl")
    dev_rows = load("data/dev.jsonl")
    rows = train_rows + dev_rows if args.all else train_rows

    print("training baseline on %d rows%s\n" % (len(rows), " (train+dev)" if args.all else ""))
    route_art = train_route(rows)
    approve_art = train_approve(rows)
    grounded_art = train_grounded(rows)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "model.pkl", "wb") as fh:
        pickle.dump({"route": route_art, "approve": approve_art,
                     "grounded": grounded_art}, fh)
    print("\nsaved artifacts to %s" % ARTIFACTS)


if __name__ == "__main__":
    main()
