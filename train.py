"""
V2 trainer: shared MiniLM encoder (frozen) + small heads.

Reads cached vectors from artifacts/cache/ (built by embed.py) plus the raw
jsonl for labels and side-feature text. Trains:
    route    <- ticket vector (+ phrase score)                   -> LogisticRegression
    approve  <- ticket vector + numeric/tier features + phrase    -> LogisticRegression vs HGB
    grounded <- pair features [u,v,|u-v|,u*v] + overlap + diff    -> LogisticRegression vs HGB

All side-feature scorers (phrase score, diff-words score) are fit
out-of-fold (5-fold) during training so the stacked heads never see a score
computed from a model that was trained on that same row - the OOF discipline
asked for in the brief. Calibration (temperature / Platt) and abstention
cutoffs are likewise read off OOF predictions, never off the training fit.

Usage:
    python train.py            # OOF report on train.jsonl, head selection
    python train.py --all      # refit heads on train+dev for the final artifact
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from calibration import (apply_beta_calibration, apply_diagonal_scaling,
                          fit_beta_calibration, fit_diagonal_scaling)
from features import (DiffWordsScorer, PhraseScorer, numeric_features,
                       overlap_features, quantity_features, rule_flags, ticket_text)

SEED = 42
N_FOLDS = 5
QUEUES = ["billing", "technical", "account", "shipping", "general"]
FP_COST = 3.0
ABSTAIN_FRAC = 0.15  # 0.17 left approve at 0.789 dev coverage, under the 80% target
CACHE = Path("artifacts/cache")
ARTIFACTS = Path("artifacts/v2")


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


def ece_binary(probs, y, bins=15):
    probs, y = np.asarray(probs), np.asarray(y)
    tot = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (probs > lo) & (probs <= hi) if lo > 0 else (probs <= hi)
        if not idx.any():
            continue
        tot += (idx.sum() / len(probs)) * abs(probs[idx].mean() - y[idx].mean())
    return tot


def calibrate_binary(head_oof: np.ndarray, y: np.ndarray) -> dict:
    """Fit Platt and Beta calibration on the same OOF head output, keep
    whichever wins on OOF ECE+Brier (both are monotonic in head_oof, so
    this only changes calibration, never the error-ranking/abstention
    score - see calibration.py's docstring)."""
    platt = LogisticRegression(max_iter=1000)
    platt.fit(head_oof.reshape(-1, 1), y)
    cal_platt = platt.predict_proba(head_oof.reshape(-1, 1))[:, 1]

    beta_a, beta_b, beta_c = fit_beta_calibration(head_oof, y)
    cal_beta = apply_beta_calibration(head_oof, beta_a, beta_b, beta_c)

    def score(p):
        return ece_binary(p, y) + np.mean((p - y) ** 2)

    s_platt, s_beta = score(cal_platt), score(cal_beta)
    use_beta = s_beta < s_platt
    print("    calibration  platt ECE+Brier %.4f   |   beta ECE+Brier %.4f   -> using %s"
          % (s_platt, s_beta, "beta" if use_beta else "platt"))
    return {"use_beta": use_beta, "platt": platt,
            "beta_params": (beta_a, beta_b, beta_c),
            "cal_oof": cal_beta if use_beta else cal_platt}


def cost_error(y_true, y_pred, fp_cost=FP_COST):
    c = 0.0
    for y, p in zip(y_true, y_pred):
        if p and not y:
            c += fp_cost
        elif y and not p:
            c += 1.0
    return c / len(y_true)


def load_vectors(splits):
    """Concatenate cached vectors across the given splits, in order."""
    out = {"ticket": [], "draft": [], "kb": []}
    for s in splits:
        for k in out:
            out[k].append(np.load(CACHE / ("%s_%s.npy" % (s, k))))
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


# --------------------------------------------------------------------------
# route
# --------------------------------------------------------------------------

def train_route(rows, vecs):
    y = np.array([QUEUES.index(r["labels"]["route"]) for r in rows])
    texts = [ticket_text(r["state"]) for r in rows]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # phrase/keyword score: a TF-IDF classifier's OOF probabilities, one
    # column per queue, stacked onto the ticket embedding. MiniLM alone
    # underperforms bag-of-words here (0.82 vs 0.84 OOF macro-F1) because
    # routing leans on exact vocabulary (refund, tracking, password, ...)
    # that mean-pooling partly washes out; the phrase score recovers that
    # signal while keeping the shared encoder in the loop.
    phrase_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                   min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=4.0)),
    ])
    phrase_oof = cross_val_predict(phrase_pipe, texts, y, cv=skf, method="predict_proba")
    phrase_pipe.fit(texts, y)

    X_vec_only = vecs["ticket"]
    X_stacked = np.column_stack([vecs["ticket"], phrase_oof])

    best_name, best_f1, best_oof, best_X = None, -1.0, None, None
    for name, X in [("vector_only", X_vec_only), ("vector+phrase", X_stacked)]:
        clf = LogisticRegression(max_iter=3000, C=2.0)
        oof_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
        f1 = macro_f1(y.tolist(), oof_proba.argmax(axis=1).tolist(), list(range(5)))
        print("    route candidate %-14s OOF macro-F1 %.4f" % (name, f1))
        if f1 > best_f1:
            best_name, best_f1, best_oof, best_X = name, f1, oof_proba, X

    use_phrase = best_name == "vector+phrase"
    clf = LogisticRegression(max_iter=3000, C=2.0)
    oof_proba = best_oof
    X = best_X

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
    cal_scalar = np.exp(scaled)
    cal_scalar /= cal_scalar.sum(axis=1, keepdims=True)

    # diagonal (per-class) scaling: compare against the scalar-temperature
    # fit on the same OOF predictions, keep whichever wins on OOF NLL/ECE.
    diag_a, diag_b = fit_diagonal_scaling(oof_proba, y, n_classes=5)
    cal_diag = apply_diagonal_scaling(oof_proba, diag_a, diag_b)

    def nll_of(p):
        return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)))

    def ece_of(p):
        conf = p.max(axis=1)
        corr = (p.argmax(axis=1) == y).astype(float)
        bins = np.linspace(0, 1, 16)
        tot = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            idx = (conf > lo) & (conf <= hi) if lo > 0 else (conf <= hi)
            if not idx.any():
                continue
            tot += (idx.sum() / len(conf)) * abs(conf[idx].mean() - corr[idx].mean())
        return tot

    scalar_nll, diag_nll = nll_of(cal_scalar), nll_of(cal_diag)
    scalar_ece, diag_ece = ece_of(cal_scalar), ece_of(cal_diag)
    print("    route calibration  scalar-T  NLL %.4f  ECE %.4f   |   diagonal  NLL %.4f  ECE %.4f"
          % (scalar_nll, scalar_ece, diag_nll, diag_ece))

    use_diag = diag_nll < scalar_nll
    cal_oof = cal_diag if use_diag else cal_scalar
    conf_oof = cal_oof.max(axis=1)
    abstain_cutoff = float(np.quantile(conf_oof, ABSTAIN_FRAC))

    clf.fit(X, y)
    pred_oof = oof_proba.argmax(axis=1)
    f1 = macro_f1(y.tolist(), pred_oof.tolist(), list(range(5)))
    print("  [route]    using=%s   OOF macro-F1 %.4f   calibration=%s   abstain_cutoff=%.4f"
          % (best_name, f1, "diagonal" if use_diag else "scalar-T", abstain_cutoff))
    return {"clf": clf, "use_phrase": use_phrase, "phrase_pipe": phrase_pipe,
            "use_diag_calibration": use_diag, "temperature": best_T,
            "diag_a": diag_a, "diag_b": diag_b,
            "abstain_cutoff": abstain_cutoff, "oof_f1": f1}


# --------------------------------------------------------------------------
# approve
# --------------------------------------------------------------------------

def train_approve(rows, vecs):
    texts = [r["state"].get("body") or "" for r in rows]
    y = np.array([1 if r["labels"]["approve"] else 0 for r in rows])
    num = np.stack([numeric_features(r["state"]) for r in rows])

    phrase = PhraseScorer(max_features=3000)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    phrase_pipe = Pipeline([("tfidf", phrase.vectorizer), ("clf", phrase.clf)])
    phrase_oof = cross_val_predict(phrase_pipe, texts, y, cv=skf,
                                    method="predict_proba")[:, 1]
    phrase.fit(texts, y)

    # Collapse the 384-d ticket vector into ONE out-of-fold score before the
    # tree sees it. Held-out test (train -> dev): the tree on raw embedding
    # dims scored cost 0.281 / abstention 0.385; on this one score + rule
    # flags, cost 0.256 / abstention 0.440 - the raw dims were drowning out
    # the amount/tier thresholds the tree is actually good at.
    embed_lin = LogisticRegression(max_iter=3000, C=1.0)
    embed_oof = cross_val_predict(embed_lin, vecs["ticket"], y, cv=skf,
                                  method="decision_function")
    embed_lin.fit(vecs["ticket"], y)
    flags = np.stack([rule_flags(r["state"]) for r in rows])

    X = np.column_stack([embed_oof, num, phrase_oof, flags])

    candidates = {
        "logreg": LogisticRegression(max_iter=3000, C=1.0),
        "hgb": HistGradientBoostingClassifier(max_depth=4, max_iter=150,
                                               random_state=SEED),
    }
    best_name, best_cost, best_oof, best_model = None, float("inf"), None, None
    for name, model in candidates.items():
        oof = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        # quick threshold sweep just to compare candidates on cost
        costs = [cost_error(y.tolist(), (oof >= t).tolist())
                 for t in np.linspace(0.05, 0.95, 91)]
        c = min(costs)
        print("    approve candidate %-8s OOF best-cost %.4f" % (name, c))
        if c < best_cost:
            best_name, best_cost, best_oof, best_model = name, c, oof, model

    head_oof = best_oof
    head = best_model
    head.fit(X, y)

    calib = calibrate_binary(head_oof, y)
    cal_oof = calib["cal_oof"]

    best_t, best_cost2 = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 181):
        pred = cal_oof >= t
        c = cost_error(y.tolist(), pred.tolist())
        if c < best_cost2:
            best_cost2, best_t = c, t

    pred_oof = cal_oof >= best_t
    cost = cost_error(y.tolist(), pred_oof.tolist())
    f1 = macro_f1(y.tolist(), pred_oof.astype(int).tolist(), [0, 1])
    exp_cost = np.where(pred_oof, 3.0 * (1 - cal_oof), cal_oof)
    abstain_cutoff = float(np.quantile(exp_cost, 1 - ABSTAIN_FRAC))

    print("  [approve]  head=%s   calibration=%s   OOF cost %.4f   macro-F1 %.4f   threshold=%.3f   abstain_cutoff=%.4f"
          % (best_name, "beta" if calib["use_beta"] else "platt", cost, f1, best_t, abstain_cutoff))
    return {"head_name": best_name, "phrase": phrase, "head": head, "embed_lin": embed_lin,
            "use_beta": calib["use_beta"], "platt": calib["platt"],
            "beta_params": calib["beta_params"],
            "threshold": best_t, "abstain_cutoff": abstain_cutoff, "oof_cost": cost}


# --------------------------------------------------------------------------
# grounded
# --------------------------------------------------------------------------

def pair_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.concatenate([u, v, np.abs(u - v), u * v], axis=-1)


def train_grounded(rows, vecs):
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
    pair = pair_features(vecs["draft"], vecs["kb"])
    # Quantities in the draft that the article doesn't state (normalised, so
    # "two weeks" == "14 days"). Held-out test (train -> dev): errors 68 -> 51,
    # Brier 0.0357 -> 0.0263, abstention 0.798 -> 0.877.
    qty = np.stack([quantity_features(d, k) for d, k in zip(drafts, kbs)])
    X = np.column_stack([pair, overlap, diff_oof, qty])

    # Tried and measured, not kept: a small MLP (64-unit ReLU, early-stopped
    # with a 15-epoch patience) scored worse than logreg on BOTH OOF
    # macro-F1 (0.786 vs 0.804) and risk@80 (0.145 vs 0.137) - confirms this
    # is a representation ceiling (the frozen pair vectors don't separate
    # subtle near-misses), not a head-capacity problem a nonlinear head can
    # fix. Left out rather than kept as a permanently-losing candidate.
    candidates = {
        "logreg": LogisticRegression(max_iter=3000, C=0.5),
        "hgb": HistGradientBoostingClassifier(max_depth=5, max_iter=200,
                                               random_state=SEED),
    }
    best_name, best_f1, best_oof, best_model = None, -1.0, None, None
    for name, model in candidates.items():
        oof = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        f1s = [macro_f1(y.tolist(), (oof >= t).astype(int).tolist(), [0, 1])
               for t in np.linspace(0.05, 0.95, 91)]
        f1 = max(f1s)
        conf = np.maximum(oof, 1 - oof)
        corr = [1.0 if (p > 0.5) == (yy > 0.5) else 0.0 for p, yy in zip(oof, y)]
        order = sorted(range(len(conf)), key=lambda i: -conf[i])
        k = max(1, int(round(len(order) * 0.80)))
        risk80 = 1.0 - sum(corr[i] for i in order[:k]) / k
        print("    grounded candidate %-8s OOF best macro-F1 %.4f   risk@80 %.4f"
              % (name, f1, risk80))
        if f1 > best_f1:
            best_name, best_f1, best_oof, best_model = name, f1, oof, model

    head_oof = best_oof
    head = best_model
    head.fit(X, y)

    calib = calibrate_binary(head_oof, y)
    cal_oof = calib["cal_oof"]

    best_t, best_f1_final = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (cal_oof >= t).astype(int)
        f1 = macro_f1(y.tolist(), pred.tolist(), [0, 1])
        if f1 > best_f1_final:
            best_f1_final, best_t = f1, t

    conf_oof = np.maximum(cal_oof, 1 - cal_oof)
    abstain_cutoff = float(np.quantile(conf_oof, ABSTAIN_FRAC))

    print("  [grounded] head=%s   calibration=%s   OOF macro-F1 %.4f   threshold=%.3f   abstain_cutoff=%.4f"
          % (best_name, "beta" if calib["use_beta"] else "platt", best_f1_final, best_t, abstain_cutoff))
    return {"head_name": best_name, "diff": diff, "head": head,
            "use_beta": calib["use_beta"], "platt": calib["platt"],
            "beta_params": calib["beta_params"],
            "threshold": best_t, "abstain_cutoff": abstain_cutoff,
            "oof_f1": best_f1_final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                     help="fit on train+dev for the final artifact")
    ap.add_argument("--heldout-report", action="store_true",
                     help="diagnostic only, does not save artifacts/v2/model.pkl: "
                          "restrict to the rows listed in "
                          "artifacts/encoder_ft_hf/val_split.json (written by "
                          "finetune_encoder.py) so the OOF numbers printed here "
                          "are honest w.r.t. the ENCODER too, not just the head "
                          "- those rows were never used to fine-tune the encoder. "
                          "Incompatible with --all.")
    args = ap.parse_args()

    if args.heldout_report:
        assert not args.all, "--heldout-report and --all don't mix"
        split_path = Path("artifacts/encoder_ft_hf/val_split.json")
        with open(split_path, encoding="utf-8") as fh:
            split = json.load(fh)
        wanted = set(split["val_request_ids"])
        all_rows = load("data/train.jsonl")
        all_vecs = load_vectors(["train"])
        idx = [i for i, r in enumerate(all_rows) if r["request_id"] in wanted]
        rows = [all_rows[i] for i in idx]
        vecs = {k: v[idx] for k, v in all_vecs.items()}
        print("heldout-report: %d / %d train rows were never seen by the encoder "
              "fine-tuning run (val_frac=%.2f) - OOF numbers below are honest "
              "w.r.t. BOTH the heads and the encoder.\n"
              % (len(rows), len(all_rows), split["val_frac"]))
        train_route(rows, vecs)
        train_approve(rows, vecs)
        train_grounded(rows, vecs)
        print("\n(nothing saved - this is a diagnostic-only report; run plain "
              "`python train.py` / `--all` for the real artifact)")
        return

    splits = ["train", "dev"] if args.all else ["train"]
    rows = []
    for s in splits:
        rows.extend(load("data/%s.jsonl" % s))
    vecs = load_vectors(splits)

    print("training V2 heads on %d rows%s\n" % (len(rows), " (train+dev)" if args.all else ""))
    t0 = time.perf_counter()
    route_art = train_route(rows, vecs)
    approve_art = train_approve(rows, vecs)
    grounded_art = train_grounded(rows, vecs)
    print("\ntotal training time: %.1fs" % (time.perf_counter() - t0))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "model.pkl", "wb") as fh:
        pickle.dump({"route": route_art, "approve": approve_art,
                     "grounded": grounded_art}, fh)
    print("saved artifacts to %s" % ARTIFACTS)


if __name__ == "__main__":
    main()
