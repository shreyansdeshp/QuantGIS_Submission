"""
Served model: ONE shared MiniLM encoder + shared featurizer + three small
heads (route / approve / grounded). See README.md "Where the sharing
happens" for the full picture.

Loaded once in __init__ (encoder session, tokenizer, head weights, warm-up).
Each decide call does only the work that decision needs: route and approve
encode the ticket text once; grounded encodes draft and kb_article (two
calls) and never touches the ticket encoder path. No network, no disk I/O
per request.

Every method is wrapped in try/except with a safe fallback that still
returns the exact shape app.py expects - a malformed response fails the
whole schema gate, so a degraded answer beats an exception.
"""

import os

# One ticket per call: extra BLAS/OpenMP threads only add spin-wait and
# memory overhead. Measured with the server pinned to 4 cores, approve p50 was
# 390 ms with default threading (OpenMP threads in HistGradientBoosting.predict
# fighting onnxruntime's threads). Must run before numpy/scikit-learn import.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from calibration import apply_beta_calibration, apply_diagonal_scaling
from encoder import Encoder
from features import (numeric_features, overlap_features, quantity_features,
                      rule_flags, ticket_text)

QUEUES: List[str] = ["billing", "technical", "account", "shipping", "general"]


def _pair_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.concatenate([u, v, np.abs(u - v), u * v], axis=-1)


def _apply_binary_calibration(art: Dict[str, Any], raw: float) -> float:
    if art.get("use_beta"):
        a, b, c = art["beta_params"]
        return float(apply_beta_calibration(np.array([raw]), a, b, c)[0])
    return float(art["platt"].predict_proba([[raw]])[0, 1])


class DecisionModel:
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        artifacts_dir = Path(artifacts_dir)
        self.encoder = Encoder(str(artifacts_dir / "encoder"))
        with open(artifacts_dir / "v2" / "model.pkl", "rb") as fh:
            art = pickle.load(fh)
        self.route_art = art["route"]
        self.approve_art = art["approve"]
        self.grounded_art = art["grounded"]

        self.encoder.warmup()
        dummy = {"subject": "x", "body": "y", "customer_tier": "free",
                 "amount": 1.0, "draft_reply": "a", "kb_article": "b"}
        self.route(dummy)
        self.approve(dummy)
        self.grounded(dummy)

    # -- routing --------------------------------------------------------
    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        try:
            text = ticket_text(state)
            vec = self.encoder.encode(text)
            if self.route_art["use_phrase"]:
                phrase = self.route_art["phrase_pipe"].predict_proba([text])[0]
                x = np.concatenate([vec, phrase]).reshape(1, -1)
            else:
                x = vec.reshape(1, -1)
            proba = self.route_art["clf"].predict_proba(x)[0]
            if self.route_art.get("use_diag_calibration"):
                p = apply_diagonal_scaling(
                    proba, self.route_art["diag_a"], self.route_art["diag_b"])
            else:
                T = self.route_art["temperature"]
                logp = np.log(np.clip(proba, 1e-9, 1.0)) / T
                logp -= logp.max()
                p = np.exp(logp)
                p /= p.sum()
            probs = {q: float(p[i]) for i, q in enumerate(QUEUES)}
            value = QUEUES[int(np.argmax(p))]
            conf = float(p.max())
            abstain = conf < self.route_art["abstain_cutoff"]
            return value, probs, conf, abstain
        except Exception:
            probs = {q: 1.0 / len(QUEUES) for q in QUEUES}
            return QUEUES[0], probs, probs[QUEUES[0]], True

    # -- approval ---------------------------------------------------------
    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        try:
            body = state.get("body") or ""
            vec = self.encoder.encode(ticket_text(state))
            phrase_score = self.approve_art["phrase"].score([body])[0]
            num = numeric_features(state)
            embed_score = self.approve_art["embed_lin"].decision_function(vec.reshape(1, -1))[0]
            X = np.concatenate([[embed_score], num, [phrase_score],
                                rule_flags(state)]).reshape(1, -1)
            raw = self.approve_art["head"].predict_proba(X)[0, 1]
            p = _apply_binary_calibration(self.approve_art, raw)
            value = p >= self.approve_art["threshold"]
            exp_cost = 3.0 * (1 - p) if value else p
            abstain = exp_cost > self.approve_art["abstain_cutoff"]
            return bool(value), p, abstain
        except Exception:
            return False, 0.5, True

    # -- grounding ----------------------------------------------------------
    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        try:
            draft = state.get("draft_reply") or ""
            kb = state.get("kb_article") or ""
            u = self.encoder.encode(draft)
            v = self.encoder.encode(kb)
            pair = _pair_features(u, v)
            diff_score = self.grounded_art["diff"].score([draft], [kb])[0]
            ov = overlap_features(draft, kb)
            X = np.concatenate([pair, ov, [diff_score],
                                quantity_features(draft, kb)]).reshape(1, -1)
            raw = self.grounded_art["head"].predict_proba(X)[0, 1]
            p = _apply_binary_calibration(self.grounded_art, raw)
            value = p >= self.grounded_art["threshold"]
            conf = max(p, 1 - p)
            abstain = conf < self.grounded_art["abstain_cutoff"]
            return bool(value), p, abstain
        except Exception:
            return False, 0.5, True
