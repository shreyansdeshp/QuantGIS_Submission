"""
Baseline DecisionModel: TF-IDF + logistic regression + HistGradientBoosting.

Loads artifacts/baseline/model.pkl produced by train_baseline.py. Kept as the
fallback behind V2 (model.py); not used by app.py directly.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from features import numeric_features, overlap_features, ticket_text

QUEUES: List[str] = ["billing", "technical", "account", "shipping", "general"]


class DecisionModel:
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        path = Path(artifacts_dir) / "baseline" / "model.pkl"
        with open(path, "rb") as fh:
            art = pickle.load(fh)
        self.route_art = art["route"]
        self.approve_art = art["approve"]
        self.grounded_art = art["grounded"]
        # warm up
        dummy = {"subject": "x", "body": "y", "customer_tier": "free",
                 "amount": 1.0, "draft_reply": "a", "kb_article": "b"}
        self.route(dummy)
        self.approve(dummy)
        self.grounded(dummy)

    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        try:
            text = ticket_text(state)
            proba = self.route_art["pipe"].predict_proba([text])[0]
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

    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        try:
            body = state.get("body") or ""
            phrase_score = self.approve_art["phrase"].score([body])[0]
            num = numeric_features(state)
            X = np.concatenate([[phrase_score], num]).reshape(1, -1)
            raw = self.approve_art["head"].predict_proba(X)[0, 1]
            p = float(self.approve_art["platt"].predict_proba([[raw]])[0, 1])
            value = p >= self.approve_art["threshold"]
            exp_cost = 3.0 * (1 - p) if value else p
            abstain = exp_cost > self.approve_art["abstain_cutoff"]
            return bool(value), p, abstain
        except Exception:
            return False, 0.5, True

    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        try:
            draft = state.get("draft_reply") or ""
            kb = state.get("kb_article") or ""
            diff_score = self.grounded_art["diff"].score([draft], [kb])[0]
            ov = overlap_features(draft, kb)
            X = np.concatenate([[diff_score], ov]).reshape(1, -1)
            raw = self.grounded_art["head"].predict_proba(X)[0, 1]
            p = float(self.grounded_art["platt"].predict_proba([[raw]])[0, 1])
            value = p >= self.grounded_art["threshold"]
            conf = max(p, 1 - p)
            abstain = conf < self.grounded_art["abstain_cutoff"]
            return bool(value), p, abstain
        except Exception:
            return False, 0.5, True
