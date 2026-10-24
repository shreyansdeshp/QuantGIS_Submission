"""
Shared featurizer module.

Used by the baseline (train_baseline.py / model_baseline.py), by the V2
embedding cache (embed.py) and by V2 training/serving (train.py / model.py).
Nothing in here is decision-specific logic - it only turns raw `state` dicts
into numeric features. Keeping it in one module is part of the "one shared
featurizer" constraint: route/approve/grounded never compute their numeric
features differently.
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TIERS = ["free", "pro", "enterprise"]
# Rough per-tier auto-approve ceilings called out in the problem statement.
TIER_CEILING = {"free": 31.0, "pro": 125.0, "enterprise": 500.0}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


# --------------------------------------------------------------------------
# numeric / tier features (route, approve)
# --------------------------------------------------------------------------

NUMERIC_FEATURE_NAMES = (
    ["amount", "log_amount"]
    + ["tier_%s" % t for t in TIERS]
    + ["ceiling_ratio_%s" % t for t in TIERS]
    + ["over_ceiling_%s" % t for t in TIERS]
)


def numeric_features(state: Dict[str, Any]) -> np.ndarray:
    amount = float(state.get("amount") or 0.0)
    tier = state.get("customer_tier") or "free"
    if tier not in TIERS:
        tier = "free"
    log_amount = float(np.log1p(max(amount, 0.0)))
    tier_onehot = [1.0 if tier == t else 0.0 for t in TIERS]
    ceiling = TIER_CEILING[tier]
    ceiling_ratio = [amount / TIER_CEILING[t] for t in TIERS]
    over_ceiling = [1.0 if amount > TIER_CEILING[t] else 0.0 for t in TIERS]
    return np.array([amount, log_amount] + tier_onehot + ceiling_ratio + over_ceiling,
                     dtype=np.float32)


# --------------------------------------------------------------------------
# overlap features (grounded: draft vs kb_article)
# --------------------------------------------------------------------------

OVERLAP_FEATURE_NAMES = ["jaccard", "draft_len", "kb_len", "len_ratio",
                         "overlap_count", "diff_draft_count", "diff_kb_count",
                         "diff_draft_ratio", "diff_kb_ratio"]


def overlap_features(draft: str, kb: str) -> np.ndarray:
    d_words = set(tokenize(draft))
    k_words = set(tokenize(kb))
    inter = d_words & k_words
    union = d_words | k_words
    jaccard = len(inter) / len(union) if union else 0.0
    diff_draft = d_words - k_words
    diff_kb = k_words - d_words
    draft_len = len(d_words)
    kb_len = len(k_words)
    len_ratio = draft_len / kb_len if kb_len else 0.0
    diff_draft_ratio = len(diff_draft) / draft_len if draft_len else 0.0
    diff_kb_ratio = len(diff_kb) / kb_len if kb_len else 0.0
    return np.array([jaccard, draft_len, kb_len, len_ratio, len(inter),
                      len(diff_draft), len(diff_kb), diff_draft_ratio,
                      diff_kb_ratio], dtype=np.float32)


# --------------------------------------------------------------------------
# phrase score (approve): TF-IDF over the ticket body -> logistic score.
# Fit OOF during training to avoid leakage; cached score is one scalar.
# --------------------------------------------------------------------------

class PhraseScorer:
    def __init__(self, max_features: int = 4000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), min_df=2,
            sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=2000, C=1.0)

    def fit(self, texts: List[str], y: np.ndarray) -> "PhraseScorer":
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, y)
        return self

    def score(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)


# --------------------------------------------------------------------------
# difference-words score (grounded): words only in the draft (D_ prefix) or
# only in the kb article (K_ prefix), TF-IDF -> logistic score. This is the
# single feature that does the most for grounded: an unsupported claim shows
# up as draft-only vocabulary that never appears in the article.
# --------------------------------------------------------------------------

class DiffWordsScorer:
    def __init__(self, max_features: int = 4000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 1), min_df=2,
            token_pattern=r"[A-Za-z_]+")
        self.clf = LogisticRegression(max_iter=2000, C=1.0)

    @staticmethod
    def _diff_doc(draft: str, kb: str) -> str:
        d_words = set(tokenize(draft))
        k_words = set(tokenize(kb))
        d_only = ["D_" + w for w in (d_words - k_words)]
        k_only = ["K_" + w for w in (k_words - d_words)]
        return " ".join(d_only + k_only)

    def _docs(self, drafts: List[str], kbs: List[str]) -> List[str]:
        return [self._diff_doc(d, k) for d, k in zip(drafts, kbs)]

    def fit(self, drafts: List[str], kbs: List[str], y: np.ndarray) -> "DiffWordsScorer":
        docs = self._docs(drafts, kbs)
        X = self.vectorizer.fit_transform(docs)
        self.clf.fit(X, y)
        return self

    def score(self, drafts: List[str], kbs: List[str]) -> np.ndarray:
        docs = self._docs(drafts, kbs)
        X = self.vectorizer.transform(docs)
        # p(grounded=True); diff-words indicate NOT grounded, so this is
        # P(y=1) from a classifier trained directly on the grounded label.
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)


def ticket_text(state: Dict[str, Any]) -> str:
    return "%s . %s" % (state.get("subject") or "", state.get("body") or "")


# --------------------------------------------------------------------------
# hard-rule flags (approve). Found in train.jsonl: tickets containing one of
# these sentences, or whose request is one of these types, are never approved
# (0% approval across 60-70 train tickets each).
# --------------------------------------------------------------------------

RISK_PHRASES = ["dispute a charge with my bank", "only signed up recently",
                "outside your usual policy", "raised something similar not long ago"]
NEVER_APPROVE_PHRASES = ["integration partnership", "reseller or affiliate",
                         "erasure of my personal data", "close the account for good",
                         "partnership enquiries", "permanently deleted",
                         "lost the phone with my authenticator", "disable two-factor",
                         "2fa codes are being rejected"]
RULE_FEATURE_NAMES = ["risk_sentence", "never_approve_request"]


def rule_flags(state: Dict[str, Any]) -> np.ndarray:
    body = (state.get("body") or "").lower()
    return np.array([float(any(p in body for p in RISK_PHRASES)),
                     float(any(p in body for p in NEVER_APPROVE_PHRASES))], dtype=np.float32)


# --------------------------------------------------------------------------
# quantity support (grounded). Most remaining grounded errors are a number or
# duration in the draft that the KB article doesn't state, written in another
# form ("a year" vs "six months", "sixty minutes" vs "half an hour"). Normalise
# every quantity to (unit family, value) so "14 days" == "fourteen days" ==
# "two weeks", then count draft quantities the article doesn't support.
# --------------------------------------------------------------------------

_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
              "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
              "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
              "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100,
              "a": 1, "an": 1, "single": 1}
_UNIT = {"minute": ("min", 1), "min": ("min", 1), "hour": ("min", 60), "hr": ("min", 60),
         "day": ("day", 1), "week": ("day", 7), "fortnight": ("day", 14),
         "month": ("month", 1), "year": ("month", 12),
         "business day": ("bday", 1), "working day": ("bday", 1),
         "attempt": ("attempt", 1), "try": ("attempt", 1), "tries": ("attempt", 1),
         "seat": ("seat", 1), "percent": ("pct", 1), "%": ("pct", 1)}
_QTY_PHRASES = [(re.compile(r"\bhalf an hour\b"), "30 minutes"),
                (re.compile(r"\ba quarter of an hour\b"), "15 minutes"),
                (re.compile(r"\bquarter of an hour\b"), "15 minutes"),
                (re.compile(r"\bhalf a day\b"), "12 hours"),
                (re.compile(r"\ban hour and a half\b"), "90 minutes"),
                (re.compile(r"\btwenty[- ]four hours\b"), "24 hours")]
_NUM_RE = r"(\d+(?:\.\d+)?|" + "|".join(
    sorted((re.escape(w) for w in _NUM_WORDS), key=len, reverse=True)) + r")"
_QTY_RE = re.compile(
    r"\b" + _NUM_RE + r"(?:\s*(?:-|to|–)\s*" + _NUM_RE + r")?\s*(?:full\s+|calendar\s+|failed\s+|more\s+)?"
    r"(business days?|working days?|minutes?|mins?|hours?|hrs?|days?|weeks?|fortnights?|"
    r"months?|years?|attempts?|tries|try|seats?|percent|%)", re.I)
QUANTITY_FEATURE_NAMES = ["draft_quantities", "unsupported_quantities",
                          "unsupported_ratio", "any_unsupported"]


def _num_value(tok: str) -> float:
    tok = tok.lower()
    return float(tok) if tok[0].isdigit() else float(_NUM_WORDS[tok])


def quantities(text: str) -> set:
    t = (text or "").lower()
    for pat, rep in _QTY_PHRASES:
        t = pat.sub(rep, t)
    out = set()
    for m in _QTY_RE.finditer(t):
        unit = m.group(3).lower()
        unit = unit if unit in ("tries", "%") else unit.rstrip("s")
        key = next((k for k in _UNIT if unit.startswith(k)), None)
        if key is None:
            continue
        family, mult = _UNIT[key]
        for g in (m.group(1), m.group(2)):
            if g:
                out.add((family, round(_num_value(g) * mult, 3)))
    return out


def quantity_features(draft: str, kb: str) -> np.ndarray:
    d, k = quantities(draft), quantities(kb)
    unsupported = d - k
    return np.array([len(d), len(unsupported), len(unsupported) / len(d) if d else 0.0,
                     float(bool(unsupported))], dtype=np.float32)
