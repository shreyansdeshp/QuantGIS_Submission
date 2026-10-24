# Model card — Decide v6

## What was built

One CPU service answers `route`, `approve` and `grounded` for a support ticket
from a single shared sentence encoder (MiniLM-L6, **fine-tuned** on this data,
then exported to int8 ONNX), one shared featurizer (`features.py`), and three
small heads (logistic regression / gradient boosting). No network calls and
no GPU at serving time.

- **Encoder:** `all-MiniLM-L6-v2` fine-tuned jointly on all three tasks (equal
  loss weights, 6 epochs, best epoch chosen on a validation slice).
- **route:** logistic regression on the ticket vector plus a keyword score.
  Per-class (diagonal) probability scaling.
- **approve:** gradient-boosted trees. Inputs are one score summarising the
  ticket vector, amount/tier features, a keyword score, and two flags for
  rules found in the training data (risk sentences, never-approved request
  types). Beta calibration; threshold tuned for the 3:1 cost.
- **grounded:** logistic regression on the draft/article vector pair, word
  overlap, a words-only-in-one-text score, and a count of quantities in the
  draft that the article does not state (normalised: "two weeks" = "14 days").
  Beta calibration.

## What it was trained on

`data/train.jsonl` only (6,500 tickets). The encoder was fine-tuned on 88% of
it, with the other 12% used to stop training. `data/dev.jsonl` was **never
trained on**, and nothing was hard-coded from it. It was, however, used to
choose between about six variants of the model (calibration methods, training
length, loss weights, new features), and we looked at a few dev errors while
designing the grounded feature. So the scores below are held-out, but
slightly optimistic: expect the hidden test score to come in a little lower.

## Measured results (heads and encoder trained on train only, scored on dev)

| ranked metric | measure | result |
|---|---|---|
| Decision quality | route macro-F1 | 0.927 (majority-class 0.085) |
| | approve cost at 3:1 | 0.165 (always-hold 0.342) |
| | grounded macro-F1 | 0.993 (majority-class 0.375) |
| Calibration | ECE route / approve / grounded | 0.039 / 0.038 / 0.019 |
| | Brier approve / grounded | 0.089 / 0.026 |
| | Brier skill (mean of three) | 0.750 |
| Latency (one request at a time, warm, server limited to 4 cores) | p95 route / approve / grounded | 14.6 / 16.6 / 20.3 ms |
| | mean p95 | 17.2 ms (full marks at 25 ms) |
| Abstention | error on the 80% most confident, route / approve / grounded | 0.064 / 0.064 / 0.006 |
| | error at full coverage | 0.137 / 0.119 / 0.031 |
| | coverage actually answered | 0.84 / 0.81 / 0.84 |

`selfcheck.py` total: **82.88 / 100** (quality 33.96, calibration 19.87,
latency 20.00, abstention 9.04). All gates pass and all 4,500 responses are
schema-valid. For reference: TF-IDF baseline 68.53, frozen-encoder version
74.85, fine-tuned v3 81.56. Latency was measured with
the server restricted to 4 CPU cores (the judges' machine shape). That test
found a real problem: with default threading, approve took about 390 ms
because the tree model's OpenMP threads fought onnxruntime's threads.
`model.py` now runs BLAS/OpenMP single-threaded, which removed it. Expect a
few ms of run-to-run variation. The hidden test set will differ from dev.

## Where it fails

- **route (86% accuracy):** about 109 of 1,500 dev tickets are vague ("Something
  is not right and I am not sure who I should be asking") and their queue labels
  look close to random; the model gets about 24% of them right and mostly
  lowers its confidence on them. Another group of tickets has a clear request
  but a different queue than the same request usually gets (all wrong). We
  believe this is label noise, but we have not proven it.
- **approve:** the hard rules are handled (over the tier limit: about 0.5% error;
  risk sentences: 0%). The remaining errors are ordinary tickets where
  identical requests received different labels, and tickets near the tier
  limit, where approval is genuinely only 30 to 50% likely. The tier limits
  the model uses are approximate (about 30 / 120 / 400-500).
- **grounded (0.993):** remaining errors are mostly a draft that restates a
  number or duration in a form the normaliser does not recognise.

## When it abstains

Each decision flags about the 15% of cases it is least sure of, using cutoffs
fixed from the training data (not tuned on dev). route and grounded abstain on
low top-class probability; approve abstains when the expected cost of its
answer (3x for a wrong approval) is high. Confidence is the calibrated
probability itself, so abstaining never changes the reported probability. On
dev the model answered 81 to 84% of tickets per decision.
