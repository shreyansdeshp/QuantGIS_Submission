# Decide — v6

CPU-only `POST /v1/decide` service answering `route`, `approve` and `grounded`
for a support ticket from one shared fine-tuned MiniLM encoder plus three small
heads. Accuracy, calibration, latency, abstention and failure modes are in
`model_card.md`. Held-out score on dev: **82.88 / 100**.

## Start command

```bash
pip install -r requirements.txt && ORT_INTRA_THREADS=3 uvicorn app:app --host 0.0.0.0 --port 8000
```

Needs **Python 3.11 or newer** (tested on 3.12): the model file was written
by scikit-learn 1.9.0, which `requirements.txt` pins exactly, and that
version needs 3.11+.

Everything the server loads is committed (`artifacts/encoder/` about 23 MB,
`artifacts/v2/model.pkl` about 3 MB). Nothing is downloaded or trained at
serve time. Score it with:

```bash
python selfcheck.py --url http://localhost:8000/v1/decide
```

## Where the sharing happens

- **One encoder.** `encoder.py` loads a single int8 ONNX session
  (`artifacts/encoder/model.onnx`, MiniLM-L6 fine-tuned on this data).
  `route` and `approve` encode the ticket text with the same call; `grounded`
  calls it on the draft and the article. No decision has its own encoder.
- **One featurizer.** `features.py` turns a ticket into numbers for training
  and serving alike: amount/tier features, word overlap, keyword scores, hard-rule
  flags, and quantity matching. `train.py` and `model.py` both import it.
- **Three small heads.** About 3 MB of scikit-learn models on top of the
  shared vectors, calibrated in `calibration.py`. (`approve` also uses a small
  linear model to turn the ticket vector into one score; that is part of its
  head, not a second encoder.)

## Pretrained components and licenses

- **sentence-transformers/all-MiniLM-L6-v2**, Apache-2.0 (per its model card on
  Hugging Face). We fine-tuned it on `train.jsonl` and exported it to ONNX
  int8. Fine-tuned weights remain under Apache-2.0.

No LLM or embedding API is used. Only `finetune_encoder.py` and
`export_encoder.py` need the network (to fetch the base model), and only
before serving.

## Reproduce

```bash
python finetune_encoder.py --epochs 6 --batch_size 12    # GPU; trains on train.jsonl only
python export_encoder.py --model_dir artifacts/encoder_ft_hf
python embed.py                                          # cache vectors
python train.py                                          # fit heads on train.jsonl only
```

The committed `artifacts/` come from exactly these steps. `train.py --all`
would also fit on dev, but we deliberately did not, so every dev number we
report is held out. The OOF numbers `train.py` prints are optimistic (the
encoder had seen every training row); trust only the `selfcheck.py` run.

## Results (train only, scored on dev)

| | TF-IDF baseline | frozen encoder | fine-tuned v3 | **v6** |
|---|---|---|---|---|
| route macro-F1 | 0.937 | 0.927 | 0.932 | 0.927 |
| approve cost (3:1) | 0.264 | 0.189 | 0.175 | **0.165** |
| grounded macro-F1 | 0.826 | 0.845 | 0.988 | **0.993** |
| mean p95 latency | 9 ms | ~17 ms | ~14 ms | ~17 ms (4 cores) |
| **selfcheck total** | 68.53 | 74.85 | 81.56 | **82.88** |

v6 latency was measured with the server limited to 4 CPU cores (p95: route
14.6, approve 16.6, grounded 20.3 ms), under the 25 ms full-marks line.
`results.json` is that held-out run.

## Known limits

Most remaining errors are in `route` (vague tickets with near-random queue
labels) and `approve` (identical requests with different labels). We believe
that is label noise, but we have not proven it. See `model_card.md`.
