"""
Shared encoder wrapper: the ONE encoder used by all three decisions.

Loads the quantized ONNX MiniLM graph + its tokenizer once, and exposes a
single `encode(text)` -> 384-d mean-pooled vector. Both embed.py (offline
caching for training) and model.py (serving) import this module, so there is
exactly one place that turns text into a vector.

Thread count is configurable via the ORT_INTRA_THREADS env var; bench.py
sweeps this to find the best setting for the serving machine (measured: 4
threads beats 1 on a 4-core box because a single BERT-sized matmul benefits
from intra-op parallelism even at batch size 1).
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

DEFAULT_MAX_LENGTH = 128


class Encoder:
    def __init__(self, artifacts_dir: str = "artifacts/encoder",
                 max_length: int = DEFAULT_MAX_LENGTH,
                 intra_threads: int = None):
        artifacts_dir = Path(artifacts_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(artifacts_dir))
        self.max_length = max_length

        threads = intra_threads or int(os.environ.get("ORT_INTRA_THREADS", "3"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(artifacts_dir / "model.onnx"), sess_options=so,
            providers=["CPUExecutionProvider"])

    def encode(self, text: str) -> np.ndarray:
        """Single text -> 384-d mean-pooled vector. Batch size 1, no padding."""
        enc = self.tokenizer([text or ""], return_tensors="np",
                              truncation=True, max_length=self.max_length)
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        out = self.session.run(["last_hidden_state"],
                                {"input_ids": ids, "attention_mask": mask})[0]
        m = mask[..., None].astype(np.float32)
        pooled = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return pooled[0].astype(np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Used only offline (embed.py cache-building); pads within the batch."""
        enc = self.tokenizer(list(texts), return_tensors="np", padding=True,
                              truncation=True, max_length=self.max_length)
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        out = self.session.run(["last_hidden_state"],
                                {"input_ids": ids, "attention_mask": mask})[0]
        m = mask[..., None].astype(np.float32)
        pooled = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return pooled.astype(np.float32)

    def warmup(self, n: int = 3) -> None:
        for _ in range(n):
            self.encode("warm up the model before serving traffic")
