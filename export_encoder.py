"""
Export sentence-transformers/all-MiniLM-L6-v2 to a quantized ONNX graph.

Produces, under artifacts/encoder/:
    model_fp32.onnx      (export target, deleted after quantization unless --keep-fp32)
    model.onnx            int8 dynamically-quantized graph actually served
    tokenizer.json, vocab.txt, tokenizer_config.json, special_tokens_map.json

The model is frozen (no fine-tuning in this stage): we only export its
pretrained weights. License: Apache-2.0 (see the model card on the Hub for
sentence-transformers/all-MiniLM-L6-v2).

Run once, before embed.py / train.py:
    python export_encoder.py

After finetune_encoder.py produces artifacts/encoder_ft_hf/, export THAT
checkpoint instead (still writes to artifacts/encoder/, overwriting the
frozen-base export - run embed.py + train.py again afterward):
    python export_encoder.py --model_dir artifacts/encoder_ft_hf
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path("artifacts/encoder")
MAX_LENGTH = 128  # see README: p99 ticket token length is 82, max observed 106


def export(out_dir: Path, max_length: int, model_dir: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model.eval()

    tok.save_pretrained(out_dir)

    dummy = tok(["warm up the exporter"], return_tensors="pt", truncation=True,
                max_length=max_length)
    fp32_path = out_dir / "model_fp32.onnx"
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    print("exported fp32 ONNX: %.1f MB" % (fp32_path.stat().st_size / 1e6))
    return fp32_path


def quantize(fp32_path: Path, out_dir: Path, keep_fp32: bool) -> Path:
    int8_path = out_dir / "model.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    print("exported int8 ONNX:  %.1f MB" % (int8_path.stat().st_size / 1e6))
    if not keep_fp32:
        fp32_path.unlink()
    return int8_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--keep-fp32", action="store_true")
    ap.add_argument("--model_dir", default=MODEL_NAME,
                     help="HF hub id (default) or a local checkpoint dir, "
                          "e.g. artifacts/encoder_ft_hf after fine-tuning")
    args = ap.parse_args()

    fp32_path = export(OUT_DIR, args.max_length, args.model_dir)
    quantize(fp32_path, OUT_DIR, args.keep_fp32)

    total = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file())
    print("\nartifacts/encoder/ total size: %.1f MB" % (total / 1e6))
    for f in sorted(OUT_DIR.iterdir()):
        print("  %-28s %8.2f KB" % (f.name, f.stat().st_size / 1024))


if __name__ == "__main__":
    main()
