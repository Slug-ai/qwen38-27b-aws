#!/usr/bin/env python3
"""Run the real checkpoint through this model and the Hugging Face layers, side by side, on CPU.

Weights are streamed layer by layer over HTTP range requests and dropped after use, so this needs no
disk space and only a few GB of RAM. Every prompt token passes through one layer before the next is
fetched, so one pass over the weights gives a next-token prediction at every prompt position.

Reports per-layer agreement with the reference, then checks that the model predicts natural text.
Arithmetic is float32 because BF16 matmul is very slow on CPU. BF16 parity is covered by the tests.
"""

import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

from qwen38_inf2 import checkpoint, fp4
from qwen38_inf2.config import LINEAR, TextConfig
from qwen38_inf2.generate import step_inputs
from qwen38_inf2.model import DecoderLayer, FP4Linear, identity_all_reduce
from tests.reference import hf_config

REPO = "https://huggingface.co/RedHatAI/Qwen3.8-27B-MXFP4/resolve/main/"
DEFAULT_TEXT = (
    "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is Rome. "
    "Water is made of hydrogen and oxygen. The Earth orbits the Sun once every year."
)
DTYPE = torch.float32
SCALAR_STATE = ("conv_state", "recurrent_state", "k_cache", "v_cache")


def fetch_text(name: str) -> str:
    with urllib.request.urlopen(REPO + name) as response:
        return response.read().decode()


def fetch_tokenizer(path: Path) -> Tokenizer:
    if not path.exists():
        path.write_text(fetch_text("tokenizer.json"))
    return Tokenizer.from_file(str(path))


def memoize_fp4_decode(layer: torch.nn.Module) -> None:
    """Decode each matrix once with the in-graph decoder. Re-decoding per token would be wasted CPU time."""
    for module in layer.modules():
        if isinstance(module, FP4Linear):
            weight = fp4.decode_in_graph(module.weight_packed, module.weight_scale).to(DTYPE)
            module.forward = lambda x, w=weight: F.linear(x, w)


def build_layer(cfg, index, weights, max_seq_len, tp_degree=1, all_reduce=identity_all_reduce):
    layer = DecoderLayer(cfg, cfg.layer_types[index], tp_degree, max_seq_len, DTYPE, all_reduce).eval()
    prefix = f"layers.{index}."
    local = {k.removeprefix(prefix): v for k, v in weights.items()}
    missing, unexpected = layer.load_state_dict(local, strict=False)
    assert not unexpected, unexpected
    assert all(m.split(".")[-1] in SCALAR_STATE for m in missing), missing
    memoize_fp4_decode(layer)
    return layer


def run_ours(layer, cfg, hidden, inputs):
    """Step one layer through every token, as the compiled graph does."""
    outputs = []
    for t in range(hidden.shape[1]):
        i = inputs[t]
        outputs.append(layer(hidden[:, t : t + 1], i["cos"], i["sin"], i["write_mask"], i["attn_mask"], i["keep_state"]))
    return torch.cat(outputs, dim=1)


def build_hf_layer(hf_cfg, index, weights):
    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(hf_cfg, index)
    state = {}
    for name, value in weights.items():
        target = name.removeprefix(f"layers.{index}.")
        state[target] = value
    layer.load_state_dict(state, strict=True, assign=True)
    return layer.to(DTYPE).eval()


def hf_state_dict(source, cfg, index):
    """Reference weights: decoded by compressed-tensors' rules, under Hugging Face's names."""
    prefix = f"{checkpoint.TEXT_PREFIX}layers.{index}."
    out = {}
    for name in source.names():
        if not name.startswith(prefix) or name.endswith(checkpoint.UNUSED_LAYER_SUFFIXES):
            continue
        local = name.removeprefix(prefix)
        if local.endswith(".weight_scale"):
            continue
        if local.endswith(".weight_packed"):
            base = name.removesuffix(".weight_packed")
            out[local.removesuffix(".weight_packed") + ".weight"] = fp4.decode_host(
                source.get(f"{base}.weight_packed"), source.get(f"{base}.weight_scale"), torch.bfloat16
            )
        else:
            out[local] = source.get(name)
    return {k: v.to(DTYPE) for k, v in out.items()}


def run_hf(layer, hidden, position_embeddings, causal_mask):
    return layer(
        hidden,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask if layer.block_type != LINEAR else None,
    )


def top_predictions(source, hidden: torch.Tensor, chunk_rows: int = 32768) -> torch.Tensor:
    """Argmax of lm_head at each position. Streams the 2.5 GB matrix in pieces."""
    rows = source.shape(checkpoint.LM_HEAD)[0]
    best_value = torch.full((hidden.shape[0],), -torch.inf)
    best_index = torch.zeros(hidden.shape[0], dtype=torch.long)
    for start in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - start)
        block = source.get_row_range(checkpoint.LM_HEAD, start, count).float()
        logits = hidden @ block.T
        value, index = logits.max(dim=-1)
        better = value > best_value
        best_value = torch.where(better, value, best_value)
        best_index = torch.where(better, index + start, best_index)
    return best_index


def check_tp(cfg, source, index, hidden, inputs, want, tp_degree=2):
    """Slice this real layer for two ranks, run both in lock-step and compare with one rank."""
    barrier = threading.Barrier(tp_degree)
    parts: dict = {}
    lock = threading.Lock()

    def reducer(rank):
        def all_reduce(x):
            with lock:
                parts[rank] = x
            barrier.wait()
            total = sum(parts.values())
            barrier.wait()
            return total
        return all_reduce

    results = {}

    def worker(rank):
        weights = checkpoint.layer_weights(source, cfg, index, rank, tp_degree)
        layer = build_layer(cfg, index, weights, inputs[0]["write_mask"].shape[2], tp_degree, reducer(rank))
        with torch.no_grad():
            results[rank] = run_ours(layer, cfg, hidden, inputs)

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(tp_degree)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return max(float((results[r] - want).abs().max()) for r in range(tp_degree))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=REPO + "model.safetensors")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--num-layers", type=int, default=64, help="Stop early for a quick partial check.")
    parser.add_argument("--tp-check-layers", type=int, nargs="*", default=[0, 3], help="Layers to also run as two ranks.")
    parser.add_argument("--tokenizer-cache", type=Path, default=Path("/tmp/qwen38_tokenizer.json"))
    parser.add_argument("--output", type=Path, default=None, help="Write results as JSON here.")
    args = parser.parse_args()

    source = checkpoint.TensorSource(args.url)
    cfg = TextConfig.from_model_config(json.loads(fetch_text("config.json")))
    hf_cfg = hf_config(cfg)
    tokens = fetch_tokenizer(args.tokenizer_cache).encode(args.text, add_special_tokens=False).ids
    length = len(tokens)
    print(f"{length} tokens, streaming {args.num_layers} layers", flush=True)

    rows = source.get_rows(checkpoint.EMBED_TOKENS, tokens).to(DTYPE).unsqueeze(0)
    ours, ref = rows.clone(), rows.clone()

    inputs = [step_inputs(cfg, t, length, DTYPE) for t in range(length)]
    rotary = Qwen3_5TextRotaryEmbedding(hf_cfg)
    position_ids = torch.arange(length).reshape(1, 1, -1).expand(3, 1, -1)
    position_embeddings = rotary(rows, position_ids)
    causal_mask = torch.triu(torch.full((length, length), -torch.inf), diagonal=1)[None, None]

    report = {"tokens": length, "layers": [], "tp": {}}
    torch.set_grad_enabled(False)
    started = time.time()
    for index in range(args.num_layers):
        weights = checkpoint.layer_weights(source, cfg, index, 0, 1, DTYPE)
        layer = build_layer(cfg, index, weights, length)
        before = ours
        ours = run_ours(layer, cfg, before, inputs)
        ref = run_hf(build_hf_layer(hf_cfg, index, hf_state_dict(source, cfg, index)), ref, position_embeddings, causal_mask)

        error = float((ours - ref).abs().max())
        relative = error / float(ref.abs().max())
        entry = {"layer": index, "type": cfg.layer_types[index], "max_abs_error": error, "relative": relative}
        if index in args.tp_check_layers:
            entry["tp2_max_abs_error"] = report["tp"][index] = check_tp(cfg, source, index, before, inputs, ours)
        report["layers"].append(entry)
        tp_note = f" tp2_err={entry['tp2_max_abs_error']:.2e}" if "tp2_max_abs_error" in entry else ""
        print(
            f"layer {index:2d} {cfg.layer_types[index]:16s} err={error:.2e} rel={relative:.2e}{tp_note} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )
        del layer, weights

    report["seconds_for_layers"] = time.time() - started
    if args.num_layers < cfg.num_hidden_layers:
        # A truncated stack predicts nothing meaningful, so stop after the parity numbers.
        print(json.dumps(report["layers"][-1]))
    else:
        norm = source.get(checkpoint.FINAL_NORM).to(DTYPE)
        from qwen38_inf2.model import rms_norm

        ours_final = rms_norm(ours, norm, cfg.rms_norm_eps)[0]
        ref_final = rms_norm(ref, norm, cfg.rms_norm_eps)[0]
        ours_top = top_predictions(source, ours_final)
        ref_top = top_predictions(source, ref_final)
        tokenizer = fetch_tokenizer(args.tokenizer_cache)
        expected = torch.tensor(tokens[1:])
        report["final_hidden_max_abs_error"] = float((ours_final - ref_final).abs().max())
        report["top1_equal_to_reference"] = int((ours_top == ref_top).sum())
        report["positions"] = length
        report["next_token_accuracy"] = float((ours_top[:-1] == expected).float().mean())
        report["predictions"] = [
            {"context_end": tokenizer.decode([tokens[t]]), "predicted": tokenizer.decode([int(ours_top[t])])}
            for t in range(length)
        ]
        for item in report["predictions"]:
            print(f"  ...{item['context_end']!r:16} -> {item['predicted']!r}")
        print(
            f"top-1 equals reference at {report['top1_equal_to_reference']}/{length} positions; "
            f"next-token accuracy {report['next_token_accuracy']:.2%}; "
            f"final hidden max error {report['final_hidden_max_abs_error']:.2e}",
            flush=True,
        )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
