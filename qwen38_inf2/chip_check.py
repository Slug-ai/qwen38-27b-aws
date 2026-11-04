"""Compare a chip with saved CPU results, one level at a time.

Each level adds one thing the chip must get right, so the first failure says where to look. CPU and
chip are not expected to agree exactly in BF16, because the chip approximates sigmoid and exp with
lookup tables and rounds differently. So token-level checks compare both against the CPU's float32
result, and allow the chip a few times the error that CPU BF16 itself has. Those limits are
judgement calls, chosen before any chip has run, and may need adjusting once one has.
"""

import time
from dataclasses import dataclass, field

import torch

from .config import TextConfig
from .generate import step_inputs

DTYPE = torch.bfloat16
# Allow the chip this many times the CPU's own BF16 error, and never less than the floor.
ERROR_FACTOR = 3.0
ERROR_FLOOR = 0.02
# A top-token change only counts as rounding if the top two logits were this close on the CPU.
NEAR_TIE_LOGITS = 0.25
MAX_NEAR_TIE_FLIPS = 2


@dataclass
class LevelResult:
    name: str
    passed: bool
    details: dict = field(default_factory=dict)
    hint: str = ""


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-12))


def check_decode(run, golden: dict) -> LevelResult:
    """FP4 decode must be exact. Every value is representable, so no rounding is involved."""
    got = run(gain=torch.ones(1, dtype=DTYPE)).reshape(golden["expected"].shape)
    wrong = int((got.float() != golden["expected"].float()).sum())
    return LevelResult(
        "decode",
        wrong == 0,
        {"wrong_values": wrong, "total_values": golden["expected"].numel()},
        "" if wrong == 0 else "FP4 decode is wrong, so nothing above it can be right. Suspect floor, clamp or the uint8 cast.",
    )


def check_matrix(run, golden: dict) -> LevelResult:
    got = run(x=golden["x"].to(DTYPE))
    error = relative_error(got, golden["float32"])
    baseline = relative_error(golden["bf16"], golden["float32"])
    limit = max(ERROR_FACTOR * baseline, ERROR_FLOOR)
    passed = bool(torch.isfinite(got.float()).all()) and error <= limit
    hint = "" if passed else "One real matrix is wrong while decode passed. Suspect weight loading or BF16 matmul."
    return LevelResult("matrix", passed, {"error": error, "limit": limit, "cpu_bf16_error": baseline}, hint)


def check_tokens(step, cfg: TextConfig, golden: dict, max_seq_len: int, name: str) -> LevelResult:
    """Run tokens in order through layers that keep state, comparing every token."""
    errors, limits = [], []
    for position in range(golden["inputs"].shape[0]):
        inputs = step_inputs(cfg, position, max_seq_len, DTYPE)
        got = step(inputs_embeds=golden["inputs"][position].reshape(1, 1, -1).to(DTYPE), **inputs)
        error = relative_error(got, golden["float32"][position])
        limit = max(ERROR_FACTOR * relative_error(golden["bf16"][position], golden["float32"][position]), ERROR_FLOOR)
        errors.append(error)
        limits.append(limit)
        if not torch.isfinite(got.float()).all() or error > limit:
            hint = (
                "The first token failed, so state is not the cause. Suspect the all-reduce or this level's math."
                if position == 0
                else "Earlier tokens passed, so suspect caches or gated delta state not persisting between tokens."
            )
            return LevelResult(name, False, {"failed_at_token": position, "errors": errors, "limits": limits}, hint)
    return LevelResult(name, True, {"tokens": len(errors), "errors": errors, "limits": limits})


def check_model(step, head, cfg: TextConfig, golden: dict, max_seq_len: int) -> LevelResult:
    """Top tokens through all blocks. A flip is only forgiven where the CPU's top two were nearly tied."""
    flips = []
    for position, token_id in enumerate(golden["tokens"]):
        inputs = step_inputs(cfg, position, max_seq_len, DTYPE)
        hidden = step(inputs_embeds=head.embed(token_id, DTYPE), **inputs)
        top = int(torch.argmax(head.logits(hidden)))
        if top != golden["top"][position]:
            flips.append({"position": position, "got": top, "want": golden["top"][position], "cpu_margin": golden["margin"][position]})
    clear = [flip for flip in flips if flip["cpu_margin"] >= NEAR_TIE_LOGITS]
    passed = not clear and len(flips) <= MAX_NEAR_TIE_FLIPS
    hint = "" if passed else "Lower levels passed, so suspect block chaining or the weights of later blocks."
    return LevelResult("model", passed, {"positions": len(golden["tokens"]), "flips": flips}, hint)


def check_greedy(step, head, cfg: TextConfig, golden: dict, max_seq_len: int, generate) -> LevelResult:
    """The real thing. One near-tie flip changes every later token, so read the text if this fails."""
    got = list(generate(step, head, cfg, golden["prompt"], len(golden["output"]), max_seq_len, (), DTYPE))
    same = got == golden["output"]
    hint = "" if same else "If `model` passed, this may be a harmless near-tie. Read the generated text before debugging."
    return LevelResult("greedy", same, {"got": got, "want": golden["output"]}, hint)


def measure_speed(step, head, cfg: TextConfig, prompt: list[int], new_tokens: int, max_seq_len: int) -> LevelResult:
    """Real per-token time, split between the chip and the host. There is no pass or fail here."""
    chip, host = [], []
    token_id = prompt[0]
    for position in range(len(prompt) + new_tokens):
        inputs = step_inputs(cfg, position, max_seq_len, DTYPE)
        embed = head.embed(token_id, DTYPE)
        started = time.perf_counter()
        hidden = step(inputs_embeds=embed, **inputs)
        chip.append(time.perf_counter() - started)
        started = time.perf_counter()
        top = int(torch.argmax(head.logits(hidden)))
        host.append(time.perf_counter() - started)
        token_id = prompt[position + 1] if position + 1 < len(prompt) else top
    # The first run on a chip can include one-off setup, so only later steps count.
    steady_chip = sorted(chip[1:])[len(chip[1:]) // 2]
    steady_host = sorted(host[1:])[len(host[1:]) // 2]
    return LevelResult(
        "speed",
        True,
        {
            "first_step_seconds": chip[0],
            "chip_seconds_per_token": steady_chip,
            "host_seconds_per_token": steady_host,
            "tokens_per_second": 1.0 / (steady_chip + steady_host),
        },
    )
