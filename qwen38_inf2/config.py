"""Text-only model configuration. The vision tower and MTP head are not represented here at all."""

import json
from dataclasses import dataclass
from pathlib import Path

LINEAR = "linear_attention"
FULL = "full_attention"


@dataclass(frozen=True)
class TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    # How FP4 weights are held on the device: "fp4" stays packed and is decoded in the graph every step,
    # "fp8" takes about 1.8 times the memory and is read straight by the Tensor Engine.
    weight_format: str = "fp4"

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layer_types)

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def linear_key_repeat(self) -> int:
        return self.linear_num_value_heads // self.linear_num_key_heads

    def validate_tp(self, tp_degree: int) -> None:
        """Every split must land on a head boundary and on a 32-element FP4 group boundary."""
        must_divide = {
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
        }
        for name, value in must_divide.items():
            if value % tp_degree:
                raise ValueError(f"{name}={value} is not divisible by tp_degree={tp_degree}")
        split_columns = {
            "mlp.down_proj": self.intermediate_size,
            "linear_attn.out_proj": self.linear_value_dim,
            "self_attn.o_proj": self.num_attention_heads * self.head_dim,
        }
        for name, cols in split_columns.items():
            if cols % (32 * tp_degree):
                raise ValueError(f"{name} input width {cols} does not split on FP4 groups at tp_degree={tp_degree}")

    @classmethod
    def from_model_config(cls, config: dict) -> "TextConfig":
        text = config.get("text_config", config)
        rope = text["rope_parameters"]
        return cls(
            vocab_size=text["vocab_size"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            layer_types=tuple(text["layer_types"]),
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_num_value_heads=text["linear_num_value_heads"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
            rms_norm_eps=text["rms_norm_eps"],
            rope_theta=rope["rope_theta"],
            partial_rotary_factor=rope["partial_rotary_factor"],
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "TextConfig":
        return cls.from_model_config(json.loads(Path(path).read_text()))


def tiny_config(num_blocks: int = 2) -> TextConfig:
    """Same shape ratios as the real model, small enough for CPU tests and quick compiles."""
    return TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        layer_types=(LINEAR, LINEAR, LINEAR, FULL) * num_blocks,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000000.0,
        partial_rotary_factor=0.25,
    )
