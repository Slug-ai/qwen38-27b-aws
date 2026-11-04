"""Build a tiny checkpoint and the matching Hugging Face reference model."""

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qwen38_inf2 import checkpoint, fp4
from qwen38_inf2.config import TextConfig


def hf_config(cfg: TextConfig) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        layer_types=list(cfg.layer_types),
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        linear_num_key_heads=cfg.linear_num_key_heads,
        linear_num_value_heads=cfg.linear_num_value_heads,
        linear_key_head_dim=cfg.linear_key_head_dim,
        linear_value_head_dim=cfg.linear_value_head_dim,
        linear_conv_kernel_dim=cfg.linear_conv_kernel_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": cfg.rope_theta,
            "partial_rotary_factor": cfg.partial_rotary_factor,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        tie_word_embeddings=False,
        attention_bias=False,
        attn_implementation="eager",
    )


def build_reference(cfg: TextConfig, seed: int = 0, dtype: torch.dtype = torch.float32):
    """Return (HF model holding FP4-quantized weights, checkpoint tensors named as the real file).

    Both sides end up with exactly the same weight values, so any logit difference is a model bug.
    """
    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(hf_config(cfg)).eval()

    tensors: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        ckpt_name = name if name == checkpoint.LM_HEAD else checkpoint.TEXT_PREFIX + name.removeprefix("model.")
        if name.endswith(".weight") and param.ndim == 2 and _is_quantized(name):
            packed, scale_bytes = fp4.encode_host(param.data)
            tensors[ckpt_name.removesuffix(".weight") + ".weight_packed"] = packed
            tensors[ckpt_name.removesuffix(".weight") + ".weight_scale"] = scale_bytes
            param.data = fp4.decode_host(packed, scale_bytes, param.dtype)
        else:
            tensors[ckpt_name] = param.data.to(torch.bfloat16)
            # Keep the reference on the same rounded values the checkpoint stores.
            param.data = tensors[ckpt_name].to(param.dtype)
    return model.to(dtype), tensors


def _is_quantized(name: str) -> bool:
    if "lm_head" in name or "embed_tokens" in name or "conv1d" in name:
        return False
    if name.endswith(("in_proj_a.weight", "in_proj_b.weight")):
        return False
    return name.endswith("proj.weight") or "proj_" in name
