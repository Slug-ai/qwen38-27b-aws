#!/usr/bin/env python3
"""Serve the model through an OpenAI-compatible API (slug-api), taking requests as they arrive.

Needs slug-api in the environment: `pip install git+https://github.com/Slug-ai/slug-api`. Loading takes a few
minutes, then it listens on 127.0.0.1:8000. Reach it from another machine through an SSH tunnel
(`ssh -L 8000:127.0.0.1:8000 ec2-user@<instance>`), or with --bind 0.0.0.0 and a --key.
"""

import argparse
import warnings

from slug_api import serve

from qwen38_inf2.generate import EOS_TOKEN_IDS
from qwen38_inf2.runner import load
from qwen38_inf2.scheduler import Scheduler
from qwen38_inf2.serving import ChipBackend, chat_prompt


def main():
    warnings.showwarning = lambda *args, **kwargs: None
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--compiled", required=True, help="A build with a prefill graph (compile.py --chunk).")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--head", required=True, help="Compiled lm_head for the chip. Its weights go in --weights too.")
    parser.add_argument("--mtp", help="Compiled MTP drafter: speculate with its drafts. Its weights go in --weights too.")
    parser.add_argument("--no-speculation", action="store_true", help="Decode one token per step even if the build has a verify graph.")
    parser.add_argument("--reuse-prefixes", action="store_true", help="Start a prompt from a slot's saved states when it extends that slot's last prompt, as multi_turn_benchmark.py does.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Tokens to generate for a request that doesn't say.")
    parser.add_argument("--model", default="qwen3.8-27b", help="The name /v1/models gives.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="127.0.0.1", help="Address to listen on (0.0.0.0 for every interface).")
    parser.add_argument("--key", help="Require Authorization: Bearer <key> on every request.")
    args = parser.parse_args()

    loaded = load(args.model_dir, args.compiled, args.weights, args.head, mtp_dir=args.mtp)
    if loaded.prefill is None:
        parser.error(f"{args.compiled} has no prefill graph: compile it with --chunk")
    scheduler = Scheduler(
        loaded.step, loaded.prefill, loaded.head, loaded.cfg, loaded.batch_size, loaded.max_seq_len, loaded.chunk, EOS_TOKEN_IDS,
        copy_states=loaded.step if args.reuse_prefixes else None,
        verify=None if args.no_speculation else loaded.verify, verify_tokens=loaded.verify_tokens, mtp=loaded.mtp,
    )
    tok = loaded.tokenizer
    serve(
        ChipBackend(scheduler, args.max_tokens),
        lambda messages: tok.encode(chat_prompt(messages), add_special_tokens=False).ids,
        lambda ids: tok.decode(ids, skip_special_tokens=True),
        args.model, args.key, args.bind, args.port,
    )


if __name__ == "__main__":
    main()
