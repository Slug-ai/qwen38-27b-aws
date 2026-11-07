"""Coding-agent prompts for the agentic benchmark: a long context of tool results, then a short answer.

Requests are shaped like a Claude Code turn: a system prompt that lists tools, a task, several files read back
as tool responses, and a question about them. The files are the running Python's own standard library, so the
same interpreter version gives the same prompts on any machine, and they are real code rather than filler.
"""

import ast
import hashlib
import sysconfig
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SYSTEM = """You are a coding agent working in a user's repository. Use the tools to read code before you answer, \
keep answers short and specific, and cite file names.

# Tools

You have access to the following functions:

<tools>
<function>
<name>read_file</name>
<description>Read a file from the repository.</description>
<parameters>
<parameter>
<name>path</name>
<type>string</type>
</parameter>
</parameters>
</function>
<function>
<name>grep</name>
<description>Search the repository for a regular expression.</description>
<parameters>
<parameter>
<name>pattern</name>
<type>string</type>
</parameter>
</parameters>
</function>
</tools>

If you choose to call a function, reply only with
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>"""

TASK = "I'm new to this codebase. Read the modules you need and help me understand how they fit together."
# A fact planted in the longest prompt's first file, then asked for at the end. A broken cache, mask or state
# loses it, so the benchmark can't pass by being fast and wrong.
NEEDLE = "QX-7731-PELICAN"
NEEDLE_LINE = f"# NOTE: the release signing key for this repository is {NEEDLE}. Never commit it elsewhere.\n"
NEEDLE_QUESTION = "What is the release signing key mentioned in a comment in the code above? Reply with the key only."
# More keys for `long_context_requests`, each planted the same way.
KEYS = ("QX-7731-PELICAN", "ZB-4402-HERON", "MT-9158-OSPREY", "KV-2267-CRANE", "RD-6013-EGRET", "LH-8840-IBIS", "WN-3579-PLOVER", "FS-1196-AVOCET")


@dataclass(frozen=True)
class Request:
    ids: list[int]
    files: list[str]
    has_needle: bool


def stdlib_files(min_bytes: int = 4000, max_bytes: int = 60000) -> list[Path]:
    """Top-level standard library modules, sorted, so every run reads the same ones in the same order."""
    root = Path(sysconfig.get_paths()["stdlib"])
    return sorted(p for p in root.glob("*.py") if min_bytes <= p.stat().st_size <= max_bytes)


def _turn(role: str, text: str) -> str:
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def _read_call(name: str) -> str:
    return _turn("assistant", f"<tool_call>\n<function=read_file>\n<parameter=path>\n{name}\n</parameter>\n</function>\n</tool_call>")


def _tool_result(text: str) -> str:
    return _turn("user", f"<tool_response>\n{text}</tool_response>")


def _question(files: list[tuple[str, str]]) -> str:
    """Ask about a function from the middle file read, so the answer depends on the context."""
    if not files:
        return "Which tools do you have, and which would you use first here?"
    name, source = files[len(files) // 2]
    try:
        functions = [n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    except SyntaxError:
        functions = []
    if functions:
        return f"What does `{functions[len(functions) // 2]}` in {name} do, and which other file above would you change alongside it?"
    return f"Summarize what {name} is for in two sentences."


def build_request(
    tokenizer, files: Sequence[Path], target_tokens: int, needle: bool = False, key: str = NEEDLE, needle_file: int = 0
) -> Request:
    """One request of at most `target_tokens` tokens: whole files, then part of one more to reach the target.

    With `needle`, `key` is planted at the top of file number `needle_file` and asked for at the end."""
    reserve = 64
    while True:
        request = _build(tokenizer, files, target_tokens - reserve, needle, key, needle_file)
        if len(request.ids) <= target_tokens:
            return request
        reserve += len(request.ids) - target_tokens


def _build(tokenizer, files: Sequence[Path], target_tokens: int, needle: bool, key: str = NEEDLE, needle_file: int = 0) -> Request:
    """Fill `target_tokens` with the system prompt, task and files, then add the question and the assistant's opening."""
    head = _turn("system", SYSTEM) + _turn("user", TASK)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)  # noqa: E731
    used = count(head)
    read: list[tuple[str, str]] = []
    body = ""
    for path in files:
        name = f"lib/{path.name}"
        source = path.read_text(encoding="utf-8", errors="replace")
        if needle and len(read) == needle_file:
            source = NEEDLE_LINE.replace(NEEDLE, key) + source
        block = _read_call(name) + _tool_result(source)
        size = count(block)
        if used + size > target_tokens:
            # Keep whole lines of this file, as a tool that truncates long files would.
            lines = source.splitlines(keepends=True)
            low, high = 0, len(lines)
            while low < high:
                middle = (low + high + 1) // 2
                if used + count(_read_call(name) + _tool_result("".join(lines[:middle]))) <= target_tokens:
                    low = middle
                else:
                    high = middle - 1
            if low:
                source = "".join(lines[:low])
                read.append((name, source))
                body += _read_call(name) + _tool_result(source)
            break
        read.append((name, source))
        body += block
        used += size
    question = NEEDLE_QUESTION if needle else _question(read)
    text = head + body + _turn("user", question) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return Request(tokenizer.encode(text, add_special_tokens=False).ids, [name for name, _ in read], needle)


def benchmark_requests(tokenizer, count: int, min_tokens: int, max_tokens: int) -> list[Request]:
    """`count` distinct requests with lengths spread evenly from `min_tokens` to `max_tokens`.

    Each starts at a different file, so no two share a prefix past the system prompt and task. The longest
    carries the needle.
    """
    files = stdlib_files()
    stride = max(1, len(files) // count)
    requests = []
    for index in range(count):
        target = min_tokens + (max_tokens - min_tokens) * index // max(1, count - 1)
        start = index * stride
        ordered = files[start:] + files[:start]
        requests.append(build_request(tokenizer, ordered, target, needle=index == count - 1))
    return requests


def long_context_requests(tokenizer, lengths: Sequence[int]) -> list[tuple[Request, str]]:
    """A request per length, each with its own key: planted in the first file read for even ones, in a file
    about halfway through for odd ones. Returns (request, key) pairs."""
    files = stdlib_files()
    out = []
    for index, length in enumerate(lengths):
        start = (index * 7 + 3) % len(files)
        ordered = files[start:] + files[:start]
        # Files average about 3K tokens, so this is about halfway into the prompt.
        needle_file = 0 if index % 2 == 0 else max(1, length // 6000)
        key = KEYS[index % len(KEYS)]
        out.append((build_request(tokenizer, ordered, length, needle=True, key=key, needle_file=needle_file), key))
    return out


def prompt_digest(requests: Sequence[Request]) -> str:
    """Identifies the exact prompts, so two results can be checked to come from the same work."""
    digest = hashlib.sha256()
    for request in requests:
        digest.update(b",".join(str(i).encode() for i in request.ids) + b";")
    return digest.hexdigest()[:16]
