"""Draft tokens for speculative decoding from a sequence's own history, with no draft model.

Coding agents copy a lot: file paths, names and whole lines they have read come back in their tool calls and
edits. So the tokens that followed the last time the sequence's latest few tokens appeared are a good guess for
what comes next (prompt lookup decoding).
"""

from collections.abc import Sequence


def lookup_draft(history: Sequence[int], count: int, max_match: int = 4, min_match: int = 1) -> list[int]:
    """Up to `count` tokens that followed the latest earlier occurrence of the history's last n tokens, trying
    the longest n from `max_match` down to `min_match`. Empty when nothing matches."""
    n_tokens = len(history)
    for n in range(min(max_match, n_tokens - 1), min_match - 1, -1):
        tail = list(history[n_tokens - n :])
        # Latest earlier occurrence, so the draft follows the most recent context.
        for start in range(n_tokens - n - 1, -1, -1):
            if list(history[start : start + n]) == tail:
                follow = list(history[start + n : start + n + count])
                if follow:
                    return follow
                break
    return []


class Drafter:
    """`lookup_draft` kept up as a sequence grows, so each draft is a few dictionary lookups instead of a scan of a
    30K-token history every step. Gives the same drafts."""

    def __init__(self, history: Sequence[int], max_match: int = 4, min_match: int = 1):
        self.tokens: list[int] = []
        self.max_match, self.min_match = max_match, min_match
        # For each length n, every n tokens that something followed, mapped to where they last started.
        self.starts = [dict() for _ in range(max_match + 1)]
        self.extend(history)

    def extend(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            end = len(self.tokens)
            for n in range(1, min(self.max_match, end) + 1):
                self.starts[n][tuple(self.tokens[end - n : end])] = end - n
            self.tokens.append(token)

    def draft(self, count: int) -> list[int]:
        end = len(self.tokens)
        for n in range(min(self.max_match, end - 1), self.min_match - 1, -1):
            start = self.starts[n].get(tuple(self.tokens[end - n :]))
            if start is not None:
                return self.tokens[start + n : start + n + count]
        return []


def accepted_per_step(history: Sequence[int], output: Sequence[int], draft_len: int) -> list[int]:
    """How many tokens each verify step would emit when greedy decoding produces `output` after `history`: the
    drafted tokens that match, plus the one the model chooses itself."""
    steps, done, seen = [], 0, list(history)
    while done < len(output):
        draft = lookup_draft(seen, draft_len)
        match = 0
        while match < len(draft) and done + match < len(output) and draft[match] == output[done + match]:
            match += 1
        emitted = min(match + 1, len(output) - done)
        steps.append(emitted)
        seen += output[done : done + emitted]
        done += emitted
    return steps
