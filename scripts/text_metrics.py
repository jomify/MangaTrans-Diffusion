"""Pure helpers for scoring rendered text with OCR: edit distance and character accuracy."""
import re

CJK = re.compile(r"[一-鿿]")


def edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance between two strings."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_char != hyp_char)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitution))
        previous = current
    return previous[-1]


def keep_cjk(text: str) -> str:
    return "".join(CJK.findall(text))


def character_accuracy(pairs: list[tuple[str, str]]) -> float:
    """1 - (total edit distance / total reference length), floored at 0; pairs are (reference, hypothesis)."""
    total = sum(len(reference) for reference, _ in pairs)
    if total == 0:
        return float("nan")
    errors = sum(edit_distance(reference, hypothesis) for reference, hypothesis in pairs)
    return max(0.0, 1.0 - errors / total)


def summarize(pairs: list[tuple[str, str]]) -> dict[str, float]:
    """Accuracy over all characters, over CJK characters only, and the share of lines read exactly."""
    cjk_pairs = [(keep_cjk(reference), keep_cjk(hypothesis)) for reference, hypothesis in pairs]
    exact = sum(reference == hypothesis for reference, hypothesis in cjk_pairs)
    return {
        "lines": len(pairs),
        "chars": sum(len(reference) for reference, _ in pairs),
        "char_accuracy": character_accuracy(pairs),
        "cjk_accuracy": character_accuracy(cjk_pairs),
        "line_exact": exact / len(pairs) if pairs else float("nan"),
    }
