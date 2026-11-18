import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from text_metrics import character_accuracy, edit_distance, keep_cjk, summarize  # noqa: E402


def test_edit_distance_is_zero_for_identical_strings():
    assert edit_distance("别担心", "别担心") == 0


def test_edit_distance_counts_substitution_insertion_and_deletion():
    assert edit_distance("别担心", "别但心") == 1
    assert edit_distance("别担心", "别担心啊") == 1
    assert edit_distance("别担心", "别心") == 1


def test_edit_distance_against_empty_string_is_the_length():
    assert edit_distance("感觉", "") == 2
    assert edit_distance("", "感觉") == 2


def test_keep_cjk_drops_punctuation_and_latin():
    assert keep_cjk("啊啊……ok,早知道!") == "啊啊早知道"


def test_character_accuracy_pools_errors_over_all_lines():
    pairs = [("别担心", "别担心"), ("预算不足", "预算不")]
    assert character_accuracy(pairs) == 1 - 1 / 7


def test_character_accuracy_is_floored_at_zero_and_nan_when_empty():
    assert character_accuracy([("好", "一二三四")]) == 0.0
    assert math.isnan(character_accuracy([]))


def test_summarize_reports_exact_lines_on_cjk_only():
    summary = summarize([("好的。", "好的"), ("不行", "不")])
    assert summary["lines"] == 2
    assert summary["chars"] == 5
    assert summary["line_exact"] == 0.5
    assert summary["cjk_accuracy"] == 0.75
