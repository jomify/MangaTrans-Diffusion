import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_synth import common_characters, grow, make_text, paper_colour  # noqa: E402
from build_v2 import layout_line  # noqa: E402


def test_common_characters_is_the_gb2312_level_one_set():
    charset = common_characters()
    assert len(charset) == 3755
    assert "的" in charset and "啊" in charset


def test_make_text_has_the_requested_length_from_both_sources():
    charset = common_characters()
    corpus = ["别担心", "预算不足，但应该勉强撑得住吧。"]
    rng = random.Random(0)
    for length in (1, 2, 7, 30):
        for _ in range(20):
            assert len(make_text(length, rng, corpus, charset)) == length
    assert make_text(0, rng, corpus, charset) == ""


def test_make_text_is_deterministic_for_a_seed():
    charset = common_characters()
    assert make_text(12, random.Random("a"), [], charset) == make_text(12, random.Random("a"), [], charset)


def test_paper_colour_accepts_text_on_white_paper():
    image = np.full((60, 60, 3), 250, np.uint8)
    image[20:40, 25:35] = 0  # the text itself, inside the box
    colour = paper_colour(image, (25, 20, 35, 40))
    assert colour is not None and colour.min() >= 245


def test_paper_colour_rejects_dark_and_patterned_backgrounds():
    dark = np.full((60, 60, 3), 90, np.uint8)
    assert paper_colour(dark, (25, 20, 35, 40)) is None
    dots = np.full((60, 60, 3), 255, np.uint8)
    dots[::2, ::2] = 0
    dots[1::2, 1::2] = 0
    assert paper_colour(dots, (25, 20, 35, 40)) is None


def test_paper_colour_rejects_text_drawn_over_artwork():
    image = np.full((60, 60, 3), 250, np.uint8)
    image[20:40, 25:35] = 150  # mid-tone artwork under the box, white around it
    assert paper_colour(image, (25, 20, 35, 40)) is None


def test_grow_clamps_to_the_image():
    assert grow((1, 1, 10, 10), 3, 12, 12) == (0, 0, 12, 12)


def test_layout_line_sets_tall_boxes_vertically():
    size, cells = layout_line((10, 0, 30, 100), "一二三四五")
    assert size == int(20 * 0.92)
    assert [char for _, char in cells] == list("一二三四五")
    assert all(centre[0] == 20 for centre, _ in cells)
    assert cells[0][0][1] == 10 and cells[-1][0][1] == 90


def test_layout_line_sets_wide_boxes_horizontally_and_skips_empty_text():
    _, cells = layout_line((0, 0, 100, 20), "一二")
    assert [centre for centre, _ in cells] == [(25.0, 10.0), (75.0, 10.0)]
    assert layout_line((0, 0, 10, 10), "") == (0, [])
