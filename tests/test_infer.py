import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_weights import export  # noqa: E402
from mtdiff.config import ModelConfig, TrainConfig  # noqa: E402
from mtdiff.glyph import find_font, line_rect  # noqa: E402
from mtdiff.infer import Region, load_model, pad_edges, paste_mask, parse_regions, scale_regions, to_lines  # noqa: E402
from mtdiff.model import DiT  # noqa: E402


def test_parse_regions_accepts_list_and_wrapped_forms_with_optional_erase():
    item = {"box": [10, 20, 50, 120], "text": "别担心", "erase": [0, 0, 60, 130]}
    assert parse_regions([item]) == parse_regions({"regions": [item]})
    assert parse_regions([item])[0].erase == (0.0, 0.0, 60.0, 130.0)
    assert parse_regions([{"box": [1, 2, 3, 4], "text": "好"}])[0].erase is None


@pytest.mark.parametrize("bad", [{"box": [5, 5, 5, 9], "text": "好"}, {"box": [1, 2, 3], "text": "好"}, {"box": [0, 0, 4, 4], "text": ""}])
def test_parse_regions_rejects_invalid_entries(bad):
    with pytest.raises(ValueError):
        parse_regions([bad])


def test_scale_regions_scales_boxes_and_erase_areas():
    scaled = scale_regions([Region((10, 20, 30, 40), "好", (0, 0, 50, 60))], 2.0)
    assert scaled[0].box == (20, 40, 60, 80)
    assert scaled[0].erase == (0, 0, 100, 120)


def test_to_lines_builds_rectangular_polygons():
    line = to_lines([Region((10, 20, 30, 40), "好")])[0]
    assert line_rect(line, 100, 100) == (10, 20, 30, 40)


def test_paste_mask_is_soft_covers_the_region_and_is_zero_far_away():
    mask = paste_mask([Region((100, 100, 140, 180), "好")], 400, 400, margin=24)
    assert mask.shape == (400, 400)
    assert mask[140, 120] > 0.99
    assert mask[0, 0] == 0
    assert 0 < mask[100 - 24, 120] < 1  # feathered edge


def test_paste_mask_prefers_the_erase_area_and_handles_no_regions():
    mask = paste_mask([Region((10, 10, 20, 20), "好", (100, 100, 200, 200))], 300, 300, margin=0)
    assert mask[150, 150] > 0.99 and mask[15, 15] == 0
    assert not paste_mask([], 50, 50, 24).any()


def test_pad_edges_rounds_up_to_the_multiple_and_replicates_the_border():
    padded = pad_edges(np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3), multiple=4)
    assert padded.shape == (4, 4, 3)
    assert (padded[3, 3] == padded[1, 2]).all()


def test_export_round_trip_matches_the_ema_weights(tmp_path):
    cfg = TrainConfig(model=ModelConfig(dim=64, depth=2, heads=4))
    model = DiT(cfg.model)
    torch.save({"ema": model.state_dict(), "config": cfg.to_dict(), "step": 7}, tmp_path / "ckpt.pt")
    config = export(tmp_path / "ckpt.pt", tmp_path / "out")
    assert config["training_step"] == 7 and config["vae"]["repo"].startswith("black-forest-labs")
    assert json.loads((tmp_path / "out" / "config.json").read_text())["model"]["dim"] == 64
    loaded, _ = load_model(tmp_path / "out", torch.device("cpu"))
    for name, value in model.state_dict().items():
        assert torch.allclose(loaded.state_dict()[name].float(), value, atol=1e-2, rtol=1e-2), name


def test_find_font_honours_the_environment_override(monkeypatch):
    monkeypatch.setenv("MTDIFF_FONT", "/some/font.ttc:3")
    assert find_font() == ("/some/font.ttc", 3)


@pytest.mark.parametrize("override,expected", [
    ("C:/Windows/Fonts/msyh.ttc", ("C:/Windows/Fonts/msyh.ttc", 0)),
    ("C:/Windows/Fonts/msyh.ttc:2", ("C:/Windows/Fonts/msyh.ttc", 2)),
    (r"C:\Windows\Fonts\msyh.ttc:1", (r"C:\Windows\Fonts\msyh.ttc", 1)),
    ("/some/font.ttc", ("/some/font.ttc", 0)),
])
def test_find_font_preserves_drive_letters_and_optional_face(override, expected, monkeypatch):
    monkeypatch.setenv("MTDIFF_FONT", override)
    assert find_font() == expected


@pytest.mark.parametrize("field", ["box", "erase"])
@pytest.mark.parametrize("coordinates", [
    [], [0, 0, 4], [0, 0, 4, 4, 5], [4, 0, 4, 4], [4, 0, 1, 4],
    [0, 4, 4, 4], [0, 4, 4, 1], [0, 0, float("nan"), 4],
    [0, 0, float("inf"), 4], [float("-inf"), 0, 4, 4],
    [0, 0, "bad", 4], [0, 0, None, 4], 3,
])
def test_parse_regions_rejects_malformed_or_nonfinite_rectangles(field, coordinates):
    item = {"box": [0, 0, 4, 4], "text": "好", field: coordinates}
    with pytest.raises(ValueError, match=f"invalid {field}"):
        parse_regions([item])


@pytest.mark.parametrize("vae,expected_subfolder", [(None, "vae"), ("local-vae", None)])
def test_translator_keeps_model_and_vae_on_requested_device(monkeypatch, vae, expected_subfolder):
    import mtdiff.infer as infer

    devices = []
    codec_calls = []
    config = {"tile": 64, "tile_stride": 48, "vae": {"repo": "test-vae", "subfolder": "vae"}}

    def load_model(weights_dir, device):
        devices.append(str(device))
        return torch.nn.Identity(), config

    def codec(repo, device="cuda", subfolder=None):
        codec_calls.append((repo, str(device), subfolder))
        return object()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(infer, "load_model", load_model)
    monkeypatch.setattr(infer, "LatentCodec", codec)
    infer.PageTranslator("unused", vae=vae, device="cuda:1")
    assert devices == ["cuda:1"]
    assert codec_calls == [(vae or "test-vae", "cuda:1", expected_subfolder)]


def test_cli_reports_failed_image_write(monkeypatch, tmp_path):
    import translate_page

    regions = tmp_path / "regions.json"
    regions.write_text("[]", encoding="utf-8")
    output = tmp_path / "out.png"
    page = np.zeros((16, 16, 3), dtype=np.uint8)

    class Translator:
        def __init__(self, *args, **kwargs):
            pass

        def translate(self, page, *args):
            return page

    monkeypatch.setattr(sys, "argv", ["translate_page.py", "page.png", str(regions), str(output),
                                      "--weights", "unused", "--width", "0"])
    monkeypatch.setattr(translate_page.cv2, "imread", lambda *args: page)
    monkeypatch.setattr(translate_page.cv2, "imwrite", lambda *args: False)
    monkeypatch.setattr(translate_page, "PageTranslator", Translator)
    with pytest.raises(SystemExit, match="cannot write image"):
        translate_page.main()
