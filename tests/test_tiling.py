import torch
from torch import nn

from mtdiff.tiling import TiledModel, tile_origins, tile_weight


class Identity(nn.Module):
    def forward(self, noisy, source, t, extra=None):
        return noisy if extra is None else noisy + extra


def test_tile_origins_cover_the_length_and_end_flush():
    origins = tile_origins(100, 64, 48)
    assert origins[0] == 0
    assert origins[-1] == 100 - 64
    covered = set()
    for origin in origins:
        covered.update(range(origin, origin + 64))
    assert covered == set(range(100))


def test_tile_origins_single_tile_when_length_fits():
    assert tile_origins(64, 64, 48) == [0]
    assert tile_origins(40, 64, 48) == [0]


def test_tile_weight_is_positive_and_peaks_in_the_centre():
    weight = tile_weight(8, torch.device("cpu"))
    assert weight.shape == (8, 8)
    assert (weight > 0).all()
    assert weight[4, 4] > weight[0, 0]


def test_tiled_model_reproduces_a_pointwise_model_exactly():
    torch.manual_seed(0)
    noisy = torch.randn(1, 4, 40, 56)
    extra = torch.randn(1, 4, 40, 56)
    tiled = TiledModel(Identity(), tile=16, stride=12, batch=5)
    out = tiled(noisy, noisy, torch.ones(1), extra)
    assert torch.allclose(out, noisy + extra, atol=1e-5)


def test_tiled_model_falls_back_when_the_input_is_smaller_than_a_tile():
    noisy = torch.randn(2, 4, 8, 8)
    out = TiledModel(Identity(), tile=16, stride=12)(noisy, noisy, torch.ones(2))
    assert torch.equal(out, noisy)
