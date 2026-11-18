"""Unit tests for the model, flow utilities, and crop selection (CPU, small sizes)."""
import pytest
import torch

from mtdiff.data import sampling_weights


def test_sampling_weights_give_synthetic_rows_the_requested_total_share():
    rows = [{"synthetic": True}] * 8 + [{}] * 2
    weights = sampling_weights(rows, 0.5)
    assert abs(sum(weights[:8]) - 0.5) < 1e-9
    assert abs(sum(weights[8:]) - 0.5) < 1e-9


def test_sampling_weights_are_uniform_without_synthetic_rows():
    assert sampling_weights([{}, {}, {}], 0.5) == [1.0, 1.0, 1.0]


from mtdiff.config import ModelConfig
from mtdiff.data import glyph_channels, pick_crop_origin
from mtdiff.flow import euler_sample, make_noisy, start_point, velocity_target, weighted_flow_loss
from mtdiff.model import DiT, apply_rope, axial_rope

TINY = ModelConfig(latent_channels=4, extra_channels=0, dim=64, depth=2, heads=4)


def test_model_output_matches_latent_shape_for_non_square_input():
    model = DiT(TINY)
    noisy, source = torch.randn(2, 4, 8, 12), torch.randn(2, 4, 8, 12)

    out = model(noisy, source, torch.rand(2))

    assert out.shape == (2, 4, 8, 12)


def test_model_predicts_zero_velocity_at_initialisation():
    model = DiT(TINY)

    out = model(torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8), torch.rand(1))

    assert torch.count_nonzero(out) == 0


def test_model_output_depends_on_source_after_head_is_trained():
    model = DiT(TINY)
    torch.nn.init.normal_(model.head.weight, std=0.02)
    noisy, t = torch.randn(1, 4, 8, 8), torch.tensor([0.5])

    first = model(noisy, torch.zeros(1, 4, 8, 8), t)
    second = model(noisy, torch.ones(1, 4, 8, 8), t)

    assert not torch.allclose(first, second)


def test_extra_channels_are_accepted_when_configured():
    model = DiT(ModelConfig(latent_channels=4, extra_channels=3, dim=64, depth=1, heads=4))

    out = model(torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8), torch.rand(1), torch.randn(1, 3, 8, 8))

    assert out.shape == (1, 4, 8, 8)


def test_unpatchify_inverts_patch_layout():
    model = DiT(TINY)
    image = torch.randn(1, 4, 8, 12)
    patches = image.reshape(1, 4, 4, 2, 6, 2).permute(0, 2, 4, 3, 5, 1).reshape(1, 24, 16)

    assert torch.equal(model.unpatchify(patches, 4, 6), image)


def test_rope_preserves_norm_and_leaves_origin_token_unchanged():
    x = torch.randn(1, 2, 12, 16)
    rope = axial_rope(3, 4, 16, x.device)

    rotated = apply_rope(x, rope)

    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    assert torch.allclose(rotated[:, :, 0], x[:, :, 0], atol=1e-6)


def test_attention_rejects_head_dim_not_divisible_by_four():
    with pytest.raises(ValueError):
        DiT(ModelConfig(latent_channels=4, extra_channels=0, dim=60, depth=1, heads=10))


def test_noisy_sample_interpolates_between_data_and_noise():
    target, noise = torch.zeros(2, 1, 2, 2), torch.ones(2, 1, 2, 2)

    noisy = make_noisy(target, noise, torch.tensor([0.0, 1.0]))

    assert torch.equal(noisy[0], target[0]) and torch.equal(noisy[1], noise[1])
    assert torch.equal(velocity_target(target, noise), noise - target)


def test_flow_loss_equals_mse_without_mask_weight():
    pred, velocity = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)

    loss = weighted_flow_loss(pred, velocity, torch.ones(2, 1, 4, 4), mask_weight=0.0)

    assert torch.allclose(loss, torch.nn.functional.mse_loss(pred, velocity))


def test_flow_loss_upweights_masked_errors():
    pred, velocity = torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)
    pred[0, 0, 0, 0] = 1.0
    mask = torch.zeros(1, 1, 2, 2)
    mask[0, 0, 0, 0] = 1.0

    plain = weighted_flow_loss(pred, velocity, mask, mask_weight=0.0)
    weighted = weighted_flow_loss(pred, velocity, mask, mask_weight=5.0)

    assert weighted > plain


def test_euler_sample_recovers_target_for_exact_velocity_field():
    target = torch.full((1, 4, 8, 8), 0.3)

    class Oracle(torch.nn.Module):
        def forward(self, x, source, t, extra=None):
            return (x - target) / t.view(-1, 1, 1, 1)

    out = euler_sample(Oracle(), torch.zeros(1, 4, 8, 8), steps=20, generator=torch.Generator().manual_seed(0))

    assert torch.allclose(out, target, atol=1e-4)


def test_text_centred_crop_contains_the_masked_pixel():
    mask = torch.zeros(40, 60, dtype=torch.uint8)
    mask[30, 50] = 1
    rng = torch.Generator().manual_seed(0)

    for _ in range(20):
        y, x = pick_crop_origin(mask, crop=16, text_center_prob=1.0, rng=rng)
        assert 0 <= y <= 24 and 0 <= x <= 44
        assert y <= 30 < y + 16 and x <= 50 < x + 16


def test_random_crop_stays_in_bounds_when_mask_is_empty():
    rng = torch.Generator().manual_seed(0)

    for _ in range(20):
        y, x = pick_crop_origin(torch.zeros(20, 30, dtype=torch.uint8), crop=16, text_center_prob=1.0, rng=rng)
        assert 0 <= y <= 4 and 0 <= x <= 14


def test_start_point_is_noise_for_baseline_and_noisy_source_for_v2():
    source, noise = torch.full((1, 2, 2, 2), 3.0), torch.ones(1, 2, 2, 2)

    assert torch.equal(start_point(source, noise, from_source=False, noise_scale=0.5), noise)
    assert torch.equal(start_point(source, noise, from_source=True, noise_scale=0.5), source + 0.5)


def test_euler_sample_from_source_keeps_start_when_velocity_is_zero():
    source = torch.randn(1, 4, 8, 8)

    class Still(torch.nn.Module):
        def forward(self, x, source, t, extra=None):
            return torch.zeros_like(x)

    out = euler_sample(Still(), source, steps=4, from_source=True, noise_scale=0.0)

    assert torch.equal(out, source)


def test_euler_sample_passes_extra_condition_to_model():
    seen = []

    class Spy(torch.nn.Module):
        def forward(self, x, source, t, extra=None):
            seen.append(extra)
            return torch.zeros_like(x)

    extra = torch.ones(1, 3, 8, 8)
    euler_sample(Spy(), torch.zeros(1, 4, 8, 8), steps=2, extra=extra)

    assert all(item is extra for item in seen) and len(seen) == 2


def test_glyph_channels_fold_and_unfold_losslessly():
    glyph = torch.randint(0, 256, (64, 96), dtype=torch.uint8)

    folded = glyph_channels(glyph, y=2, x=4, crop=4)

    assert folded.shape == (64, 4, 4)
    restored = torch.nn.functional.pixel_shuffle(folded[None], 8)[0, 0] * 255
    assert torch.equal(restored.round().to(torch.uint8), glyph[16:48, 32:64])
