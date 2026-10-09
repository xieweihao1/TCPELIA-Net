"""Unit tests for the user-provided TCPELIA mathematical implementation."""

import io

import pytest
import torch

from tcpelia import TCPELIA


@pytest.mark.parametrize("iters", [1, 2, 5])
def test_shape_range_and_backward(iters):
    torch.manual_seed(5)
    module = TCPELIA(c_x=16, c_z=32, c_r=8, iters=iters)
    current = torch.randn(2, 16, 13, 17, requires_grad=True)
    feedback = torch.randn(2, 32, 7, 9, requires_grad=True)

    output = module([current, feedback])

    assert output.shape == current.shape
    assert module.attn_map.shape == (2, 1, 13, 17)
    assert torch.isfinite(output).all()
    assert torch.isfinite(module.attn_map).all()
    assert bool((module.attn_map >= 0).all())
    assert bool((module.attn_map <= 1).all())
    assert not module.attn_map.requires_grad  # inspection-only buffer by design
    output.mean().backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert feedback.grad is not None and torch.isfinite(feedback.grad).all()
    assert module.proj_x.weight.grad is not None
    assert module.proj_z.weight.grad is not None


def test_tensor_input_self_feedback_and_non_square_spatial_dims():
    module = TCPELIA(c_x=16, c_z=16, c_r=8, iters=2)
    current = torch.randn(1, 16, 11, 15)
    assert module(current).shape == current.shape


def test_output_matches_multiplicative_residual_formula():
    module = TCPELIA(c_x=16, c_z=16, c_r=8, iters=2, lam=0.75)
    x = torch.randn(1, 16, 9, 11)
    z = torch.randn(1, 16, 5, 7)
    y = module([x, z])
    expected = x * (1 + module.lam * module.attn_map)
    torch.testing.assert_close(y, expected)


def test_inhibition_kernel_center_and_normalization():
    module = TCPELIA(c_x=16, c_z=16, c_r=8)
    assert module.k_inh.shape == (1, 1, 5, 5)
    assert module.k_inh[0, 0, 2, 2].item() == 0
    torch.testing.assert_close(module.k_inh.sum(), torch.tensor(1.0))


def test_state_dict_round_trip():
    torch.manual_seed(7)
    model = TCPELIA(c_x=16, c_z=32, c_r=8, iters=2).eval()
    x = torch.randn(1, 16, 9, 15)
    z = torch.randn(1, 32, 5, 7)
    with torch.no_grad():
        y1 = model([x, z])
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = TCPELIA(c_x=16, c_z=32, c_r=8, iters=2).eval()
    # weights_only exists in newer PyTorch; this archive is created locally in the test.
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    with torch.no_grad():
        y2 = restored([x, z])
    torch.testing.assert_close(y1, y2)
