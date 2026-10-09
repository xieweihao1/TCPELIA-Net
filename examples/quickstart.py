"""Runnable standalone forward/backward example without YOLO dependencies."""

import torch

from tcpelia import TCPELIA


def main():
    torch.manual_seed(42)

    # Current feature (B, C_x, H, W); feedback may be coarser and have C_z != C_x.
    x_cur = torch.randn(2, 128, 40, 40, requires_grad=True)
    z_fb = torch.randn(2, 256, 20, 20, requires_grad=True)

    model = TCPELIA(c_x=128, c_z=256, c_r=64, iters=2)
    output = model([x_cur, z_fb])
    loss = output.square().mean()
    loss.backward()

    print("Current feature:", tuple(x_cur.shape))
    print("Feedback feature:", tuple(z_fb.shape))
    print("Output feature:", tuple(output.shape))
    print("Attention map:", tuple(model.attn_map.shape))
    print("Backprop through current/feedback:", x_cur.grad is not None, z_fb.grad is not None)
    print("Attention min/max:", model.attn_map.min().item(), model.attn_map.max().item())


if __name__ == "__main__":
    main()
