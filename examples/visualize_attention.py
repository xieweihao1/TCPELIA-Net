"""Save a visual inspection of synthetic attention (NOT a learned benchmark result).

Requires: pip install matplotlib
Usage: python -m examples.visualize_attention
"""

from pathlib import Path

import torch

from tcpelia import TCPELIA


def main():
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    current = torch.randn(1, 64, 64, 64)
    feedback = torch.randn(1, 128, 32, 32)
    module = TCPELIA(c_x=64, c_z=128, c_r=64, iters=2).eval()
    with torch.no_grad():
        _ = module([current, feedback])
        attn = module.attn_map[0, 0].cpu()

    destination = Path("attention_demo.png")
    plt.figure(figsize=(5, 4))
    plt.imshow(attn, cmap="inferno", vmin=0, vmax=1)
    plt.colorbar(label="Attention intensity")
    plt.title("TCPELIA attention on random synthetic inputs")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()
    print(f"Saved {destination.resolve()} (synthetic visualization only)")


if __name__ == "__main__":
    main()
