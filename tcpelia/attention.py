import torch
import torch.nn as nn
import torch.nn.functional as F


class TCPELIA(nn.Module):
    """
    Thalamo-Cortical Predictive Error Lateral-Inhibition Attention

    YAML:
        - [[15, 10], 1, TCPELIA, [64, 2]]

    parse_model 实际初始化:
        TCPELIA(c_x, c_z, c_r=64, iters=2)

    forward 实际输入:
        x = [x_cur, z_fb]

    输出:
        y_cur，尺寸与 x_cur 一致
    """

    def __init__(
        self,
        c_x,
        c_z,
        c_r=64,
        iters=5,
        rho=0.5,
        alpha=2.0,
        beta=1.0,
        lam=0.75,
        U0=0.2,
        Uf=0.2,
        tau_a=1.5,
        tau_h=1.0,
        tau_d=5.0,
        tau_f=2.0,
    ):
        super().__init__()

        self.c_x = c_x
        self.c_z = c_z
        self.c_r = c_r
        self.iters = iters

        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.U0 = U0
        self.Uf = Uf
        self.tau_a = tau_a
        self.tau_h = tau_h
        self.tau_d = tau_d
        self.tau_f = tau_f

        # 显式通道，不再使用 LazyConv2d
        self.proj_x = nn.Conv2d(c_x, c_r, kernel_size=1, bias=False)
        self.proj_z = nn.Conv2d(c_z, c_r, kernel_size=1, bias=False)

        self.norm_x = nn.GroupNorm(8, c_r)
        self.norm_z = nn.GroupNorm(8, c_r)

        self.fb_smooth = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)

        k = torch.tensor([
            [0.02, 0.04, 0.05, 0.04, 0.02],
            [0.04, 0.06, 0.08, 0.06, 0.04],
            [0.05, 0.08, 0.00, 0.08, 0.05],
            [0.04, 0.06, 0.08, 0.06, 0.04],
            [0.02, 0.04, 0.05, 0.04, 0.02],
        ], dtype=torch.float32)

        k = k / k.sum()
        self.register_buffer("k_inh", k.view(1, 1, 5, 5))

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x_cur, z_fb = x
        else:
            x_cur = x
            z_fb = x

        xr = self.norm_x(self.proj_x(x_cur))
        zr = self.norm_z(self.proj_z(z_fb))

        if zr.shape[-2:] != xr.shape[-2:]:
            zr = F.interpolate(
                zr,
                size=xr.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        B, _, H, W = xr.shape

        a = xr.new_zeros(B, 1, H, W)
        h = xr.new_zeros(B, 1, H, W)
        r = xr.new_ones(B, 1, H, W)
        u = xr.new_full((B, 1, H, W), self.U0)

        eta_a = 1.0 / self.tau_a
        eta_h = 1.0 / self.tau_h

        for _ in range(self.iters):
            fb = torch.sigmoid(self.fb_smooth(a))

            x_hat = zr * (1.0 + self.rho * fb)

            q = (xr - x_hat).pow(2).mean(dim=1, keepdim=True)

            h_new = (1.0 - eta_h) * h + eta_h * F.conv2d(
                a, self.k_inh, padding=2
            )

            r_new = r + ((1.0 - r) / self.tau_d - u * r * q)
            r_new = torch.clamp(r_new, 0.0, 1.0)

            u_new = u + ((self.U0 - u) / self.tau_f + self.Uf * (1.0 - u) * q)
            u_new = torch.clamp(u_new, self.U0, 1.0)

            g = r_new * u_new

            a_prop = torch.sigmoid(self.alpha * g * q - self.beta * h_new)
            a_new = (1.0 - eta_a) * a + eta_a * a_prop

            a, h, r, u = a_new, h_new, r_new, u_new

        self.attn_map = a.detach()

        y = x_cur * (1.0 + self.lam * a)

        return y