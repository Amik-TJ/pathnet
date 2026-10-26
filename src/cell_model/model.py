from __future__ import annotations

import torch
from torch import nn

ARMS = ("mlp_raw", "mlp_norm", "kernel", "kernel_sep")


def mlp(n_in: int, n_out: int, width: int, depth: int, act=nn.SiLU) -> nn.Sequential:
    layers, n = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(n, width), act()]
        n = width
    return nn.Sequential(*layers, nn.Linear(n, n_out))


def squash(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


class MlpRaw(nn.Module):
    def __init__(self, width=512, depth=4):
        super().__init__()
        self.net = mlp(3 + 98, 2, width, depth)
        self.register_buffer("in_mean", torch.zeros(101)); self.register_buffer("in_std", torch.ones(101))
        self.register_buffer("y_mean", torch.zeros(2)); self.register_buffer("y_std", torch.ones(2))

    def inputs(self, b):
        return torch.cat([b["s"][:, None], b["c"][:, None], b["cap0"][:, None], b["tables_raw"].flatten(1)], 1)

    def fit_stats(self, b):
        x = self.inputs(b)
        self.in_mean.copy_(x.mean(0)); self.in_std.copy_(x.std(0).clamp_min(1e-6))
        self.y_mean.copy_(b["y"].mean(0)); self.y_std.copy_(b["y"].std(0))

    def forward(self, b):
        out = self.net((self.inputs(b) - self.in_mean) / self.in_std)
        return out * self.y_std + self.y_mean, out, (b["y"] - self.y_mean) / self.y_std


class MlpNorm(nn.Module):
    def __init__(self, width=512, depth=4):
        super().__init__()
        self.net = mlp(2 + 98, 2, width, depth)

    def fit_stats(self, b):
        pass

    def forward(self, b):
        out = self.net(torch.cat([b["g"], squash(b["tables_norm"]).flatten(1)], 1))
        return out * b["scale"] + b["offset"], out, b["y_norm"]


class Kernel(nn.Module):
    def __init__(self, width=512, depth=4):
        super().__init__()
        self.net = mlp(2, 49, width, depth)

    def fit_stats(self, b):
        pass

    def weights(self, g):
        w = self.net(g)
        return w - w.mean(1, keepdim=True) + 1.0 / 49.0

    def forward(self, b):
        w = self.weights(b["g"])
        out = (w[:, None, :] * b["tables_norm"].flatten(2)).sum(-1)
        return out * b["scale"] + b["offset"], out, b["y_norm"]


class KernelSep(Kernel):
    def __init__(self, width=512, depth=4):
        nn.Module.__init__(self)
        self.net_u = mlp(1, 7, width, depth, nn.ReLU)
        self.net_v = mlp(1, 7, width, depth, nn.ReLU)

    def weights(self, g):
        wu, wv = self.net_u(g[:, :1]), self.net_v(g[:, 1:])
        wu = wu - wu.mean(1, keepdim=True) + 1.0 / 7.0
        wv = wv - wv.mean(1, keepdim=True) + 1.0 / 7.0
        return (wu[:, :, None] * wv[:, None, :]).flatten(1)


def make_model(arm: str, width: int = 512, depth: int = 4) -> nn.Module:
    return {"mlp_raw": MlpRaw, "mlp_norm": MlpNorm, "kernel": Kernel, "kernel_sep": KernelSep}[arm](width, depth)
