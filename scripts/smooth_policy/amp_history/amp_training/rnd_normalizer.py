"""Running normalizer for RND feature inputs and rewards."""

from __future__ import annotations

import torch


class RunningMeanStd:
    """Simple running mean/std estimator for 1D feature tensors."""

    def __init__(self, shape, device: str = "cuda", clip: float | None = 10.0):
        self.shape = shape
        self.device = device
        self.clip = clip
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(1e-4, device=device)

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False).clamp_min(1e-8)
        batch_count = torch.tensor(float(x.shape[0]), device=self.device)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var.clamp_min(1e-8)
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        out = (x - self.mean) / (torch.sqrt(self.var) + 1e-8)
        if self.clip is not None:
            out = torch.clamp(out, -self.clip, self.clip)
        return out

    def state_dict(self):
        return {
            "shape": self.shape,
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
            "clip": self.clip,
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].to(self.device)
        self.var = state_dict["var"].to(self.device)
        self.count = state_dict["count"].to(self.device)
        self.clip = state_dict.get("clip", self.clip)
