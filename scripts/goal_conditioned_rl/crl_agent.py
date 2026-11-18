"""Contrastive RL (Eysenbach et al. 2022) in PyTorch, with the ``c0`` variant as switches.

Two presets (``CRLConfig.paper()`` / ``CRLConfig.c0()``), every difference a field:

=====================  =====================================  ==========================================
field                  ``paper`` (official contrastive_rl)    ``c0`` (scripts/goal_conditioned_rl/c0.py)
=====================  =====================================  ==========================================
``critic_loss``        ``nce``: sigmoid BCE vs identity,      ``infonce``: row softmax CE +
                       mean over the B x B matrix             ``lse_coef`` * logsumexp(row)^2 (0.1)
``network``            ``mlp``: 2 x 256 ReLU                  ``residual``: Dense-LayerNorm-swish,
                                                              depth // 4 residual blocks of 4 (w 256, d 8)
``actor_head``         ``acme``: tanh(N(loc, softplus+1e-6))  ``c0``: tanh(N(mu, exp(log_std))),
                                                              log_std = -5 + 3.5 (tanh(x) + 1)
``entropy``            ``off`` (entropy_coefficient 0 for     ``auto``: alpha = exp(log_value), dual on
                       state-based envs)                      alpha (target -0.5 * action_dim)
``adam_eps``           1e-7                                   1e-8 (optax default)
``skip_nonfinite``     False                                  True
=====================  =====================================  ==========================================

Shared: phi(s, a) . psi(g) critic (repr 64), no target networks, no reward; the actor
maximises phi(s, pi(s, g)) . psi(g) through the reparameterised sample with the critic
held at its pre-update parameters; ``random_goals`` picks the actor-loss goals exactly as
the official code (0 = each row's own future goal, 0.5 = batch doubled: own goals + goals
rolled by one row, 1 = rolled only). Adam 3e-4, batch 256.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- config
@dataclass
class CRLConfig:
    critic_loss: str = "nce"  # "nce" | "infonce"
    lse_coef: float = 0.0  # logsumexp^2 penalty (infonce only)
    network: str = "mlp"  # "mlp" | "residual"
    hidden_sizes: tuple = (256, 256)  # mlp
    width: int = 256  # residual
    depth: int = 8  # residual (multiple of 4)
    repr_dim: int = 64
    actor_head: str = "acme"  # "acme" | "c0"
    actor_min_std: float = 1e-6  # acme head
    entropy: str = "off"  # "off" | "auto"
    entropy_param: float = 0.5  # target entropy = -entropy_param * action_dim (auto only)
    random_goals: float = 0.5  # 0.0 | 0.5 | 1.0
    lr: float = 3e-4
    adam_eps: float = 1e-7
    batch_size: int = 256
    skip_nonfinite: bool = False
    # Not in CRL / c0: set goal dims [2:] (puck vx, vy) of the actor-loss goals to 0, i.e. train the actor to
    # reach a stationary puck at the goal position. The critic still trains on true hindsight future states.
    actor_goal_zero_velocity: bool = False

    @staticmethod
    def paper(**kw) -> "CRLConfig":
        return CRLConfig(**kw)

    @staticmethod
    def c0(**kw) -> "CRLConfig":
        base = dict(critic_loss="infonce", lse_coef=0.1, network="residual", actor_head="c0",
                    entropy="auto", adam_eps=1e-8, skip_nonfinite=True)
        base.update(kw)
        return CRLConfig(**base)

    @staticmethod
    def from_dict(d: dict) -> "CRLConfig":
        d = dict(d)
        preset = d.pop("preset", "paper")
        if "hidden_sizes" in d:
            d["hidden_sizes"] = tuple(d["hidden_sizes"])
        return CRLConfig.c0(**d) if preset == "c0" else CRLConfig.paper(**d)

    def validate(self):
        assert self.critic_loss in ("nce", "infonce"), self.critic_loss
        assert self.network in ("mlp", "residual"), self.network
        assert self.actor_head in ("acme", "c0"), self.actor_head
        assert self.entropy in ("off", "auto"), self.entropy
        assert self.random_goals in (0.0, 0.5, 1.0), self.random_goals
        assert self.network != "residual" or (self.depth >= 4 and self.depth % 4 == 0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d


# ----------------------------------------------------------------------------- init helpers
def _variance_scaling_uniform_(w: torch.Tensor, scale: float, mode: str):
    """JAX/Haiku VarianceScaling(scale, mode, 'uniform') for a torch Linear weight [out, in]."""
    fan_out, fan_in = w.shape
    fan = {"fan_in": fan_in, "fan_out": fan_out, "fan_avg": 0.5 * (fan_in + fan_out)}[mode]
    bound = math.sqrt(3.0 * scale / fan)
    with torch.no_grad():
        w.uniform_(-bound, bound)


def _linear(n_in: int, n_out: int, scale: float, mode: str) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    _variance_scaling_uniform_(layer.weight, scale, mode)
    nn.init.zeros_(layer.bias)
    return layer


# ----------------------------------------------------------------------------- networks
class MLP(nn.Module):
    """Haiku hk.nets.MLP: Linear-ReLU for every hidden layer; final ReLU iff ``activate_final``."""

    def __init__(self, n_in: int, sizes: list, init_mode: str, activate_final: bool = False):
        super().__init__()
        layers, prev = [], n_in
        for s in sizes:
            layers.append(_linear(prev, s, 1.0, init_mode))
            prev = s
        self.layers = nn.ModuleList(layers)
        self.activate_final = activate_final
        self.out_dim = prev

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.activate_final:
                x = F.relu(x)
        return x


class ResidualMLP(nn.Module):
    """c0.py ``ResidualMLP``: Dense -> LayerNorm -> swish, then depth // 4 residual blocks of four such layers."""

    def __init__(self, n_in: int, width: int, depth: int, out_dim: Optional[int]):
        super().__init__()
        scale, mode = 1.0 / 3.0, "fan_in"  # c0 _init = variance_scaling(1/3, 'fan_in', 'uniform')
        self.first = _linear(n_in, width, scale, mode)
        self.first_ln = nn.LayerNorm(width, eps=1e-6)  # flax LayerNorm epsilon
        self.blocks = nn.ModuleList()
        for _ in range(depth // 4):
            self.blocks.append(nn.ModuleList([_linear(width, width, scale, mode) for _ in range(4)]))
        self.block_lns = nn.ModuleList(
            [nn.ModuleList([nn.LayerNorm(width, eps=1e-6) for _ in range(4)]) for _ in range(depth // 4)])
        self.out = None if out_dim is None else _linear(width, out_dim, scale, mode)
        self.out_dim = width if out_dim is None else out_dim

    def forward(self, x):
        x = F.silu(self.first_ln(self.first(x)))
        for block, lns in zip(self.blocks, self.block_lns):
            identity = x
            for layer, ln in zip(block, lns):
                x = F.silu(ln(layer(x)))
            x = x + identity
        return x if self.out is None else self.out(x)


def _encoder(cfg: CRLConfig, n_in: int) -> nn.Module:
    if cfg.network == "mlp":
        # contrastive_rl networks.py: VarianceScaling(1.0, 'fan_avg', 'uniform'), linear output.
        return MLP(n_in, list(cfg.hidden_sizes) + [cfg.repr_dim], "fan_avg", activate_final=False)
    return ResidualMLP(n_in, cfg.width, cfg.depth, cfg.repr_dim)


class TanhGaussianActor(nn.Module):
    """pi(a | s, g) = tanh(N(mu, sigma)). Heads: Acme NormalTanhDistribution or c0 ScalingCRLActor."""

    def __init__(self, cfg: CRLConfig, n_in: int, action_dim: int):
        super().__init__()
        self.head = cfg.actor_head
        self.min_std = cfg.actor_min_std
        if cfg.network == "mlp":
            self.trunk = MLP(n_in, list(cfg.hidden_sizes), "fan_in", activate_final=True)
        else:
            self.trunk = ResidualMLP(n_in, cfg.width, cfg.depth, None)
        h = self.trunk.out_dim
        if self.head == "acme":
            # Acme NormalTanhDistribution: VarianceScaling(1.0, 'fan_in', 'uniform') on both heads.
            self.loc = _linear(h, action_dim, 1.0, "fan_in")
            self.scale = _linear(h, action_dim, 1.0, "fan_in")
        else:
            self.loc = _linear(h, action_dim, 1.0 / 3.0, "fan_in")
            self.scale = _linear(h, action_dim, 1.0 / 3.0, "fan_in")

    def dist_params(self, x):
        h = self.trunk(x)
        loc = self.loc(h)
        if self.head == "acme":
            std = F.softplus(self.scale(h)) + self.min_std
        else:
            log_std = -5.0 + 0.5 * 7.0 * (torch.tanh(self.scale(h)) + 1.0)
            std = torch.exp(log_std)
        return loc, std

    @staticmethod
    def tanh_log_prob(u, loc, std):
        """log pi(tanh(u)) with the stable log-det 2 (log 2 - u - softplus(-2u)) (distrax / Acme form)."""
        normal = -0.5 * ((u - loc) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)
        log_det = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        return (normal - log_det).sum(-1)

    def sample(self, x, noise=None):
        loc, std = self.dist_params(x)
        eps = torch.randn_like(loc) if noise is None else noise
        u = loc + std * eps
        return torch.tanh(u), self.tanh_log_prob(u, loc, std)

    def mode(self, x):
        loc, _ = self.dist_params(x)
        return torch.tanh(loc)


class CRLAgent(nn.Module):
    def __init__(self, cfg: CRLConfig, state_dim: int, action_dim: int, goal_dim: int, device="cpu",
                 capturable: bool = False):
        """``capturable``: CUDA-graph-ready optimizers (``build_graphed_update``); needs a CUDA ``device``."""
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.state_dim, self.action_dim, self.goal_dim = state_dim, action_dim, goal_dim
        self.sa_encoder = _encoder(cfg, state_dim + action_dim)
        self.g_encoder = _encoder(cfg, goal_dim)
        self.actor = TanhGaussianActor(cfg, state_dim + goal_dim, action_dim)
        self.log_alpha = nn.Parameter(torch.zeros(()))  # alpha = exp(log_alpha) = 1 at init (c0); unused if "off"
        self.target_entropy = -cfg.entropy_param * action_dim
        self.to(device)
        self.capturable = capturable
        # With capturable optimizers the learning rate is a tensor, so a graphed step can zero it (NaN skip).
        self._lr = torch.tensor(cfg.lr, device=device) if capturable else cfg.lr
        opt = dict(lr=self._lr, eps=cfg.adam_eps, capturable=capturable)
        critic_params = list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, **opt)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), **opt)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], **opt)
        self.skipped_updates = 0
        self._graph = None

    # --------------------------------------------------------------------- critic
    def phi(self, s, a):
        return self.sa_encoder(torch.cat([s, a], -1))

    def psi(self, g):
        return self.g_encoder(g)

    def logits(self, s, a, g):
        return self.phi(s, a) @ self.psi(g).T

    def critic_loss(self, s, a, g):
        logits = self.logits(s, a, g)
        b = logits.shape[0]
        eye = torch.eye(b, device=logits.device)
        if self.cfg.critic_loss == "nce":
            loss = F.binary_cross_entropy_with_logits(logits, eye)  # mean over B x B (official reduction)
            lse = torch.logsumexp(logits, dim=1)
        else:
            lse = torch.logsumexp(logits, dim=1)
            nce = lse - torch.diagonal(logits)
            penalty = torch.logsumexp(logits + 1e-6, dim=1) ** 2
            loss = nce.mean() + self.cfg.lse_coef * penalty.mean()
        with torch.no_grad():
            diag = torch.diagonal(logits)
            off = (logits.sum() - diag.sum()) / max(b * b - b, 1)
            info = dict(critic_loss=loss.detach(), logits_pos=diag.mean(), logits_neg=off,
                        accuracy=(logits.argmax(1) == torch.arange(b, device=logits.device)).float().mean(),
                        logsumexp=lse.mean())
        return loss, info

    # --------------------------------------------------------------------- actor
    def actor_goals(self, s, g):
        rg = self.cfg.random_goals
        if self.cfg.actor_goal_zero_velocity:
            g = torch.cat([g[:, :2], torch.zeros_like(g[:, 2:])], -1)
        if rg == 0.0:
            return s, g
        rolled = torch.roll(g, 1, dims=0)
        if rg == 1.0:
            return s, rolled
        return torch.cat([s, s], 0), torch.cat([g, rolled], 0)

    def actor_loss(self, s, g, noise=None):
        """Critic gradients are not taken here (the caller restricts autograd to the actor / alpha)."""
        s2, g2 = self.actor_goals(s, g)
        a2, logp = self.actor.sample(torch.cat([s2, g2], -1), noise)
        score = (self.phi(s2, a2) * self.psi(g2)).sum(-1)
        if self.cfg.entropy == "off":
            objective = -score
            alpha_loss = torch.zeros((), device=s.device)
            alpha = torch.zeros((), device=s.device)
        else:
            alpha = torch.exp(self.log_alpha)
            objective = alpha.detach() * logp - score
            entropy = -logp.detach().mean()
            alpha_loss = alpha * (entropy - self.target_entropy)  # c0: alpha value (not log) times the gap
        loss = objective.mean()
        info = dict(actor_loss=loss.detach(), score=score.detach().mean(), entropy=-logp.detach().mean(),
                    alpha=alpha.detach() if torch.is_tensor(alpha) else torch.tensor(alpha))
        return loss, alpha_loss, info

    # --------------------------------------------------------------------- update
    def update(self, s, a, g):
        """One step on a batch of (s_t, a_t, future goal). Critic and actor both use the pre-update critic."""
        critic_params = self.critic_opt.param_groups[0]["params"]
        actor_params = self.actor_opt.param_groups[0]["params"]
        c_loss, c_info = self.critic_loss(s, a, g)
        a_loss, alpha_loss, a_info = self.actor_loss(s, g)
        c_grads = torch.autograd.grad(c_loss, critic_params)
        a_grads = torch.autograd.grad(a_loss, actor_params)
        alpha_grad = torch.autograd.grad(alpha_loss, [self.log_alpha])[0] if self.cfg.entropy == "auto" else None
        if self.cfg.skip_nonfinite:
            finite = torch.isfinite(c_loss) & torch.isfinite(a_loss)
            for gr in c_grads + a_grads:
                finite = finite & torch.isfinite(gr).all()
            if not bool(finite):
                self.skipped_updates += 1
                return {**c_info, **a_info, "skipped": torch.tensor(1.0)}
        for p, gr in zip(critic_params, c_grads):
            p.grad = gr
        for p, gr in zip(actor_params, a_grads):
            p.grad = gr
        self.critic_opt.step()
        self.actor_opt.step()
        if alpha_grad is not None:
            self.log_alpha.grad = alpha_grad
            self.alpha_opt.step()
        return {**c_info, **a_info}

    def _graph_step(self, s, a, g):
        """The body of ``update`` without host syncs, for CUDA-graph capture."""
        critic_params = self.critic_opt.param_groups[0]["params"]
        actor_params = self.actor_opt.param_groups[0]["params"]
        c_loss, c_info = self.critic_loss(s, a, g)
        a_loss, alpha_loss, a_info = self.actor_loss(s, g)
        c_grads = torch.autograd.grad(c_loss, critic_params)
        a_grads = torch.autograd.grad(a_loss, actor_params)
        alpha_grad = torch.autograd.grad(alpha_loss, [self.log_alpha])[0] if self.cfg.entropy == "auto" else None
        if self.cfg.skip_nonfinite:
            finite = torch.isfinite(c_loss) & torch.isfinite(a_loss)
            for gr in c_grads + a_grads:
                finite = finite & torch.isfinite(gr).all()
            # A non-finite step leaves the parameters where they are (lr 0); the grads are zeroed so Adam's
            # moments stay finite. Unlike c0.py the moments still decay on that step.
            self._lr.copy_(torch.where(finite, torch.full_like(self._lr, self.cfg.lr), torch.zeros_like(self._lr)))
            c_grads = [torch.where(finite, gr, torch.zeros_like(gr)) for gr in c_grads]
            a_grads = [torch.where(finite, gr, torch.zeros_like(gr)) for gr in a_grads]
            if alpha_grad is not None:
                alpha_grad = torch.where(finite, alpha_grad, torch.zeros_like(alpha_grad))
            skipped = (~finite).float()
        else:
            skipped = torch.zeros((), device=s.device)
        for p, gr in zip(critic_params, c_grads):
            p.grad = gr
        for p, gr in zip(actor_params, a_grads):
            p.grad = gr
        self.critic_opt.step()
        self.actor_opt.step()
        if alpha_grad is not None:
            self.log_alpha.grad = alpha_grad
            self.alpha_opt.step()
        return {**c_info, **a_info, "skipped": skipped}

    def build_graphed_update(self, warmup_batches):
        """Capture one update as a CUDA graph. ``warmup_batches``: >= 3 real (s, a, g) batches, trained on.

        Returns ``update(s, a, g) -> info`` (the info tensors are overwritten by the next call).
        """
        assert self.capturable, "build the agent with capturable=True"
        s0, a0, g0 = warmup_batches[0]
        static = [torch.zeros_like(s0), torch.zeros_like(a0), torch.zeros_like(g0)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for batch in warmup_batches:
                for dst, src in zip(static, batch):
                    dst.copy_(src)
                self._graph_step(*static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self._graph_step(*static)
        self._graph = graph

        def update(s, a, g):
            for dst, src in zip(static, (s, a, g)):
                dst.copy_(src)
            graph.replay()
            return out

        return update

    # --------------------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, s, g, deterministic: bool):
        x = torch.cat([s, g], -1)
        if deterministic:
            return self.actor.mode(x)
        return self.actor.sample(x)[0]

    def checkpoint(self) -> dict:
        return dict(config=self.cfg.to_dict(), dims=dict(state_dim=self.state_dim, action_dim=self.action_dim,
                                                         goal_dim=self.goal_dim),
                    model=self.state_dict(), critic_opt=self.critic_opt.state_dict(),
                    actor_opt=self.actor_opt.state_dict(), alpha_opt=self.alpha_opt.state_dict())

    @staticmethod
    def from_checkpoint(ckpt: dict, device="cpu") -> "CRLAgent":
        cfg_d = dict(ckpt["config"])
        cfg_d["hidden_sizes"] = tuple(cfg_d["hidden_sizes"])
        agent = CRLAgent(CRLConfig(**cfg_d), **ckpt["dims"], device=device)
        agent.load_state_dict(ckpt["model"])
        return agent
