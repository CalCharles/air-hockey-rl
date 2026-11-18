"""Y3H4 in PyTorch: z-conditioned contrastive RL with an ensemble of independent heads.

A port of the colleague's ``y3h4.py`` (R2CRL phase-1 arm ``y3h4-converge``: critic_code=commanded,
critic_code_grad=true, actor_condition=embedding, ensemble_heads=4). Everything below follows that file;
``parity_y3h4.py`` checks this port against it numerically. Per head:

- psi(g): goal encoder (residual LayerNorm/swish MLP -> Euclidean projection head), repr_dim numbers.
- phi(s, a, z): state-action encoder, also given z = psi(commanded goal) (the goal the agent was told to reach
  when it acted). Gradient flows into psi through z.
- Critic: InfoNCE over the batch with L2 energy, logits[i, j] = -||phi_i - psi(g_j)||, positives = each row's
  hindsight future goal plus Gaussian jitter; + logsumexp_penalty * logsumexp(row)^2.
- Actor pi(a | s, z): tanh-Gaussian (log-std squashed into [-5, 2]) on the goal *embedding* z = psi(g) of the
  row's own hindsight goal (no gradient into psi). Objective: alpha * log pi - (-||phi(s, a, z) - z||) with the
  pre-update critic; alpha auto-tuned towards target entropy -entropy_param * action_dim.
- Heads are independent (own init, own Adam, own noise) and trained on the same batches. Acting: every head
  proposes its mean action, every head scores every proposal, the proposal with the best mean score wins;
  during collection only ``commit_fraction`` of the lanes follow that vote, the others are pinned to one head.

Implementation: the heads are stored stacked (weights [H, in, out]), so one batched matmul per layer runs all
heads. One Adam over all parameters is the same as y3h4's per-head Adam (Adam is elementwise, no clipping).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
LN_EPS = 1e-6  # flax LayerNorm default


@dataclass
class Y3H4Config:
    heads: int = 4
    width: int = 256
    depth: int = 8  # residual layers after the first one; multiple of 4
    repr_dim: int = 64
    lr: float = 3e-4
    adam_eps: float = 1e-8  # optax default
    batch_size: int = 256
    logsumexp_penalty: float = 0.1
    goal_jitter: float = 0.02  # goal units (here m and m/s)
    entropy_param: float = 0.5  # target entropy = -entropy_param * action_dim
    commit_fraction: float = 0.25  # share of collection lanes that act on the heads' vote
    diagnostic_interval: int = 16  # rolled-code accuracy every this many updates
    skip_nonfinite: bool = True

    def validate(self):
        assert self.heads >= 1 and self.repr_dim >= 2
        assert self.depth >= 4 and self.depth % 4 == 0, self.depth
        assert 0.0 <= self.commit_fraction <= 1.0

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------- stacked layers
class EnsLinear(nn.Module):
    """H independent Linear layers. Weight [H, in, out] (flax kernel layout). Input [B, in] (shared by all heads)
    or [H, B, in]; output [H, B, out]."""

    def __init__(self, heads: int, n_in: int, n_out: int, init: str = "lecun_uniform"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(heads, n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(heads, n_out))
        with torch.no_grad():
            if init == "lecun_uniform":  # variance_scaling(1/3, 'fan_in', 'uniform'): U(+-sqrt(1/fan_in))
                bound = math.sqrt(1.0 / n_in)
                self.weight.uniform_(-bound, bound)
            elif init == "lecun_normal":  # flax Dense default: variance_scaling(1, 'fan_in', 'truncated_normal')
                std = math.sqrt(1.0 / n_in) / 0.87962566103423978
                nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)
            else:
                raise ValueError(init)

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias.unsqueeze(1)


class EnsLayerNorm(nn.Module):
    def __init__(self, heads: int, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, width))
        self.bias = nn.Parameter(torch.zeros(heads, width))
        self.width = width

    def forward(self, x):  # [H, B, width]
        return F.layer_norm(x, (self.width,), eps=LN_EPS) * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)


class EnsResidualMLP(nn.Module):
    """y3h4 ResidualMLP: Dense -> LayerNorm -> swish, then depth // 4 residual blocks of four such layers."""

    def __init__(self, heads, n_in, width, depth, out_dim):
        super().__init__()
        n_layers = 1 + depth
        self.dense = nn.ModuleList([EnsLinear(heads, n_in if i == 0 else width, width) for i in range(n_layers)])
        self.norms = nn.ModuleList([EnsLayerNorm(heads, width) for _ in range(n_layers)])
        self.out = None if out_dim is None else EnsLinear(heads, width, out_dim)
        self.depth = depth

    def _layer(self, i, h):
        return F.silu(self.norms[i](self.dense[i](h)))

    def forward(self, x):
        x = self._layer(0, x)
        i = 1
        for _ in range(self.depth // 4):
            identity = x
            for _ in range(4):
                x = self._layer(i, x)
                i += 1
            x = x + identity
        return x if self.out is None else self.out(x)


class EnsEncoder(nn.Module):
    """ResidualMLP -> EuclideanHead: affine map of [1, h] to repr_dim - 1 coordinates, a zero coordinate prepended."""

    def __init__(self, heads, n_in, width, depth, repr_dim):
        super().__init__()
        self.mlp = EnsResidualMLP(heads, n_in, width, depth, repr_dim)
        self.htc = EnsLinear(heads, repr_dim + 1, repr_dim - 1, init="lecun_normal")

    def forward(self, x):
        t = self.mlp(x)
        spatial = self.htc(torch.cat([torch.ones_like(t[..., :1]), t], -1))
        return torch.cat([torch.zeros_like(spatial[..., :1]), spatial], -1)


class EnsActor(nn.Module):
    def __init__(self, heads, n_in, width, depth, action_dim):
        super().__init__()
        self.trunk = EnsResidualMLP(heads, n_in, width, depth, None)
        self.mean = EnsLinear(heads, width, action_dim)
        self.log_std = EnsLinear(heads, width, action_dim)

    def forward(self, x):
        h = self.trunk(x)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(self.log_std(h)) + 1.0)
        return self.mean(h), torch.exp(log_std)


def l2_logits(x, y):
    """All pairs per head: [H, B, R] x [H, B, R] -> logits[h, i, j] = -||x_hi - y_hj||."""
    return -torch.sqrt(torch.sum(torch.square(x.unsqueeze(2) - y.unsqueeze(1)), -1) + 1e-12)


def l2_pair(x, y):
    return -torch.sqrt(torch.sum(torch.square(x - y), -1) + 1e-12)


def tanh_log_prob(u, loc, std):
    """log pi(tanh(u)): diagonal Gaussian log-density minus the distrax Tanh log-det 2 (log 2 - u - softplus(-2u))."""
    normal = -0.5 * ((u - loc) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)
    log_det = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
    return (normal - log_det).sum(-1)


# ----------------------------------------------------------------------------- agent
class Y3H4Agent(nn.Module):
    def __init__(self, cfg: Y3H4Config, state_dim: int, action_dim: int, goal_dim: int, device="cpu",
                 capturable: bool = False):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.state_dim, self.action_dim, self.goal_dim = state_dim, action_dim, goal_dim
        H, W, D, R = cfg.heads, cfg.width, cfg.depth, cfg.repr_dim
        self.sa_encoder = EnsEncoder(H, state_dim + action_dim + R, W, D, R)  # phi(s, [a, z])
        self.g_encoder = EnsEncoder(H, goal_dim, W, D, R)  # psi(g)
        self.actor = EnsActor(H, state_dim + R, W, D, action_dim)  # pi(a | s, z)
        self.log_alpha = nn.Parameter(torch.zeros(H))  # alpha = exp(log_alpha) = 1 at init
        self.target_entropy = -cfg.entropy_param * action_dim
        self.to(device)
        self.capturable = capturable
        self._lr = torch.tensor(cfg.lr, device=device) if capturable else cfg.lr
        self.opt = torch.optim.Adam(self.parameters(), lr=self._lr, eps=cfg.adam_eps, capturable=capturable)
        self.critic_params = list(self.sa_encoder.parameters()) + list(self.g_encoder.parameters())
        self.actor_params = list(self.actor.parameters())
        self.skipped_updates = 0
        self._graph = None

    # --------------------------------------------------------------------- encoders
    def phi(self, s, a, z):
        """s [B, S] or [H, B, S]; a [B, A] or [H, B, A]; z [H, B, R] -> [H, B, R]."""
        H = self.cfg.heads
        if s.dim() == 2:
            s = s.expand(H, *s.shape)
        if a.dim() == 2:
            a = a.expand(H, *a.shape)
        return self.sa_encoder(torch.cat([s, a, z], -1))

    def psi(self, g):
        return self.g_encoder(g)

    def policy(self, s, z):
        if s.dim() == 2:
            s = s.expand(self.cfg.heads, *s.shape)
        return self.actor(torch.cat([s, z], -1))

    # --------------------------------------------------------------------- losses
    def critic_loss(self, s, a, value_goals, commanded, jitter_noise=None):
        cfg = self.cfg
        if jitter_noise is None:
            jitter_noise = torch.randn((cfg.heads,) + tuple(value_goals.shape), device=value_goals.device)
        goals = value_goals + cfg.goal_jitter * jitter_noise  # [H, B, G], own noise per head
        z = self.psi(commanded)  # [H, B, R]; the code carries gradient into psi
        sa = self.phi(s, a, z)
        g = self.psi(goals)
        logits = l2_logits(sa, g)  # [H, B, B]
        lse = torch.logsumexp(logits, -1)
        diag = torch.diagonal(logits, dim1=1, dim2=2)
        per_head = -(diag - lse).mean(-1) + cfg.logsumexp_penalty * torch.square(
            torch.logsumexp(logits + 1e-6, -1)).mean(-1)
        with torch.no_grad():
            idx = torch.arange(logits.shape[1], device=logits.device)
            acc = (logits.argmax(-1) == idx).float().mean()
            info = dict(critic_loss=per_head.detach().mean(), accuracy=acc, logsumexp=lse.mean(),
                        logits_pos=diag.mean(), logits_neg=(logits.sum((1, 2)) - diag.sum(-1)).mean() / max(
                            logits.shape[1] ** 2 - logits.shape[1], 1))
        return per_head.sum(), info, z, g

    def actor_loss(self, s, actor_goals, actor_noise=None):
        """Critic gradients are excluded by the caller (autograd restricted to actor / alpha parameters)."""
        code = self.psi(actor_goals).detach()  # [H, B, R]
        loc, std = self.policy(s, code)
        eps = torch.randn_like(loc) if actor_noise is None else actor_noise
        u = loc + std * eps
        a = torch.tanh(u)
        logp = tanh_log_prob(u, loc, std)  # [H, B]
        score = l2_pair(self.phi(s, a, code), code)
        alpha = torch.exp(self.log_alpha)  # [H]
        objective = alpha.detach().unsqueeze(1) * logp - score
        entropy = -logp.detach().mean(-1)  # [H]
        per_head = objective.mean(-1) + alpha * (entropy - self.target_entropy)
        info = dict(actor_loss=objective.detach().mean(), entropy=entropy.mean(), alpha=alpha.detach().mean(),
                    score=score.detach().mean())
        return per_head.sum(), info

    @torch.no_grad()
    def rolled_code_accuracy(self, s, a, value_goals, commanded, jitter_noise=None):
        """y3h4 diagnostic: shift every row's z by one row; does phi still pick out the positive? Near the plain
        accuracy = the critic ignores the commanded goal; much lower = it relies on it. Uses jittered goals,
        as y3h4 does (with fresh noise here)."""
        if jitter_noise is None:
            jitter_noise = torch.randn((self.cfg.heads,) + tuple(value_goals.shape), device=value_goals.device)
        z = self.psi(commanded)
        g = self.psi(value_goals + self.cfg.goal_jitter * jitter_noise)
        logits = l2_logits(self.phi(s, a, torch.roll(z, 1, dims=1)), g)
        idx = torch.arange(logits.shape[1], device=logits.device)
        return (logits.argmax(-1) == idx).float().mean()

    def _losses(self, s, a, value_goals, actor_goals, commanded, jitter_noise=None, actor_noise=None):
        c_loss, c_info, _, _ = self.critic_loss(s, a, value_goals, commanded, jitter_noise)
        a_loss, a_info = self.actor_loss(s, actor_goals, actor_noise)
        c_grads = torch.autograd.grad(c_loss, self.critic_params)
        a_grads = torch.autograd.grad(a_loss, self.actor_params + [self.log_alpha])
        return c_loss, a_loss, c_grads, a_grads, {**c_info, **a_info}

    # --------------------------------------------------------------------- update
    def update(self, s, a, value_goals, actor_goals, commanded):
        c_loss, a_loss, c_grads, a_grads, info = self._losses(s, a, value_goals, actor_goals, commanded)
        if self.cfg.skip_nonfinite:
            finite = torch.isfinite(c_loss) & torch.isfinite(a_loss)
            for gr in c_grads + a_grads:
                finite = finite & torch.isfinite(gr).all()
            if not bool(finite):
                self.skipped_updates += 1
                return {**info, "skipped": torch.tensor(1.0)}
        for p, gr in zip(self.critic_params + self.actor_params + [self.log_alpha], c_grads + a_grads):
            p.grad = gr
        self.opt.step()
        return {**info, "skipped": torch.tensor(0.0)}

    def _graph_step(self, s, a, value_goals, actor_goals, commanded):
        c_loss, a_loss, c_grads, a_grads, info = self._losses(s, a, value_goals, actor_goals, commanded)
        grads = list(c_grads + a_grads)
        if self.cfg.skip_nonfinite:
            finite = torch.isfinite(c_loss) & torch.isfinite(a_loss)
            for gr in grads:
                finite = finite & torch.isfinite(gr).all()
            # Non-finite step: lr 0 and zeroed grads leave every head's parameters unchanged (y3h4 keeps the old
            # heads). Unlike y3h4 the Adam moments still decay on that step.
            self._lr.copy_(torch.where(finite, torch.full_like(self._lr, self.cfg.lr), torch.zeros_like(self._lr)))
            grads = [torch.where(finite, gr, torch.zeros_like(gr)) for gr in grads]
            skipped = (~finite).float()
        else:
            skipped = torch.zeros((), device=s.device)
        for p, gr in zip(self.critic_params + self.actor_params + [self.log_alpha], grads):
            p.grad = gr
        self.opt.step()
        return {**info, "skipped": skipped}

    def build_graphed_update(self, warmup_batches):
        """Capture one update as a CUDA graph; ``warmup_batches``: >= 3 real batches (trained on)."""
        assert self.capturable, "build the agent with capturable=True"
        static = [torch.zeros_like(x) for x in warmup_batches[0]]
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

        def update(*batch):
            for dst, src in zip(static, batch):
                dst.copy_(src)
            graph.replay()
            return out

        return update

    # --------------------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, s, goals, temperature: float = 0.0, lane_ids=None, n_lanes=None, noise=None):
        """y3h4 ``act``: each head proposes its mean action, all heads score all proposals (z = psi(goal)), the
        best mean score wins. temperature 0 = deterministic means; otherwise each head samples with its std
        scaled by temperature. With ``lane_ids`` (collection) only the first commit_fraction * n_lanes lanes
        follow the vote; lane l >= commit is pinned to head (l - commit) % H."""
        H, N = self.cfg.heads, s.shape[0]
        z = self.psi(goals)  # [H, N, R]
        loc, std = self.policy(s, z)
        means = torch.clamp(torch.tanh(loc), -1.0, 1.0)  # [H, N, A]
        # scores[scorer h, proposer p, lane n] = -||phi_h(s_n, means[p, n], z_h) - psi_h(g_n)||
        props = means.reshape(1, H * N, -1).expand(H, H * N, means.shape[-1])
        s_rep = s.repeat(H, 1)
        z_rep = z.repeat(1, H, 1)
        sa = self.phi(s_rep, props, z_rep).reshape(H, H, N, -1)
        scores = l2_pair(sa, z.unsqueeze(1))  # [H, H, N]
        selected = scores.mean(0).argmax(0)  # [N]
        if temperature > 0:
            eps = torch.randn_like(loc) if noise is None else noise
            actions = torch.clamp(torch.tanh(loc + std * temperature * eps), -1.0, 1.0)
        else:
            actions = means
        if lane_ids is not None:
            commit = int(self.cfg.commit_fraction * (n_lanes if n_lanes is not None else N))
            selected = torch.where(lane_ids < commit, selected, (lane_ids - commit) % H)
        return actions[selected, torch.arange(N, device=s.device)]

    # --------------------------------------------------------------------- checkpoints
    def checkpoint(self) -> dict:
        return dict(algo="y3h4", config=self.cfg.to_dict(),
                    dims=dict(state_dim=self.state_dim, action_dim=self.action_dim, goal_dim=self.goal_dim),
                    model=self.state_dict(), opt=self.opt.state_dict())

    @staticmethod
    def from_checkpoint(ckpt: dict, device="cpu") -> "Y3H4Agent":
        agent = Y3H4Agent(Y3H4Config(**ckpt["config"]), **ckpt["dims"], device=device)
        agent.load_state_dict(ckpt["model"])
        return agent
