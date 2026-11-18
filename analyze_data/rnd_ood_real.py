"""RND novelty detection on real-robot juggling data: were the observations before a miss ID?

ID set   = every policy decision from the start of an episode up to and including the decision
           that produced the last successful hit (juggle).
Holdout  = every decision after the last successful hit until the episode terminates
           (the puck goes up, comes back, the robot misses and the puck hits the bottom).

Observation = the exact input of the deployed long-history policy (rma_mode history,
step_features latest_frame, history_len 50, use_last_action true):
    [ x_t (30) | window (50 x 8 = 400) | a_{t-1} (2) ]   -> dim(Oe) = 432
rebuilt from the split-schema episode HDF5s the way ``scripts/rma/evaluate.HistoryAgent`` builds it.

Alignment: each HDF5 row is written *after* env.step, so row i holds the state after
decision i and ``policy_action[i]`` = a_i. The observation used to choose a_i is therefore
built from rows <= i-1; for i = 0 (pre-episode obs, not logged) row 0 is used as a proxy.

Hits use the eval's own juggle counter (``scripts/td3/helper/juggle_counter.py``: paddle-puck
contact whose long-term puck-x direction flips, 30-frame cooldown). A hit at row t means the
contact state follows decision t, so the ID part is decisions 0..t_last.

RND (Burda et al.): frozen random target g: D -> 1024 -> 2048 -> 4096 -> m (LeakyReLU), predictor
f: D -> 1024 -> 2048 -> 4096 -> 2048 -> 1024 -> m (first three layers LeakyReLU like g, the two
extra layers ReLU), loss = E ||f(o) - g(o)||^2 over ID data, score s_RND = ||f(o) - g(o)||_2.

Modes
  full : train on all ID data (as specified), score the ID training set and the holdout.
  cv   : leave-episodes-out control. Training-set scores are biased low (the predictor has
         fitted those exact points), so "holdout > train" alone cannot separate OOD from merely
         unseen. Each fold trains on the ID data of the other episodes and scores the held-out
         episodes' ID part (unseen ID) and their post-last-hit part.

Usage
  CUDA_VISIBLE_DEVICES=2 python -m analyze_data.rnd_ood_real --mode full
  CUDA_VISIBLE_DEVICES=2 python -m analyze_data.rnd_ood_real --mode cv --folds 5
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from scripts.td3.helper.juggle_counter import (
    CONTACT_THRESH,
    COOLDOWN_FRAMES,
    _classify_event,
    _find_contact_event_centers,
)

HISTORY_LEN = 50
N_FRAMES = 5
ACT_DIM = 2
OBS_DIM = 30
FEAT_DIM = 6 + ACT_DIM
INPUT_DIM = OBS_DIM + HISTORY_LEN * FEAT_DIM + ACT_DIM  # 432


# ----------------------------------------------------------------------------- data


@dataclass
class Episode:
    name: str
    end_reason: str
    n_steps: int
    hits: list[int]
    obs: np.ndarray  # (T, INPUT_DIM)
    t_last: int  # last ID decision index (-1 if no hit)
    puck_y: np.ndarray = field(repr=False, default=None)

    @property
    def id_idx(self) -> np.ndarray:
        return np.arange(0, self.t_last + 1)

    @property
    def tail_idx(self) -> np.ndarray:
        return np.arange(self.t_last + 1, self.n_steps)

    @property
    def tail_is_miss(self) -> bool:
        return self.end_reason == "puck_hit_bottom"


def juggle_hit_rows(paddle_xy: np.ndarray, puck: np.ndarray) -> list[int]:
    """Row indices of accepted juggles, identical to juggle_counter.count_juggles_from_arrays."""
    dist = np.linalg.norm(paddle_xy - puck[:, :2], axis=1)
    dist[puck[:, 2] > 0.5] = np.inf
    T = len(dist)
    hits, last = [], -(10**9)
    for t in _find_contact_event_centers(dist):
        if _classify_event(t, puck[:, 0], T) != "juggle" or (t - last) < COOLDOWN_FRAMES:
            continue
        hits.append(t)
        last = t
    return hits


def build_policy_inputs(paddle_xy: np.ndarray, puck: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """(T, 432) policy inputs, one per decision, mirroring HistoryAgent."""
    T = len(actions)
    frames = np.concatenate([paddle_xy, np.zeros((T, 1)), puck[:, :2], puck[:, 2:3]], axis=1)  # (T, 6) per row
    # state seen at decision i = row i-1 (row 0 as proxy for the unlogged initial obs)
    S = np.concatenate([frames[:1], frames[:-1]], axis=0).astype(np.float32)
    A = actions.astype(np.float32)
    # 5-frame history obs: frames S[i-4..i], repeat-first padding
    Spad = np.concatenate([np.repeat(S[:1], N_FRAMES - 1, axis=0), S], axis=0)
    hist = np.stack([Spad[k : k + T] for k in range(N_FRAMES)], axis=1)  # (T, 5, 6) oldest -> newest
    x = np.concatenate([hist[:, :, 0:3].reshape(T, -1), hist[:, :, 3:6].reshape(T, -1)], axis=1)  # (T, 30)
    # window at decision i = pairs (S_j, a_j) for j < i, prefix (S_0, 0)
    pairs = np.concatenate([S, A], axis=1)
    prefix = np.concatenate([np.repeat(S[:1], HISTORY_LEN, axis=0), np.zeros((HISTORY_LEN, ACT_DIM), np.float32)], axis=1)
    P = np.concatenate([prefix, pairs], axis=0)
    window = np.stack([P[i : i + HISTORY_LEN] for i in range(T)], axis=0).reshape(T, -1)
    prev_a = np.concatenate([np.zeros((1, ACT_DIM), np.float32), A[:-1]], axis=0)
    return np.concatenate([x, window, prev_a], axis=1).astype(np.float32)


def load_episodes(data_dir: Path) -> list[Episode]:
    episodes = []
    for meta_path in sorted(data_dir.glob("s*/data_*/eval_per_episode.jsonl")):
        run_dir = meta_path.parent
        seed = run_dir.parent.name
        for line in meta_path.read_text().splitlines():
            m = json.loads(line)
            rel = m["artifact_path"].split("episode_hdf5/")[1]
            with h5py.File(run_dir / "episode_hdf5" / rel, "r") as h:
                paddle_xy = h["pose"][:, :2].astype(np.float64)
                puck = h["puck"][:, :3].astype(np.float64)
                actions = h["policy_action"][:, :2].astype(np.float64)
            hits = juggle_hit_rows(paddle_xy, puck)
            if len(hits) != m["episode_juggles"]:
                print(f"[warn] {seed} ep{m['episode_id']}: recount {len(hits)} juggles vs logged {m['episode_juggles']}")
            episodes.append(
                Episode(
                    name=f"{seed}_ep{m['episode_id']}",
                    end_reason=m["episode_end_reason"],
                    n_steps=len(actions),
                    hits=hits,
                    obs=build_policy_inputs(paddle_xy, puck, actions),
                    t_last=hits[-1] if hits else -1,
                    puck_y=puck[:, 1],
                )
            )
    return episodes


# ----------------------------------------------------------------------------- RND


def target_net(d: int, m: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d, 1024), nn.LeakyReLU(),
        nn.Linear(1024, 2048), nn.LeakyReLU(),
        nn.Linear(2048, 4096), nn.LeakyReLU(),
        nn.Linear(4096, m),
    )


def predictor_net(d: int, m: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d, 1024), nn.LeakyReLU(),
        nn.Linear(1024, 2048), nn.LeakyReLU(),
        nn.Linear(2048, 4096), nn.LeakyReLU(),
        nn.Linear(4096, 2048), nn.ReLU(),
        nn.Linear(2048, 1024), nn.ReLU(),
        nn.Linear(1024, m),
    )


class RND(nn.Module):
    def __init__(self, d: int, m: int, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.g = target_net(d, m)
        for p in self.g.parameters():
            p.requires_grad_(False)
        self.f = predictor_net(d, m)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """s_RND(o) = ||f(o) - g(o)||_2 per sample."""
        z = (o - self.mean) / self.std
        return torch.linalg.vector_norm(self.f(z) - self.g(z), dim=-1)


@torch.no_grad()
def score(model: RND, X: np.ndarray, device, bs: int = 4096) -> np.ndarray:
    model.eval()
    out = [model(torch.as_tensor(X[i : i + bs], device=device)).cpu().numpy() for i in range(0, len(X), bs)]
    model.train()
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def train_rnd(X_id: np.ndarray, args, device, writer: SummaryWriter, monitor: dict[str, np.ndarray], tag: str) -> RND:
    torch.manual_seed(args.seed)
    mean = X_id.mean(0)
    std = X_id.std(0)
    std[std < 1e-6] = 1.0  # constant dims (paddle flag) pass through
    model = RND(X_id.shape[1], args.m, mean, std).to(device)
    opt = torch.optim.Adam(model.f.parameters(), lr=args.lr)
    X = torch.as_tensor(X_id, device=device)
    n = len(X)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device, generator=gen)
        s = model(X[idx])
        loss = (s**2).mean()  # L(theta) = E ||f - g||^2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == 1:
            writer.add_scalar(f"{tag}/loss", loss.item(), step)
            writer.add_scalar(f"{tag}/sRND_train_batch", s.mean().item(), step)
        if step % args.eval_every == 0 or step == args.steps:
            for name, Xm in monitor.items():
                if len(Xm):
                    writer.add_scalar(f"{tag}/sRND_mean_{name}", float(score(model, Xm, device).mean()), step)
            print(f"[{tag}] step {step}/{args.steps} loss {loss.item():.5f} ({time.time() - t0:.0f}s)", flush=True)
    return model


# ----------------------------------------------------------------------------- reporting


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "sem": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
    }


def episode_mean_stats(per_ep: list[np.ndarray]) -> dict:
    """Mean ± s.e. over episodes of each episode's mean score (episodes are the independent unit)."""
    means = np.array([v.mean() for v in per_ep if len(v)])
    return stats(means)


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(score_pos > score_neg) (Mann-Whitney), ties count 1/2."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    # average ranks for ties
    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[len(neg) :].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tail_profile(tail_scores: list[np.ndarray], ref_thresh: float, max_back: int = 60) -> list[dict]:
    """Mean score and fraction above the ID threshold vs steps-before-termination (aligned at the end)."""
    rows = []
    for k in range(1, max_back + 1):
        vals = np.array([s[-k] for s in tail_scores if len(s) >= k])
        if len(vals) == 0:
            break
        rows.append({"steps_before_end": k, "n_eps": int(len(vals)), "mean": float(vals.mean()),
                     "sem": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
                     "frac_above_id_p99": float((vals > ref_thresh).mean())})
    return rows


def summarize(ref_id: np.ndarray, id_by_ep: dict, tail_by_ep: dict, label_ref: str) -> dict:
    """Compare holdout (post-last-hit) scores with a reference ID score distribution."""
    tail_all = np.concatenate([v for v in tail_by_ep.values()]) if tail_by_ep else np.zeros(0)
    p95, p99 = float(np.percentile(ref_id, 95)), float(np.percentile(ref_id, 99))
    # last inter-hit interval of the ID part (second-to-last hit -> last hit), phase-matched to the tail
    return {
        "reference": label_ref,
        "ref_id_steps": stats(ref_id),
        "ref_id_episode_means": episode_mean_stats(list(id_by_ep.values())),
        "holdout_steps": stats(tail_all),
        "holdout_episode_means": episode_mean_stats(list(tail_by_ep.values())),
        "ratio_holdout_over_ref_mean": float(tail_all.mean() / ref_id.mean()) if len(tail_all) else float("nan"),
        "auroc_holdout_vs_ref": auroc(ref_id, tail_all),
        "ref_p95": p95,
        "ref_p99": p99,
        "holdout_frac_above_ref_p95": float((tail_all > p95).mean()) if len(tail_all) else float("nan"),
        "holdout_frac_above_ref_p99": float((tail_all > p99).mean()) if len(tail_all) else float("nan"),
        "tail_profile_vs_end": tail_profile(list(tail_by_ep.values()), p99),
    }


def last_interval_scores(ep: Episode, s_id: np.ndarray) -> np.ndarray:
    """Scores of the ID decisions between the second-to-last and last hit (phase-matched control)."""
    if len(ep.hits) < 2:
        return np.zeros(0)
    return s_id[ep.hits[-2] + 1 : ep.hits[-1] + 1]


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/eval_new_sim_T10000"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--mode", choices=["full", "cv"], default="full")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--m", type=int, default=512, help="RND output dim")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out_dir or Path("runs/rnd_ood") / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out / "tb"))

    episodes = load_episodes(args.data_dir)
    # Holdout = tails that end in a miss (puck_hit_bottom) after at least one hit.
    # Tails ending otherwise (e.g. detector readiness failure) and no-hit episodes are reported separately.
    miss_eps = [e for e in episodes if e.t_last >= 0 and e.tail_is_miss]
    other_tail_eps = [e for e in episodes if e.t_last >= 0 and not e.tail_is_miss]
    nohit_eps = [e for e in episodes if e.t_last < 0]
    print(f"{len(episodes)} episodes: {len(miss_eps)} miss tails, {len(other_tail_eps)} non-miss tails "
          f"({[e.name + ':' + e.end_reason for e in other_tail_eps]}), {len(nohit_eps)} with no hit ({[e.name for e in nohit_eps]})")
    data_info = {
        e.name: {"n_steps": e.n_steps, "n_hits": len(e.hits), "t_last": e.t_last, "n_id": len(e.id_idx),
                 "n_tail": len(e.tail_idx), "end_reason": e.end_reason}
        for e in episodes
    }

    result = {"args": {k: str(v) for k, v in vars(args).items()}, "input_dim": INPUT_DIM, "episodes": data_info}

    if args.mode == "full":
        id_eps = [e for e in episodes if e.t_last >= 0]
        X_id = np.concatenate([e.obs[e.id_idx] for e in id_eps])
        X_tail = np.concatenate([e.obs[e.tail_idx] for e in miss_eps])
        print(f"ID train {X_id.shape}, holdout {X_tail.shape}")
        model = train_rnd(X_id, args, device, writer, {"id_train": X_id, "holdout": X_tail}, tag="full")
        torch.save(model.state_dict(), out / "rnd.pt")
        id_by_ep = {e.name: score(model, e.obs[e.id_idx], device) for e in id_eps}
        tail_by_ep = {e.name: score(model, e.obs[e.tail_idx], device) for e in miss_eps}
        ref = np.concatenate(list(id_by_ep.values()))
        result["full"] = summarize(ref, id_by_ep, tail_by_ep, "ID training set (seen by the predictor)")
        result["full"]["other_tails"] = {e.name: stats(score(model, e.obs[e.tail_idx], device)) for e in other_tail_eps}
        result["full"]["no_hit_episodes"] = {e.name: stats(score(model, e.obs, device)) for e in nohit_eps}
        result["full"]["per_episode"] = {
            n: {"id_mean": float(id_by_ep[n].mean()), "tail_mean": float(tail_by_ep[n].mean()),
                "last_interval_mean": float(last_interval_scores(next(e for e in miss_eps if e.name == n), id_by_ep[n]).mean())
                if len(next(e for e in miss_eps if e.name == n).hits) >= 2 else float("nan")}
            for n in tail_by_ep
        }
    else:
        rng = np.random.default_rng(args.seed)
        id_eps = [e for e in episodes if e.t_last >= 0]
        order = rng.permutation(len(id_eps))
        folds = [order[k :: args.folds] for k in range(args.folds)]
        id_by_ep, tail_by_ep, lastint_by_ep, train_ref = {}, {}, {}, []
        for k, test in enumerate(folds):
            test_set = set(test.tolist())
            train_eps = [e for i, e in enumerate(id_eps) if i not in test_set]
            test_eps = [id_eps[i] for i in test]
            X_id = np.concatenate([e.obs[e.id_idx] for e in train_eps])
            mon = {
                "id_train": X_id,
                "id_unseen": np.concatenate([e.obs[e.id_idx] for e in test_eps]),
                "holdout_unseen": np.concatenate([e.obs[e.tail_idx] for e in test_eps if e.tail_is_miss] or [np.zeros((0, INPUT_DIM), np.float32)]),
            }
            print(f"[fold {k}] train eps {len(train_eps)} ({len(X_id)} steps), test eps {[e.name for e in test_eps]}")
            model = train_rnd(X_id, args, device, writer, mon, tag=f"fold{k}")
            train_ref.append(score(model, X_id, device))
            for e in test_eps:
                s_id = score(model, e.obs[e.id_idx], device)
                id_by_ep[e.name] = s_id
                lastint_by_ep[e.name] = last_interval_scores(e, s_id)
                if e.tail_is_miss:
                    tail_by_ep[e.name] = score(model, e.obs[e.tail_idx], device)
            del model
            torch.cuda.empty_cache()
        ref = np.concatenate(list(id_by_ep.values()))
        result["cv"] = summarize(ref, id_by_ep, tail_by_ep, "unseen ID (held-out episodes, pre-last-hit)")
        li = np.concatenate([v for v in lastint_by_ep.values() if len(v)])
        tail_all = np.concatenate(list(tail_by_ep.values()))
        result["cv"]["last_interval_unseen_id_steps"] = stats(li)
        result["cv"]["auroc_holdout_vs_last_interval"] = auroc(li, tail_all)
        result["cv"]["train_fit_steps"] = stats(np.concatenate(train_ref))
        # paired per-episode comparison: tail mean minus own unseen-ID mean
        diffs = np.array([tail_by_ep[n].mean() - id_by_ep[n].mean() for n in tail_by_ep])
        result["cv"]["paired_tail_minus_id_episode"] = stats(diffs)
        result["cv"]["per_episode"] = {
            n: {"id_mean": float(id_by_ep[n].mean()), "tail_mean": float(tail_by_ep[n].mean()),
                "last_interval_mean": float(lastint_by_ep[n].mean()) if len(lastint_by_ep[n]) else float("nan")}
            for n in tail_by_ep
        }

    (out / "results.json").write_text(json.dumps(result, indent=2))
    writer.close()
    r = result[args.mode]
    print(json.dumps({k: v for k, v in r.items() if k not in ("tail_profile_vs_end", "per_episode")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
