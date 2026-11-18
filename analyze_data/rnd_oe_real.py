"""RND-OE on real-robot juggling: is a held-out episode's ID data scored as ID?

Same data and ID / holdout split as ``rnd_ood_real.py`` (ID = decisions up to the last successful
hit, tail = after it until termination), but RND runs on the policy's observation *embedding*
instead of the raw input, as in RND-OE:

    Oe_t = h(O_t) = output of the actor's ResidualMLPTrunk (64 dims), i.e. the last hidden layer
                    before ``actor_mean_head`` produces the action. h is frozen.

Each episode is embedded by the policy that collected it (one encoder per seed folder), so the
RND model is trained per policy, leave-one-episode-out:
    for every episode e of policy p: train on the ID part of p's other episodes, then score e's
    ID part (unseen ID) and e's tail.

Policy inputs are rebuilt recurrently the way the deployed HistoryAgent built them: the window holds
(x_j, a_j^policy) with the action the policy itself returned (also during the 8-step post-reset
hold, when the robot executed zeros), while the last-action input is the executed (logged) action.
The match rate of the recomputed vs the logged actions is reported as a reconstruction check.

Decision rule (deployable, uses only the training episodes): an observation is predicted OOD when
its score exceeds the 95th (or 99th) percentile of the training episodes' scores on their held-back
validation chunk (the last 20 % of each training episode's ID part, also used for early stopping).
A well-calibrated detector predicts ~5 % (1 %) of unseen ID as OOD.

Positive controls (should be flagged as OOD): unseen-ID inputs with the 50-step window shuffled in
time, and unseen-ID inputs with the puck history swapped for the puck history of a random other
time step of the same episode (inconsistent paddle / puck / window combination).

Usage
  CUDA_VISIBLE_DEVICES=2 python -m analyze_data.rnd_oe_real --seeds s1 s11 --out-dir runs/rnd_oe/v1
  CUDA_VISIBLE_DEVICES=3 python -m analyze_data.rnd_oe_real --seeds s21 s31 s41 --out-dir runs/rnd_oe/v1
  python -m analyze_data.rnd_oe_real --summarize --out-dir runs/rnd_oe/v1
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from analyze_data.rnd_ood_real import (
    build_policy_inputs,
    ACT_DIM,
    FEAT_DIM,
    HISTORY_LEN,
    N_FRAMES,
    OBS_DIM,
    RND,
    auroc,
    juggle_hit_rows,
    score,
    stats,
)
from scripts.rma.bundle import load_phase1

POLICIES = {
    "s1": "night1__ft_term03_nopretanh_s1__swa1200k",
    "s11": "night7__ft_term03_nopretanh_s11__swa1200k",
    "s21": "night8__ft_term03_nopretanh_s21__swa1200k",
    "s31": "night8__ft_term03_nopretanh_s31__swa1200k",
    "s41": "night8__ft_term03_nopretanh_s41__swa1200k",
}


# ----------------------------------------------------------------------------- policy inputs / embeddings


def frames_from_rows(paddle_xy: np.ndarray, puck: np.ndarray) -> np.ndarray:
    """State seen at decision i (row i-1; row 0 stands in for the unlogged initial obs), (T, 6)."""
    T = len(paddle_xy)
    f = np.concatenate([paddle_xy, np.zeros((T, 1)), puck[:, :2], puck[:, 2:3]], axis=1).astype(np.float32)
    return np.concatenate([f[:1], f[:-1]], axis=0)


def x_from_frames(S: np.ndarray) -> np.ndarray:
    """(T, 30) 5-frame history obs from per-decision states S (repeat-first padding)."""
    T = len(S)
    Spad = np.concatenate([np.repeat(S[:1], N_FRAMES - 1, axis=0), S], axis=0)
    hist = np.stack([Spad[k : k + T] for k in range(N_FRAMES)], axis=1)
    return np.concatenate([hist[:, :, 0:3].reshape(T, -1), hist[:, :, 3:6].reshape(T, -1)], axis=1)


@torch.no_grad()
def recurrent_rollout(actor, S: np.ndarray, a_exec: np.ndarray):
    """Replay the deployed HistoryAgent on logged states. Returns x (T,30), windows (T,50,8),
    prev (T,2), the policy's own actions (T,2)."""
    T = len(S)
    x = x_from_frames(S)
    prev = np.concatenate([np.zeros((1, ACT_DIM), np.float32), a_exec[:-1].astype(np.float32)], axis=0)
    feats = S  # latest_frame step features = [paddle x,y,flag, puck x,y,flag] of the newest frame
    buf = np.zeros((HISTORY_LEN, FEAT_DIM), np.float32)
    buf[:, :6] = feats[0]
    windows = np.zeros((T, HISTORY_LEN, FEAT_DIM), np.float32)
    a_pol = np.zeros((T, ACT_DIM), np.float32)
    xt, pt = torch.as_tensor(x), torch.as_tensor(prev)
    for i in range(T):
        windows[i] = buf
        z = actor.encoder(torch.as_tensor(buf).unsqueeze(0))
        a = actor.get_action_from_latent(xt[i : i + 1], pt[i : i + 1], z).numpy()[0]
        a_pol[i] = a
        buf[:-1] = buf[1:]
        buf[-1, :6] = feats[i]
        buf[-1, 6:] = a
    return x, windows, prev, a_pol


@torch.no_grad()
def embed(actor, x: np.ndarray, windows: np.ndarray, prev: np.ndarray, bs: int = 4096) -> np.ndarray:
    """h(O) = ResidualMLPTrunk output (input of actor_mean_head)."""
    out = []
    for i in range(0, len(x), bs):
        z = actor.encoder(torch.as_tensor(windows[i : i + bs]))
        inp = torch.cat([torch.as_tensor(x[i : i + bs]), torch.as_tensor(prev[i : i + bs]), z], dim=-1)
        out.append(actor.actor(inp).numpy())
    return np.concatenate(out).astype(np.float32)


def load_policy_episodes(data_dir: Path, seed: str, actor, rng: np.random.Generator, recurrent: bool = False) -> list[dict]:
    """Episodes of seed folder ``seed``, embedded by ``actor``.

    Default (recurrent=False): the window holds the logged (executed) actions, which reproduces 96 %
    of the logged actions to 1e-3 with the collecting policy (recurrent replay compounds every
    mismatch and does worse). ``action_match`` is only meaningful when ``actor`` collected the data.
    """
    eps = []
    meta_path = next((data_dir / seed).glob("data_*/eval_per_episode.jsonl"))
    for line in meta_path.read_text().splitlines():
        m = json.loads(line)
        rel = m["artifact_path"].split("episode_hdf5/")[1]
        with h5py.File(meta_path.parent / "episode_hdf5" / rel, "r") as h:
            paddle_xy = h["pose"][:, :2].astype(np.float64)
            puck = h["puck"][:, :3].astype(np.float64)
            a_exec = h["policy_action"][:, :2].astype(np.float32)
        hits = juggle_hit_rows(paddle_xy, puck)
        S = frames_from_rows(paddle_xy, puck)
        T = len(S)
        if recurrent:
            x, W, prev, a_pol = recurrent_rollout(actor, S, a_exec)
        else:
            O = build_policy_inputs(paddle_xy, puck, a_exec)
            x = O[:, :OBS_DIM]
            W = O[:, OBS_DIM : OBS_DIM + HISTORY_LEN * FEAT_DIM].reshape(T, HISTORY_LEN, FEAT_DIM).copy()
            prev = O[:, OBS_DIM + HISTORY_LEN * FEAT_DIM :]
            with torch.no_grad():
                a_pol = actor.get_action(torch.as_tensor(O)).numpy()
        executed = np.abs(a_exec).sum(1) > 0  # outside the post-reset hold
        match = float((np.abs(a_pol - a_exec).max(1)[executed] < 1e-3).mean())
        # positive control 1: shuffle each window in time
        W_shuf = W.copy()
        for i in range(T):
            W_shuf[i] = W_shuf[i][rng.permutation(HISTORY_LEN)]
        # positive control 2: swap the puck history (x and window) for a random other time step's
        j = rng.integers(0, T, size=T)
        x_swap = x.copy()
        x_swap[:, 15:30] = x[j, 15:30]
        W_swap = W.copy()
        W_swap[:, :, 3:6] = W[j, :, 3:6]
        eps.append(
            {
                "name": f"{seed}_ep{m['episode_id']}",
                "end_reason": m["episode_end_reason"],
                "T": T,
                "hits": hits,
                "t_last": hits[-1] if hits else -1,
                "action_match": match,
                "E": embed(actor, x, W, prev),
                "E_shuffle": embed(actor, x, W_shuf, prev),
                "E_puckswap": embed(actor, x_swap, W_swap, prev),
            }
        )
    return eps


# ----------------------------------------------------------------------------- RND with early stopping


def train_rnd_es(E_tr, E_val, args, device, writer, tag, monitor):
    torch.manual_seed(args.seed)
    mean, std = E_tr.mean(0), E_tr.std(0)
    std[std < 1e-6] = 1.0
    model = RND(E_tr.shape[1], args.m, mean, std).to(device)
    opt = torch.optim.Adam(model.f.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    X = torch.as_tensor(E_tr, device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    best, best_state, best_step = float("inf"), None, 0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(X), (args.batch_size,), device=device, generator=gen)
        s = model(X[idx])
        loss = (s**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == 1:
            writer.add_scalar(f"{tag}/loss", loss.item(), step)
            writer.add_scalar(f"{tag}/sRND_train_batch", s.mean().item(), step)
        if step % args.eval_every == 0 or step == args.steps:
            v = float(score(model, E_val, device).mean())
            writer.add_scalar(f"{tag}/sRND_mean_val_chunk", v, step)
            for name, Em in monitor.items():
                if len(Em):
                    writer.add_scalar(f"{tag}/sRND_mean_{name}", float(score(model, Em, device).mean()), step)
            if v < best:
                best, best_step, best_state = v, step, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    writer.add_scalar(f"{tag}/best_step", best_step, 0)
    return model, best_step


def split_train_val(ep: dict, frac: float):
    """ID part -> (first 1-frac, last frac) contiguous chunks."""
    n = ep["t_last"] + 1
    cut = int(round(n * (1 - frac)))
    return ep["E"][:cut], ep["E"][cut:n]


# ----------------------------------------------------------------------------- main


def run_seed(seed: str, args, device, writer) -> dict:
    actor = load_phase1(str(Path("runs/analysis/swa") / POLICIES[seed]))["actor"].cpu().eval()
    rng = np.random.default_rng(args.seed)
    eps = load_policy_episodes(args.data_dir, seed, actor, rng)
    for e in eps:
        print(f"[{seed}] {e['name']}: T={e['T']} hits={len(e['hits'])} action match {e['action_match']:.3f}", flush=True)
    id_eps = [e for e in eps if e["t_last"] >= 0]
    folds = {}
    for e in id_eps:
        others = [o for o in id_eps if o is not e]
        tr, va = zip(*[split_train_val(o, args.val_frac) for o in others])
        E_tr, E_val = np.concatenate(tr), np.concatenate(va)
        n_id = e["t_last"] + 1
        test = {
            "id_unseen": e["E"][:n_id],
            "tail_unseen": e["E"][n_id:] if e["end_reason"] == "puck_hit_bottom" else np.zeros((0, E_tr.shape[1]), np.float32),
            "ctrl_shuffle": e["E_shuffle"][:n_id],
            "ctrl_puckswap": e["E_puckswap"][:n_id],
        }
        t0 = time.time()
        model, best_step = train_rnd_es(E_tr, E_val, args, device, writer, f"{seed}/{e['name']}", {"id_train": E_tr, **test})
        s_val = score(model, E_val, device)
        res = {
            "train_episodes": [o["name"] for o in others],
            "best_step": best_step,
            "n_train": int(len(E_tr)),
            "thr_p95": float(np.percentile(s_val, 95)),
            "thr_p99": float(np.percentile(s_val, 99)),
            "scores": {"train_fit": score(model, E_tr, device).tolist(), "val_chunk": s_val.tolist(),
                       **{k: score(model, v, device).tolist() for k, v in test.items()}},
        }
        folds[e["name"]] = res
        print(f"[{seed}] fold {e['name']}: best step {best_step}, train {np.mean(res['scores']['train_fit']):.4f} "
              f"val {s_val.mean():.4f} unseen-ID {np.mean(res['scores']['id_unseen']):.4f} "
              f"tail {np.mean(res['scores']['tail_unseen']) if len(test['tail_unseen']) else float('nan'):.4f} "
              f"shuffle {np.mean(res['scores']['ctrl_shuffle']):.4f} puckswap {np.mean(res['scores']['ctrl_puckswap']):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    return {"episodes": {e["name"]: {"T": e["T"], "n_hits": len(e["hits"]), "t_last": e["t_last"],
                                     "end_reason": e["end_reason"], "action_match": e["action_match"]} for e in eps},
            "folds": folds}


def run_shared(encoder_seed: str, args, device, writer) -> dict:
    """One frozen actor (``encoder_seed``) embeds every episode; K-fold split by episode.

    Per fold: train RND on the ID part of the training episodes (minus their last-20 % validation
    chunk, used for early stopping and the OOD threshold); evaluate on the held-out episodes'
    ID part (holdout ID) and post-last-hit part (holdout OOD candidate). Results are keyed by
    held-out episode so ``summarize`` aggregates over episodes.
    """
    actor = load_phase1(str(Path("runs/analysis/swa") / POLICIES[encoder_seed]))["actor"].cpu().eval()
    rng = np.random.default_rng(args.seed)
    eps = []
    for seed in POLICIES:
        for e in load_policy_episodes(args.data_dir, seed, actor, rng):
            eps.append(e)
            print(f"[{seed}] {e['name']}: T={e['T']} hits={len(e['hits'])} action match vs own logs {e['action_match']:.3f}", flush=True)
    id_eps = [e for e in eps if e["t_last"] >= 0]
    order = np.random.default_rng(args.seed).permutation(len(id_eps))
    out = {}
    for k in range(args.folds):
        test_eps = [id_eps[i] for i in order[k :: args.folds]]
        train_eps = [e for e in id_eps if all(e is not t for t in test_eps)]
        tr, va = zip(*[split_train_val(o, args.val_frac) for o in train_eps])
        E_tr, E_val = np.concatenate(tr), np.concatenate(va)
        def cat(key, part):
            parts = [part(e) for e in test_eps]
            parts = [p for p in parts if len(p)]
            return np.concatenate(parts) if parts else np.zeros((0, E_tr.shape[1]), np.float32)
        monitor = {
            "holdout_id": cat("E", lambda e: e["E"][: e["t_last"] + 1]),
            "holdout_tail": cat("E", lambda e: e["E"][e["t_last"] + 1 :] if e["end_reason"] == "puck_hit_bottom" else e["E"][:0]),
            "ctrl_shuffle": cat("E", lambda e: e["E_shuffle"][: e["t_last"] + 1]),
            "ctrl_puckswap": cat("E", lambda e: e["E_puckswap"][: e["t_last"] + 1]),
        }
        t0 = time.time()
        model, best_step = train_rnd_es(E_tr, E_val, args, device, writer, f"fold{k}", {"id_train": E_tr, **monitor})
        s_val = score(model, E_val, device)
        s_tr = score(model, E_tr, device)
        thr95, thr99 = float(np.percentile(s_val, 95)), float(np.percentile(s_val, 99))
        for e in test_eps:
            n_id = e["t_last"] + 1
            tail = e["E"][n_id:] if e["end_reason"] == "puck_hit_bottom" else e["E"][:0]
            out[e["name"]] = {
                "fold": k, "train_episodes": [o["name"] for o in train_eps], "best_step": best_step,
                "n_train": int(len(E_tr)), "thr_p95": thr95, "thr_p99": thr99,
                "scores": {"train_fit": s_tr.tolist(), "val_chunk": s_val.tolist(),
                           "id_unseen": score(model, e["E"][:n_id], device).tolist(),
                           "tail_unseen": score(model, tail, device).tolist(),
                           "ctrl_shuffle": score(model, e["E_shuffle"][:n_id], device).tolist(),
                           "ctrl_puckswap": score(model, e["E_puckswap"][:n_id], device).tolist()},
            }
        print(f"[fold {k}] test {[e['name'] for e in test_eps]} best step {best_step}: train {s_tr.mean():.4f} "
              f"val {s_val.mean():.4f} holdout-ID {score(model, monitor['holdout_id'], device).mean():.4f} "
              f"holdout-tail {score(model, monitor['holdout_tail'], device).mean():.4f} "
              f"shuffle {score(model, monitor['ctrl_shuffle'], device).mean():.4f} "
              f"puckswap {score(model, monitor['ctrl_puckswap'], device).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    return {"encoder_seed": encoder_seed,
            "episodes": {e["name"]: {"T": e["T"], "n_hits": len(e["hits"]), "t_last": e["t_last"],
                                     "end_reason": e["end_reason"], "action_match": e["action_match"]} for e in eps},
            "folds": out}


def summarize(out: Path) -> dict:
    per_seed = {p.stem.split("_", 1)[1]: json.loads(p.read_text()) for p in sorted(out.glob("seed_*.json"))}
    sets = ["train_fit", "val_chunk", "id_unseen", "tail_unseen", "ctrl_shuffle", "ctrl_puckswap"]
    rows = []  # one row per fold (= held-out episode)
    for seed, r in per_seed.items():
        for ep, f in r["folds"].items():
            s = {k: np.asarray(f["scores"][k]) for k in sets}
            row = {"seed": seed, "episode": ep, "best_step": f["best_step"], "n_train": f["n_train"]}
            for k in sets:
                row[f"mean_{k}"] = float(s[k].mean()) if len(s[k]) else float("nan")
            for q in ("p95", "p99"):
                for k in ("id_unseen", "tail_unseen", "ctrl_shuffle", "ctrl_puckswap", "val_chunk"):
                    row[f"ood_rate_{q}_{k}"] = float((s[k] > f[f"thr_{q}"]).mean()) if len(s[k]) else float("nan")
            row["auroc_unseenID_vs_train"] = auroc(s["train_fit"], s["id_unseen"])
            row["auroc_unseenID_vs_val"] = auroc(s["val_chunk"], s["id_unseen"])
            row["auroc_tail_vs_unseenID"] = auroc(s["id_unseen"], s["tail_unseen"]) if len(s["tail_unseen"]) else float("nan")
            row["auroc_shuffle_vs_unseenID"] = auroc(s["id_unseen"], s["ctrl_shuffle"])
            row["auroc_puckswap_vs_unseenID"] = auroc(s["id_unseen"], s["ctrl_puckswap"])
            # tail profile aligned at termination, normalised by this fold's p99 threshold
            if len(s["tail_unseen"]):
                row["tail_last10_ood_rate_p99"] = float((s["tail_unseen"][-10:] > f["thr_p99"]).mean())
                row["tail_before_last10_ood_rate_p99"] = float((s["tail_unseen"][:-10] > f["thr_p99"]).mean()) if len(s["tail_unseen"]) > 10 else float("nan")
            rows.append(row)
    keys = [k for k in rows[0] if k not in ("seed", "episode")]
    agg = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])], dtype=np.float64)
        agg[k] = {"n": int(len(v)), "mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")}
    summary = {"aggregate_over_folds": agg, "per_fold": rows,
               "action_match": {ep: m["action_match"] for r in per_seed.values() for ep, m in r["episodes"].items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/eval_new_sim_T10000"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", default=list(POLICIES))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--shared-encoder", default=None, help="embed every episode with this seed's actor, K-fold by episode")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        s = summarize(args.out_dir)
        print(json.dumps(s["aggregate_over_folds"], indent=1))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.shared_encoder:
        writer = SummaryWriter(str(args.out_dir / "tb" / f"shared_{args.shared_encoder}"))
        r = run_shared(args.shared_encoder, args, device, writer)
        r["args"] = {k: str(v) for k, v in vars(args).items()}
        (args.out_dir / f"seed_shared-{args.shared_encoder}.json").write_text(json.dumps(r))
        writer.close()
        s = summarize(args.out_dir)
        print(json.dumps(s["aggregate_over_folds"], indent=1))
        return
    for seed in args.seeds:
        writer = SummaryWriter(str(args.out_dir / "tb" / seed))
        r = run_seed(seed, args, device, writer)
        r["args"] = {k: str(v) for k, v in vars(args).items()}
        (args.out_dir / f"seed_{seed}.json").write_text(json.dumps(r))
        writer.close()


if __name__ == "__main__":
    main()
