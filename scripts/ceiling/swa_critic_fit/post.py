"""Post-processing of one critic-fit run (frozen SWA actor, critics trained):

1. actor check: run model.pth and actor_target.pth must equal the SWA actor exactly;
2. Q-vs-return check: on rollouts of the SWA actor in the training env (new sim + latency 2 + term03 shaping, cap raised
   to 3000 so returns from late states are not cut by truncation -- the trainer bootstraps through truncation, so Q
   does not depend on the cap), compare min(Q1, Q2)(s_t, a_t) (h-inverse applied) with the discounted shaped return
   sum_k gamma^k r_{t+k} (to episode termination or 1500 steps, gamma^1500 = 5e-4). Done for the fitted critics and for
   the warm-start critics (raw checkpoint, paired with a different actor). Two rollout policies: deterministic, and
   with TD3's target-smoothing noise (N(0, 0.2) clipped to 0.5), which is the policy the critic target actually uses;
3. copy the checkpoint files into latest_models/ceiling/newsim_scratch_cap7000_swa/seedN/.

    python post.py cap7000_sN [--episodes 16] [--no-copy]
"""
import argparse, copy, json, shutil, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.rma.action_latency import install_action_latency
from scripts.rma.env_wrapper import PrivilegedHistoryEnvVector
from scripts.rma.motion_shaping import MotionShaping, install_motion_shaping
from scripts.rma.networks import PrivilegedHistoryActor
from scripts.rma.train_base_policy import make_env
from scripts.td3.helper.q_network import TD3QNetwork
from scripts.td3.helper.td3_graphed_update import deterministic_actor_action, h_inverse
from scripts.td3.td3_training import augment_policy_observation

ROOT = REPO / "runs/ceiling/swa_critic_fit"
T_POINTS = (50, 250, 500, 1000, 1500)
HORIZON = 1500
torch.set_num_threads(4)


def load_sd(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def rollouts(job, a, cfg, actor, critic_sets, episodes, noisy, seed_base):
    sim = copy.deepcopy(cfg["air_hockey"]); sim["max_timesteps"] = HORIZON + max(T_POINTS) + 1
    shaping = MotionShaping.from_args(a)
    base = make_env(sim)
    env_fn = lambda: install_motion_shaping(install_action_latency(base(), a.action_latency_steps), shaping)  # noqa: E731
    envs = PrivilegedHistoryEnvVector(env_fn, a.history_len, a.step_features)
    g, eps = a.gamma, a.h_transform_eps
    rng = torch.Generator().manual_seed(seed_base)
    rows = []
    for ep in range(episodes):
        obs, _ = envs.reset(seed=seed_base + ep)
        last = torch.zeros((1, 2))
        O, A, R = [], [], []
        terminated = False
        while True:
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                act = deterministic_actor_action(actor, augment_policy_observation(o, last, a.use_last_action_in_policy_state))
            if noisy:
                act = act + (torch.randn(act.shape, generator=rng) * a.policy_noise).clamp(-a.noise_clip, a.noise_clip)
            act = act.clamp(-1, 1)
            O.append(o[0]); A.append(act[0])
            obs, r, term, trunc, _ = envs.step(act.numpy())
            R.append(float(r[0])); last = act.clone()
            if term[0] or trunc[0]:
                terminated = bool(term[0]); break
        n = len(R); R = np.asarray(R)
        for t in T_POINTS:
            if t >= n or (not terminated and t + HORIZON > n):
                continue
            k = R[t:t + HORIZON]; mc = float(np.sum(k * g ** np.arange(len(k))))
            row = {"ep": ep, "t": t, "mc": mc, "ep_len": n, "terminated": terminated}
            with torch.no_grad():
                for name, (q1, q2) in critic_sets.items():
                    oo, aa = O[t][None], A[t][None]
                    v1 = float(h_inverse(q1(oo, aa), eps=eps)); v2 = float(h_inverse(q2(oo, aa), eps=eps))
                    row[name] = {"q1": v1, "q2": v2, "qmin": min(v1, v2)}
            rows.append(row)
    return rows


def summarize(rows, names):
    out = {"n_states": len(rows), "mc_mean": float(np.mean([r["mc"] for r in rows])) if rows else None}
    for nm in names:
        if not rows:
            break
        mc = np.array([r["mc"] for r in rows]); q = np.array([r[nm]["qmin"] for r in rows])
        err = q - mc
        out[nm] = {
            "qmin_mean": float(q.mean()),
            "bias": float(err.mean()), "bias_se": float(err.std(ddof=1) / np.sqrt(len(err))) if len(err) > 1 else None,
            "mae": float(np.abs(err).mean()),
            "rel_bias": float(err.mean() / max(abs(mc.mean()), 1e-8)),
            "corr": float(np.corrcoef(q, mc)[0, 1]) if len(q) > 2 and q.std() > 0 and mc.std() > 0 else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("job"); ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--no-copy", action="store_true"); ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--actor", default=None, help="actor to roll out (default: the SWA actor); also skips the actor check")
    o = ap.parse_args()
    seed = int(o.job.rsplit("_s", 1)[1])
    run = Path(o.run_dir) if o.run_dir else ROOT / o.job
    dest = REPO / f"latest_models/ceiling/newsim_scratch_cap7000_swa/seed{seed}"
    a = SimpleNamespace(**yaml.safe_load(open(run / "args.yaml")))
    cfg = yaml.safe_load(open(run / "config.yaml"))
    report = {"job": o.job, "run_dir": str(run.relative_to(REPO)) if run.is_relative_to(REPO) else str(run)}

    # 1. actor identity
    swa = load_sd(o.actor) if o.actor else load_sd(dest / "model.pth")
    report["actor_equals_swa"] = same(load_sd(run / "model.pth"), swa)
    report["actor_target_equals_swa"] = same(load_sd(run / "actor_target.pth"), swa)
    ts = load_sd(run / "training_state.pth")
    report["global_step"] = int(ts["global_step"]); report["total_critic_updates"] = int(ts["total_critic_updates"])
    print(json.dumps({k: report[k] for k in ("actor_equals_swa", "actor_target_equals_swa", "global_step")}), flush=True)

    # 2. Q vs return
    meta = json.load(open(run / "rma_meta.json"))
    actor = PrivilegedHistoryActor(privileged_dim=8, obs_dim=meta["obs_dim"], act_dim=2, history_len=a.history_len,
                                   feature_dim=meta["feature_dim"], latent_dim=a.rma_latent_dim,
                                   use_last_action=a.use_last_action_in_policy_state,
                                   hidden_layer_size=a.agent_hidden_layer_size, num_hidden_layers=a.agent_num_hidden_layers,
                                   embed_dim=a.history_embed_dim, conv_channels=a.history_conv_channels,
                                   action_scale=1, action_bias=0.0)
    actor.load_state_dict(swa); actor.step_features = a.step_features; actor.eval()

    def critic(sd):
        q = TD3QNetwork(obs_dim=sd["trunk.blocks.0.units.0.0.weight"].shape[1] - 2, act_dim=2,
                        hidden_layer_size=a.q_hidden_layer_size, num_hidden_layers=a.q_num_hidden_layers)
        q.load_state_dict(sd); return q.eval()

    sets = {"fitted": (critic(load_sd(run / "qf1.pth")), critic(load_sd(run / "qf2.pth")))}
    if a.model_path:
        init = load_sd(REPO / a.model_path)
        sets["warm_start_raw"] = (critic(init["qf1"]), critic(init["qf2"]))
    report["q_check"] = {}
    for label, noisy in (("deterministic", False), ("target_smoothing_noise", True)):
        rows = rollouts(o.job, a, cfg, actor, sets, o.episodes, noisy, seed_base=2_000_000)
        report["q_check"][label] = summarize(rows, list(sets))
        (Path(o.out_prefix + f".qcheck_{label}.jsonl") if o.out_prefix else ROOT / f"{o.job}.qcheck_{label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(label, json.dumps(report["q_check"][label]), flush=True)

    # 3. copy into latest_models
    if not o.no_copy:
        if not (report["actor_equals_swa"] and report["actor_target_equals_swa"]):
            raise SystemExit("actor changed; not copying")
        if (dest / "args.yaml").exists() and not (dest / "args_actor_training.yaml").exists():
            shutil.copy2(dest / "args.yaml", dest / "args_actor_training.yaml")  # the SWA actor's own training args
        for f in ("actor_target.pth", "qf1.pth", "qf2.pth", "qf1_target.pth", "qf2_target.pth", "training_state.pth",
                  "args.yaml", "config.yaml"):
            shutil.copy2(run / f, dest / f)
        report["copied_to"] = str(dest.relative_to(REPO))
    (Path(o.out_prefix + ".post.json") if o.out_prefix else ROOT / f"{o.job}.post.json").write_text(json.dumps(report, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
