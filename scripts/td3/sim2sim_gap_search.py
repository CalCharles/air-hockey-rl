"""Sim2sim gap search: find in-distribution envs where the sim-trained hist4 policy loses
performance, then train oracles there and run the 4-trial data-recycling fine-tune.

Motivation: with filter + PID control, 5 random envs in the ±25 % box over paddle_density /
puck_damping / gravity showed no zero-shot gap (runs/td3/sim2sim_online_finetune). This
script widens the perturbation to every sysid-v2 parameter and searches for the envs that do
open a gap, as a proxy for the residual sim-to-real mismatch.

Stages (``--stage all`` runs them in order; each is resumable):

1. ``screen``: draw candidate envs, each parameter uniform in ±r of its
   configs/new_juggle/sysid_v2_hist4.yaml value (PARAMS below), and evaluate the 5 source
   policies zero-shot (``--screen-episodes`` deterministic episodes per seed). Gap per seed =
   source return on the nominal env (``--nominal-episodes`` episodes) − zero-shot return; a
   candidate passes when the mean gap ≥ ``--min-gap`` and mean / SE (over seeds) ≥ ``--min-t``.
   Passing candidates are re-evaluated on fresh episode seeds (``--confirm-episodes``) and kept
   only if they pass again (guards against selecting on eval noise). r climbs the
   ``--rel-ranges`` ladder (±5 % .. ±25 %), up to ``--candidates-per-level`` candidates or
   ``--max-envs`` kept envs per level; the search stops at the first (smallest) r with
   ≥ ``--min-envs`` kept envs and gives up after ``--max-search-hours``. Output: screen.jsonl
   (every candidate), screen_sensitivity.txt, envs.json (the kept envs, in the format
   sim2sim_online_finetune's workers read).
2. ``train``: ``--oracle-seeds`` hist4 oracles per kept env (recipe of
   configs/td3/sysid_v2_hist/juggle_sysid_v2_hist4.yaml, ``--total-timesteps``), in train/.
3. ``finetune``: the data-recycling fine-tune of sim2sim_online_finetune (``--recycle-trials``
   trials of ``--num-online-episodes`` episodes, α 0.5 -> 1 over each trial) for every source
   seed in every kept env, in online_recycle_alpha0.5/. Runs alongside ``train`` with ``all``
   (it does not need the oracles).
4. ``report``: every policy (source zero-shot, oracles, fine-tuned per trial) re-evaluated on
   one fresh set of ``--report-episodes`` episode seeds per env, tables to stdout and report.md,
   TensorBoard aggregation of the fine-tuning curves.

    .venv/bin/python -m scripts.td3.sim2sim_gap_search --stage all --root runs/td3/sim2sim_gap_search
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.td3 import sim2sim_online_finetune as s2s

# Every parameter of the sysid-v2 header comment, each uniform in ±25 % of its sysid-v2 value.
# paddle_restitution and pid_kd are 0 in sysid v2, so they are drawn in [0, 25 % of their v1
# value] instead (ZERO_REF); hist_len (the PID-target moving-average length) is an integer,
# drawn uniformly from round(4 · [0.75, 1.25]) = {3, 4, 5}.
PARAMS = ["gravity", "puck_damping", "side_wall_restitution", "end_wall_restitution", "puck_restitution",
          "paddle_restitution", "pid_kp", "pid_ki", "pid_kd", "paddle_density", "hist_len"]
ZERO_REF = {"paddle_restitution": 1.0, "pid_kd": 50.0}  # sysid v1 values
INT_PARAMS = {"hist_len"}
SHORT = {"gravity": "grav", "puck_damping": "damp", "side_wall_restitution": "side_w", "end_wall_restitution": "end_w",
         "puck_restitution": "puck_rest", "paddle_restitution": "pad_rest", "pid_kp": "kp", "pid_ki": "ki",
         "pid_kd": "kd", "paddle_density": "pad_dens", "hist_len": "hist"}
BASE_CONFIG = s2s.SOURCE_SIM_CONFIG  # configs/new_juggle/sysid_v2_hist4.yaml, the source policies' env
SOURCE_SEEDS = [0, 1, 2, 3, 4]
# Episode-seed index offsets for evaluate_actor (seed = 900000 + 1000 * index + episode):
# screening uses index = candidate id (= the env id the fine-tune workers use for their own
# zero-shot / fine-tuned evals), so confirmation and report get disjoint seed ranges.
NOMINAL_INDEX = 0
CONFIRM_OFFSET = 1_000_000
REPORT_OFFSET = 2_000_000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def base_params() -> dict:
    sp = s2s._load_yaml(BASE_CONFIG)["air_hockey"]["simulator_params"]
    return {n: float(sp[n]) for n in PARAMS}


def ranges(rel: float) -> dict:
    out = {}
    for n, v in base_params().items():
        if n in INT_PARAMS:
            out[n] = [int(round(v * (1 - rel))), int(round(v * (1 + rel)))]
        elif v == 0:
            out[n] = [0.0, rel * ZERO_REF[n]]
        else:
            out[n] = sorted([v * (1 - rel), v * (1 + rel)])
    return out


def draw(rng, box: dict) -> dict:
    return {n: int(rng.integers(box[n][0], box[n][1] + 1)) if n in INT_PARAMS else round(float(rng.uniform(*box[n])), 6)
            for n in PARAMS}


def write_sim_config(path: Path, params: dict, cid: int, rel: float) -> None:
    import copy

    cfg = copy.deepcopy(s2s._load_yaml(BASE_CONFIG))
    cfg["air_hockey"]["simulator_params"].update(params)
    s2s._write_yaml(path, f"sim2sim gap-search candidate {cid}: {BASE_CONFIG} with {', '.join(PARAMS)} drawn uniformly "
                          f"in ±{rel:.0%} of their values (zero-valued ones in [0, {rel:.0%} of v1], hist_len rounded). Generated by scripts/td3/sim2sim_gap_search.py; do not edit.", cfg)


def source_ckpt(seed: int) -> str:
    return str(s2s._repo_path(s2s.SOURCE_CHECKPOINT.format(seed=seed)))


def _eval_task(task) -> tuple:
    """(key, checkpoint, sim_config, env_index, n_episodes) -> (key, eval dict). One CPU thread."""
    import torch

    torch.set_num_threads(1)
    key, ckpt, sim_config, index, n = task
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = state["args"] if isinstance(state, dict) and "args" in state else s2s._load_yaml(Path(ckpt).parent / "args.yaml")
    use_last = bool((arch if isinstance(arch, dict) else vars(arch)).get("use_last_action_in_policy_state", True))
    return key, s2s.evaluate_actor(s2s.load_actor(ckpt, arch), sim_config, index, n, use_last)


def gap_stats(nominal: dict, zs: dict) -> tuple:
    """Paired gap over source seeds: mean, SE, t."""
    g = np.array([nominal[s] - zs[s] for s in SOURCE_SEEDS])
    se = float(g.std(ddof=1) / math.sqrt(len(g)))
    return float(g.mean()), se, float(g.mean() / se) if se > 0 else math.inf


def mean_std(v) -> str:
    v = [x for x in v if x is not None and not math.isnan(x)]
    if not v:
        return "-"
    return f"{np.mean(v):.1f} ± {np.std(v):.1f}"  # population std, as in sim2sim_online_finetune's tables


def mean_se(v) -> str:
    v = [x for x in v if x is not None and not math.isnan(x)]
    if len(v) < 2:
        return f"{np.mean(v):.1f}" if v else "-"
    return f"{np.mean(v):.1f} ± {np.std(v, ddof=1) / math.sqrt(len(v)):.1f}"


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 1. screen
# ---------------------------------------------------------------------------


def nominal_returns(root: Path, cli, pool) -> dict:
    path = root / "source_nominal.json"
    if path.exists():
        d = json.load(open(path))
        if d["n_episodes"] == cli.nominal_episodes:
            return {int(k): v["return_mean"] for k, v in d["per_seed"].items()}
    res = pool.map(_eval_task, [(s, source_ckpt(s), BASE_CONFIG, NOMINAL_INDEX, cli.nominal_episodes) for s in SOURCE_SEEDS])
    with open(path, "w") as f:
        json.dump({"n_episodes": cli.nominal_episodes, "sim_config": BASE_CONFIG, "per_seed": {str(k): v for k, v in res}}, f, indent=2)
    return {k: v["return_mean"] for k, v in res}


def _screen_level(root: Path, cli, pool, nominal: dict, rel: float, done: list, next_id: int, t0: float) -> tuple:
    """Screen candidates drawn in ±``rel`` until ``--max-envs`` confirmed, ``--candidates-per-level``
    screened at this level, or ``--max-search-hours``. Returns (kept, next_id)."""
    rng_box = ranges(rel)
    rng = np.random.default_rng([cli.search_seed, int(round(rel * 1000))])
    level_done = [d for d in done if d["rel_range"] == rel]
    for _ in level_done:  # replay the draws so a resumed search continues the same sequence
        draw(rng, rng_box)
    kept = [d for d in level_done if d.get("confirmed")][:cli.max_envs]
    n_level = len(level_done)
    log_path = root / "screen.jsonl"
    print(f"\n[screen] ===== ±{rel:.0%} ===== ({n_level} candidates already screened, {len(kept)} kept)", flush=True)
    while len(kept) < cli.max_envs and n_level < cli.candidates_per_level and (time.time() - t0) / 3600 < cli.max_search_hours:
        batch = []
        for _ in range(min(cli.batch_candidates, cli.candidates_per_level - n_level)):
            cid = next_id; next_id += 1; n_level += 1
            params = draw(rng, rng_box)
            cfg = root / "candidates" / f"c{cid}_sim.yaml"
            write_sim_config(cfg, params, cid, rel)
            batch.append({"id": cid, "rel_range": rel, "params": params, "sim_config": str(cfg)})
        res = dict(pool.map(_eval_task, [((c["id"], s), source_ckpt(s), c["sim_config"], c["id"], cli.screen_episodes)
                                          for c in batch for s in SOURCE_SEEDS]))
        passed = []
        for c in batch:
            zs = {s: res[(c["id"], s)]["return_mean"] for s in SOURCE_SEEDS}
            g, se, t = gap_stats(nominal, zs)
            c.update(screen_zs=zs, screen_gap=g, screen_se=se, screen_t=t, passed=bool(g >= cli.min_gap and t >= cli.min_t))
            if c["passed"]:
                passed.append(c)
        if passed:
            cres = dict(pool.map(_eval_task, [((c["id"], s), source_ckpt(s), c["sim_config"], c["id"] + CONFIRM_OFFSET,
                                               cli.confirm_episodes) for c in passed for s in SOURCE_SEEDS]))
            for c in passed:
                zs = {s: cres[(c["id"], s)]["return_mean"] for s in SOURCE_SEEDS}
                g, se, t = gap_stats(nominal, zs)
                c.update(confirm_zs=zs, confirm_gap=g, confirm_se=se, confirm_t=t,
                         confirmed=bool(g >= cli.min_gap and t >= cli.min_t))
        with open(log_path, "a") as f:
            for c in batch:
                c.setdefault("confirmed", False)
                f.write(json.dumps(c) + "\n")
                if c["confirmed"] and len(kept) < cli.max_envs:
                    kept.append(c)
                    print(f"[screen] KEEP c{c['id']} (±{rel:.0%}): gap {c['screen_gap']:.1f} ± {c['screen_se']:.1f} -> "
                          f"confirmed {c['confirm_gap']:.1f} ± {c['confirm_se']:.1f}  ({len(kept)}/{cli.max_envs})", flush=True)
        print(f"[screen] ±{rel:.0%}: {n_level}/{cli.candidates_per_level} candidates, "
              f"{sum(1 for c in batch if c['passed'])}/{len(batch)} passed screen this batch, {len(kept)} kept, "
              f"{(time.time() - t0) / 60:.1f} min total", flush=True)
    return kept, next_id


def screen(root: Path, cli) -> list:
    """Climb the ``--rel-ranges`` ladder (smallest first) and stop at the first level that
    yields ≥ ``--min-envs`` confirmed envs: the smallest perturbation that opens a gap."""
    envs_path = root / "envs.json"
    if envs_path.exists():
        envs = json.load(open(envs_path))["envs"]
        print(f"[screen] {envs_path} already holds {len(envs)} env(s); skipping the search", flush=True)
        return envs
    (root / "candidates").mkdir(parents=True, exist_ok=True)
    log_path = root / "screen.jsonl"
    done = [json.loads(l) for l in open(log_path)] if log_path.exists() else []
    next_id = max((d["id"] for d in done), default=0) + 1
    t0 = time.time()
    kept, rel = [], None
    with Pool(processes=cli.cpu_workers) as pool:
        nominal = nominal_returns(root, cli, pool)
        print(f"[screen] source on nominal env ({cli.nominal_episodes} eps/seed): "
              + ", ".join(f"s{s} {nominal[s]:.1f}" for s in SOURCE_SEEDS) + f"  (mean {np.mean(list(nominal.values())):.1f})", flush=True)
        print(f"[screen] pass = mean gap ≥ {cli.min_gap} and gap/SE ≥ {cli.min_t} on {cli.screen_episodes} eps/seed, "
              f"then again on {cli.confirm_episodes} fresh eps/seed; ladder {[f'±{r:.0%}' for r in cli.rel_ranges]}", flush=True)
        for rel in sorted(cli.rel_ranges):
            kept, next_id = _screen_level(root, cli, pool, nominal, rel, done, next_id, t0)
            if len(kept) >= cli.min_envs or (time.time() - t0) / 3600 >= cli.max_search_hours:
                break
    sensitivity(root)
    if len(kept) < cli.min_envs:
        print(f"[screen] only {len(kept)} env(s) found (< {cli.min_envs}) up to ±{rel:.0%} / {cli.max_search_hours} h: "
              "giving up.", flush=True)
        return []
    envs = [{"env": c["id"], "params": c["params"], "sim_config": c["sim_config"]} for c in kept]
    with open(envs_path, "w") as f:
        json.dump({"base_config": BASE_CONFIG, "source_sim_config": BASE_CONFIG, "rel_range": rel,
                   "dr_params": PARAMS, "ranges": ranges(rel), "means": base_params(), "selection": vars(cli),
                   "envs": envs}, f, indent=2, default=str)
    print(f"[screen] kept {len(envs)} env(s) at ±{rel:.0%}: {[e['env'] for e in envs]} -> {envs_path}", flush=True)
    return envs


def sensitivity(root: Path) -> str:
    """Per ladder level: pass rate and gap quantiles; Spearman ρ of the gap with each
    parameter's relative deviation from sysid v2, pooled over levels (which knobs open the gap)."""
    rows = [json.loads(l) for l in open(root / "screen.jsonl")]
    if len(rows) < 10:
        return ""
    lines = [f"Screening: {len(rows)} candidates."]
    for rel in sorted({r["rel_range"] for r in rows}):
        rr = [r for r in rows if r["rel_range"] == rel]
        g = np.array([r["screen_gap"] for r in rr])
        lines.append(f"  ±{rel:.0%}: {len(rr)} screened, {sum(r['passed'] for r in rr)} passed screen, "
                     f"{sum(r['confirmed'] for r in rr)} confirmed; gap mean {g.mean():.1f}, p90 {np.percentile(g, 90):.1f}, "
                     f"max {g.max():.1f}")
    base = base_params()
    gaps = np.array([r["screen_gap"] for r in rows])
    rank = lambda x: np.argsort(np.argsort(x))
    out = []
    for n in PARAMS:
        ref = base[n] if base[n] != 0 else ZERO_REF[n]
        x = np.array([(r["params"][n] - base[n]) / abs(ref) for r in rows])
        if np.ptp(x) == 0:
            continue
        out.append((n, np.corrcoef(rank(x), rank(gaps))[0, 1]))
    out.sort(key=lambda t: -abs(t[1]))
    lines.append("Spearman ρ(gap, parameter) (positive = larger parameter value -> bigger drop; gravity is negative, "
                 "so positive = weaker gravity):")
    lines += [f"  {n:24s} {rho:+.2f}" for n, rho in out]
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    (root / "screen_sensitivity.txt").write_text(text + "\n")
    return text


# ---------------------------------------------------------------------------
# 2 + 3. oracles and data-recycling fine-tune
# ---------------------------------------------------------------------------


def recycle_cli(root: Path, cli) -> SimpleNamespace:
    return SimpleNamespace(recycle_alpha_start=0.5, recycle_trials=cli.recycle_trials, overwrite=False,
                           num_online_episodes=cli.num_online_episodes, eval_episodes=cli.screen_episodes,
                           online_eval_every=cli.online_eval_every, online_eval_episodes=cli.online_eval_episodes,
                           min_episode_steps=1, source_checkpoint=s2s.SOURCE_CHECKPOINT, warm_start_episodes=0,
                           recycle_anneal_episodes=cli.num_online_episodes, policy_lr=None,
                           allow_hist_mismatch=True)  # hist_len is one of the sampled parameters


def train(root: Path, envs: list, cli) -> None:
    jobs = s2s.build_train_jobs(root, envs, cli.oracle_seeds, cli.total_timesteps, 0)
    print(f"[train] {len(jobs)} oracle run(s), {cli.train_jobs_per_gpu} per GPU on GPUs {cli.gpus}", flush=True)
    s2s.run_jobs(jobs, cli.gpus, cli.train_jobs_per_gpu, poll_s=30.0)


def finetune(root: Path, envs: list, cli) -> None:
    jobs = s2s.build_recycle_jobs(root, envs, SOURCE_SEEDS, recycle_cli(root, cli))
    print(f"[finetune] {len(jobs)} recycling job(s) ({cli.recycle_trials} trials × {cli.num_online_episodes} eps), "
          f"{cli.finetune_jobs_per_gpu} per GPU on GPUs {cli.gpus}", flush=True)
    s2s.run_jobs(jobs, cli.gpus, cli.finetune_jobs_per_gpu, poll_s=15.0)


# ---------------------------------------------------------------------------
# 4. report
# ---------------------------------------------------------------------------


def report(root: Path, envs: list, cli) -> None:
    rc = recycle_cli(root, cli)
    cache_path = root / "report_eval.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    if cache.get("n_episodes") != cli.report_episodes:
        cache = {"n_episodes": cli.report_episodes, "evals": {}}
    tasks = []
    for env in envs:
        e, cfg, idx = env["env"], env["sim_config"], env["env"] + REPORT_OFFSET
        for s in SOURCE_SEEDS:
            tasks.append((f"zs/{e}/{s}", source_ckpt(s), cfg, idx, cli.report_episodes))
            for t in range(cli.recycle_trials):
                p = s2s.recycle_dir(root, 0.5, t, e, s) / "final" / "training_state.pth"
                if p.exists():
                    tasks.append((f"ft{t}/{e}/{s}", str(p), cfg, idx, cli.report_episodes))
        for s in cli.oracle_seeds:
            p = s2s.train_run_dir(root, e, s) / "model.pth"
            if p.exists():
                tasks.append((f"oracle/{e}/{s}", str(p), cfg, idx, cli.report_episodes))
    tasks = [t for t in tasks if t[0] not in cache["evals"]]
    if tasks:
        print(f"[report] evaluating {len(tasks)} policies × {cli.report_episodes} episodes...", flush=True)
        with Pool(processes=min(cli.cpu_workers, len(tasks))) as pool:
            for k, v in pool.imap_unordered(_eval_task, tasks):
                cache["evals"][k] = {key: v[key] for key in ("return_mean", "return_sem", "length_mean", "juggles_mean", "contacts_mean")}
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    ev = cache["evals"]
    nominal = json.load(open(root / "source_nominal.json"))
    nom = [v["return_mean"] for v in nominal["per_seed"].values()]
    screen_rows = {c["id"]: c for c in map(json.loads, open(root / "screen.jsonl"))}

    def col(kind, e, seeds, metric="return_mean"):
        return [ev[f"{kind}/{e}/{s}"][metric] for s in seeds if f"{kind}/{e}/{s}" in ev]

    T = cli.recycle_trials
    md = [f"# Sim2sim gap search — {root}\n",
          f"Source: the 5 sysid-v2 hist4 juggle policies (filter + PID), `{s2s.SOURCE_CHECKPOINT}`. Every policy "
          f"re-evaluated on the same {cli.report_episodes} fresh deterministic episodes per env (not the screening "
          f"episodes, so the zero-shot column carries no selection bias). Env rows: mean ± std over seeds "
          f"(5 source seeds; {len(cli.oracle_seeds)} oracle seeds). Mean row: mean ± SE across envs.\n",
          f"Source on its own nominal env: {mean_std(nom)} ({nominal['n_episodes']} episodes/seed).\n"]

    for metric, title in (("return_mean", "Juggle return"), ("juggles_mean", "Juggles per episode"),
                          ("length_mean", "Episode length")):
        header = ["env", "oracle", "zero-shot"] + [f"trial {t + 1}" for t in range(T)] + ["oracle − zs"]
        rows, agg = [], {k: [] for k in ["oracle", "zs"] + [f"ft{t}" for t in range(T)] + ["gap"]}
        for env in envs:
            e = env["env"]
            o, z = col("oracle", e, cli.oracle_seeds, metric), col("zs", e, SOURCE_SEEDS, metric)
            fts = [col(f"ft{t}", e, SOURCE_SEEDS, metric) for t in range(T)]
            gap = np.mean(o) - np.mean(z) if o and z else math.nan
            rows.append([f"c{e}", mean_std(o), mean_std(z)] + [mean_std(f) for f in fts] + [f"{gap:+.1f}"])
            for k, v in [("oracle", o), ("zs", z)] + [(f"ft{t}", f) for t, f in enumerate(fts)]:
                if v:
                    agg[k].append(np.mean(v))
            agg["gap"].append(gap)
        rows.append(["**Mean ± SE**"] + [f"**{mean_se(agg[k])}**" for k in ["oracle", "zs"] + [f"ft{t}" for t in range(T)] + ["gap"]])
        md += [f"## {title}\n", md_table(header, rows), ""]

    header = ["env"] + [f"trial {t + 1} D0" for t in range(T)] + [f"trial {t + 1} online last-10 ret" for t in range(T)]
    rows = []
    for env in envs:
        e = env["env"]
        res = [[json.load(open(f)) for s in SOURCE_SEEDS if (f := s2s.recycle_dir(root, 0.5, t, e, s) / s2s.RESULT_FILE).exists()]
               for t in range(T)]
        rows.append([f"c{e}"] + [f"{np.mean([r['recycle']['d0_transitions'] for r in rt]):.0f}" if rt else "-" for rt in res]
                    + [mean_std([r["online"]["last10_return_mean"] for r in rt]) for rt in res])
    md += ["## Online fine-tuning (exploration-noise returns, reused transitions)\n", md_table(header, rows), ""]

    header = ["env", "screen gap ± SE", "confirm gap ± SE"] + [SHORT[n] for n in PARAMS]
    rows = [["nominal", "-", "-"] + [f"{v:.4g}" for v in base_params().values()]]
    for env in envs:
        c = screen_rows[env["env"]]
        rows.append([f"c{env['env']}", f"{c['screen_gap']:.1f} ± {c['screen_se']:.1f}", f"{c['confirm_gap']:.1f} ± {c['confirm_se']:.1f}"]
                    + [f"{c['params'][n]:.4g}" for n in PARAMS])
    md += ["## Selected envs (gap = source nominal − zero-shot, paired over the 5 source seeds)\n", md_table(header, rows), ""]
    sens = root / "screen_sensitivity.txt"
    if sens.exists():
        md += ["## Screening\n", "```", sens.read_text().rstrip(), "```", ""]
    text = "\n".join(md)
    (root / "report.md").write_text(text)
    print("\n" + text + f"\n[report] written to {root / 'report.md'}", flush=True)

    trial_dirs = [str(s2s.recycle_root(root, 0.5) / f"trial{t}") for t in range(T)]
    subprocess.run([sys.executable, "scripts/analysis/aggregate_online_tb.py", *trial_dirs], cwd=str(s2s.REPO_ROOT))


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "screen", "train", "finetune", "report"], default="all")
    ap.add_argument("--root", default="runs/td3/sim2sim_gap_search")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--cpu-workers", type=int, default=48, help="parallel CPU eval processes")
    ap.add_argument("--rel-ranges", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.25],
                    help="perturbation ladder, smallest first; the search stops at the first level with ≥ --min-envs envs")
    ap.add_argument("--candidates-per-level", type=int, default=1000)
    ap.add_argument("--search-seed", type=int, default=0)
    ap.add_argument("--batch-candidates", type=int, default=48)
    ap.add_argument("--nominal-episodes", type=int, default=100)
    ap.add_argument("--screen-episodes", type=int, default=30)
    ap.add_argument("--confirm-episodes", type=int, default=50)
    ap.add_argument("--report-episodes", type=int, default=50)
    ap.add_argument("--min-gap", type=float, default=15.0)
    ap.add_argument("--min-t", type=float, default=2.0)
    ap.add_argument("--min-envs", type=int, default=5)
    ap.add_argument("--max-envs", type=int, default=10)
    ap.add_argument("--max-search-hours", type=float, default=3.0, help="give up after this long")
    ap.add_argument("--oracle-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--total-timesteps", type=int, default=2_000_000)
    ap.add_argument("--train-jobs-per-gpu", type=int, default=15)
    ap.add_argument("--recycle-trials", type=int, default=4)
    ap.add_argument("--num-online-episodes", type=int, default=20)
    ap.add_argument("--online-eval-every", type=int, default=5)
    ap.add_argument("--online-eval-episodes", type=int, default=10)
    ap.add_argument("--finetune-jobs-per-gpu", type=int, default=15)
    cli = ap.parse_args()
    root = s2s._repo_path(cli.root)
    root.mkdir(parents=True, exist_ok=True)

    if cli.stage in ("all", "screen"):
        envs = screen(root, cli)
        if not envs or cli.stage == "screen":
            return
    envs = json.load(open(root / "envs.json"))["envs"]
    print(f"\nTensorBoard:\n  oracles     : .venv/bin/tensorboard --logdir {root / 'train'}\n"
          f"  fine-tuning : .venv/bin/tensorboard --logdir {s2s.recycle_root(root, 0.5)}  (per seed, live)\n"
          f"  aggregated  : .venv/bin/tensorboard --logdir {s2s.recycle_root(root, 0.5) / 'online_agg'}  (after report)\n", flush=True)
    if cli.stage == "all":
        th = threading.Thread(target=finetune, args=(root, envs, cli))
        th.start()
        train(root, envs, cli)
        th.join()
    elif cli.stage == "train":
        train(root, envs, cli)
    elif cli.stage == "finetune":
        finetune(root, envs, cli)
    if cli.stage in ("all", "report"):
        report(root, envs, cli)


if __name__ == "__main__":
    main()
