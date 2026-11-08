"""Overnight orchestrator: from-scratch training of the best recipe on the new sim.

- Trains every job of configs/ceiling/scratch_newsim.yaml, 5 slots per GPU (cap2000 on GPU 0, cap7000 on GPU 1).
- Convergence check at 5M, 7M, 9M, ... (once the checkpoint at that step exists): over the last 1M steps, compare the
  mean of [M-1M, M-0.5M) with [M-0.5M, M] for losses/q_loss, losses/actor_loss and charts/avg_episodic_return;
  converged if every relative change is < 10 %. Not converged -> keep training (+2M) and check again (bound 15M).
- A run also needs to have taken off (mean return over the last 0.5M > 50); one still at the floor at 9M is stopped.
- Converged: stop the run, SWA the actor over checkpoints in [0.6 M, M] (the best recipe averaged the last 40 % of its
  2M run), evaluate SWA + the raw checkpoint at M on the new sim (100 episodes, 10k cap, seeds 1000000+ = the final
  test of the fine-tuned policies). Evals run as separate CPU processes (at most 3 at once).
- When the first 5 seeds of a cap are evaluated, their mean SWA reach-10k is the cap's result. If it is below TARGET
  and no cap has reached TARGET, a cap + 5000 variant (5 seeds) is queued on that cap's GPU (cap = max cap so far + 5k).
- A seed whose SWA reaches 10k in < 10 % of episodes gets one extra seed of the same cap (<= 3 extras per cap).
State runs/ceiling/scratch_newsim/orchestrator_state.json, log orchestrator.log, table summary.md.

    python scripts/ceiling/scratch_newsim_orchestrator.py            # main loop (run from the repo venv)
    python scripts/ceiling/scratch_newsim_orchestrator.py --eval JOB STEP   # (internal) SWA + eval of one run

GPUs are hard-coded: cap2000 on GPU 0, cap7000 on GPU 1 (see main()). The original run (2026-11-03) is
written up in notes/scratch/experiments/2026-11-03_08-19_new-sim-from-scratch-convergence-cap7000.md.
"""
import json, os, signal, statistics, subprocess, sys, time
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "configs/ceiling/scratch_newsim.yaml"
ROOT = REPO / "runs/ceiling/scratch_newsim"
GEN = REPO / "configs/ceiling/generated/scratch_newsim"
SWA = REPO / "scripts/ceiling/swa.py"
PY = sys.executable
STATE = ROOT / "orchestrator_state.json"
SLOTS, MAX_EVALS = 5, 3
FIRST_CHECK, STEP_CHECK, MAX_STEPS = 5_000_000, 2_000_000, 15_000_000
TOL = 0.10
TARGET = 0.83          # fine-tuned recipe: 87.4 +- 4.5 % reach 10k over 5 seeds -> within one std err
MAX_EXTRA = 3
TAKEOFF = 50.0         # a run stuck at the ~17-return floor has flat losses too; it is not "converged", it never learned
STUCK_STOP = 9_000_000  # ... and is given up (and replaced by an extra seed) if still stuck at 9M
TAGS = ("losses/q_loss", "losses/actor_loss", "charts/avg_episodic_return")


def log(msg):
    with open(ROOT / "orchestrator.log", "a") as f:
        f.write(f"{time.strftime('%F %T')} {msg}\n")


def cap_of(job):
    return job.split("_s")[0][3:]


def seed_of(job):
    return int(job.rsplit("_s", 1)[1])


# ---------------------------------------------------------------- spec editing
def edit_spec(cap, seed):
    text = SPEC.read_text()
    header = "".join(l + "\n" for l in text.splitlines() if l.startswith("#"))
    spec = yaml.safe_load(text)
    names = [v["name"] for v in spec["variants"]]
    if f"cap{cap}" not in names:
        v = json.loads(json.dumps(spec["variants"][0]))
        v.update(name=f"cap{cap}", seeds=[]); v["sim"]["max_timesteps"] = int(cap)
        spec["variants"].append(v)
    for v in spec["variants"]:
        if v["name"] == f"cap{cap}":
            v["seeds"] = sorted(set(v.get("seeds", spec["seeds"])) | {seed})
    SPEC.write_text(header + yaml.safe_dump(spec, sort_keys=False))
    subprocess.run([PY, "-m", "scripts.ceiling.sweep", str(SPEC), "--prepare"], cwd=REPO, check=True,
                   stdout=subprocess.DEVNULL)


# ---------------------------------------------------------------- processes
def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def launch(st, job, gpu):
    out = open(ROOT / f"{job}.log", "a")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1")
    p = subprocess.Popen([PY, "-m", "scripts.rma.train_base_policy", "--args-file", str(GEN / f"{job}.yaml")],
                         cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    st["jobs"][job] = {"state": "training", "gpu": gpu, "pid": p.pid, "started": time.time(),
                       "next_check": FIRST_CHECK, "checks": []}
    log(f"launch {job} on GPU {gpu} pid {p.pid}")


def stop(pid):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except OSError:
            return
        time.sleep(15)


# ---------------------------------------------------------------- convergence
def ckpt_ready(run, step):
    c = run / f"checkpoint_{step}"
    return (c / "model.pth").exists() and ((c / "survival_eval/results.json").exists()
                                           or (run / f"checkpoint_{step + 100000}").exists())


def convergence(run, m):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    pts = {t: [] for t in TAGS}
    for f in sorted(run.glob("events.out.tfevents*")):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0}); ea.Reload()
        for t in TAGS:
            if t in ea.Tags()["scalars"]:
                pts[t] += [(s.step, s.value) for s in ea.Scalars(t)]
    res = {}
    for t, p in pts.items():
        a = [v for s, v in p if m - 1_000_000 <= s < m - 500_000]
        b = [v for s, v in p if m - 500_000 <= s <= m]
        if not a or not b:
            return False, {t: {"error": "no data"}}
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        res[t] = {"first_half": ma, "second_half": mb, "rel_change": abs(mb - ma) / max(abs(ma), 1e-8)}
    took_off = res["charts/avg_episodic_return"]["second_half"] > TAKEOFF
    res["took_off"] = took_off
    return took_off and all(r["rel_change"] < TOL for r in res.values() if isinstance(r, dict)), res


# ---------------------------------------------------------------- eval (subprocess)
def eval_one(job, m):
    run = ROOT / job
    lo = int(round(0.6 * m / 100000)) * 100000
    r = subprocess.run([PY, str(SWA), "runs/analysis/swa", str(run.relative_to(REPO)), str(lo), str(m)], cwd=REPO,
                       capture_output=True, text=True, check=True)
    swa_dir = os.path.relpath(r.stdout.strip().split()[0], REPO)
    raw_dir = str((run / f"checkpoint_{m}").relative_to(REPO))
    label = f"scratch_newsim__{job}__{m // 1000}k"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    subprocess.run([PY, "-m", "scripts.analysis.new_sim_eval", "--label", label, "--episodes", "100", "--horizon",
                    "10000", "--seed-base", "1000000", "--workers", "12", swa_dir, raw_dir], cwd=REPO, env=env,
                   check=True)
    res = json.loads((REPO / "runs/analysis/new_sim_eval" / label / "results.json").read_text())["results"]
    keys = ("ret", "len", "reach_5000", "reach_10000", "edge_hit_frac", "near_edge_frac", "dtop_median_cm", "vfwd", "pout")
    find = lambda d: next(v for k, v in res.items() if k.rstrip("/").endswith(d))["summary"]
    out = {"swa_dir": swa_dir, "swa_window": [lo, m], "label": label,
           "swa": {k: find(swa_dir)[k] for k in keys}, "raw": {k: find(raw_dir)[k] for k in keys}}
    (ROOT / f"eval_{job}.json").write_text(json.dumps(out, indent=2))


def start_eval(st, job, m):
    f = open(ROOT / f"eval_{job}.log", "a")
    p = subprocess.Popen([PY, str(Path(__file__).resolve()), "--eval", job, str(m)], cwd=REPO, stdout=f,
                         stderr=subprocess.STDOUT, start_new_session=True)
    st["jobs"][job].update(state="evaluating", eval_pid=p.pid)
    log(f"eval {job} @{m} pid {p.pid}")


# ---------------------------------------------------------------- summary
def write_summary(st):
    pct = lambda x: f"{100 * x[0]:.0f} ± {100 * x[1]:.0f} %"
    L = ["# scratch_newsim — the best recipe trained from scratch on the new sim", "",
         f"Updated {time.strftime('%F %T')}. Target: fine-tuned recipe 87.4 ± 4.5 % reach 10k (5 seeds; best single 94 ± 2 %).",
         "Eval: new sim, 100 episodes, 10k cap, seeds 1000000+. Per policy ± = std err over episodes.", "",
         "| job | state | step | converged | SWA reach 5k | SWA reach 10k | SWA return | raw ckpt reach 10k | edge hits |",
         "|---|---|---|---|---|---|---|---|---|"]
    for job, j in sorted(st["jobs"].items(), key=lambda kv: (int(cap_of(kv[0])), seed_of(kv[0]))):
        e, step = j.get("eval"), j.get("stop_step", f"(next check {j['next_check'] // 1000}k)")
        cells = (f"{pct(e['swa']['reach_5000'])} | {pct(e['swa']['reach_10000'])} | {e['swa']['ret'][0]:.0f} | "
                 f"{pct(e['raw']['reach_10000'])} | {100 * e['swa']['edge_hit_frac']:.1f} %") if e else "- | - | - | - | -"
        L.append(f"| {job} | {j['state']} | {step} | {j.get('converged', '-')} | {cells} |")
    L += ["", "## Per-cap means over seeds (SWA reach 10k, mean ± std err over seeds)", ""]
    for cap in sorted({cap_of(k) for k in st["jobs"]}, key=int):
        ev = [(k, j["eval"]["swa"]["reach_10000"][0]) for k, j in st["jobs"].items() if cap_of(k) == cap and j.get("eval")]
        for name, sel in (("first 5 seeds", [v for k, v in ev if seed_of(k) < 5]), ("all seeds", [v for _, v in ev])):
            if sel:
                se = statistics.stdev(sel) / len(sel) ** 0.5 if len(sel) > 1 else float("nan")
                L.append(f"- cap {cap}, {name}: {100 * statistics.fmean(sel):.1f} ± {100 * se:.1f} % (n={len(sel)}: "
                         + ", ".join(f"{100 * v:.0f}" for v in sel) + ")")
    L += ["", "## Convergence checks (relative change between the two halves of the last 1M steps; < 10 % = converged)", ""]
    for job, j in sorted(st["jobs"].items()):
        for c in j["checks"]:
            rel = ", ".join(f"{k.split('/')[-1]} {v['rel_change'] * 100:.1f} %" for k, v in c["res"].items() if isinstance(v, dict) and "rel_change" in v)
            rel += "" if c["res"].get("took_off", True) else ", return still at the floor"
            L.append(f"- {job} @ {c['step'] // 1000}k: {'converged' if c['ok'] else 'not converged'} ({rel})")
    (ROOT / "summary.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- main loop
def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATE.read_text()) if STATE.exists() else {
        "jobs": {}, "pending": [f"cap{c}_s{s}" for c in (2000, 7000) for s in range(5)],
        "caps": {"2000": {"gpu": 0}, "7000": {"gpu": 1}}}
    log("orchestrator start")
    while True:
        for gpu in (0, 1):
            busy = sum(j["gpu"] == gpu and j["state"] == "training" for j in st["jobs"].values())
            for job in [p for p in st["pending"] if st["caps"][cap_of(p)]["gpu"] == gpu]:
                if busy >= SLOTS:
                    break
                launch(st, job, gpu); st["pending"].remove(job); busy += 1
        for job, j in st["jobs"].items():
            if j["state"] == "evaluating" and not alive(j["eval_pid"]):
                f = ROOT / f"eval_{job}.json"
                if f.exists():
                    j["eval"] = json.loads(f.read_text()); j["state"] = "done"
                    log(f"{job} eval: SWA 5k {j['eval']['swa']['reach_5000']} 10k {j['eval']['swa']['reach_10000']}, "
                        f"raw 10k {j['eval']['raw']['reach_10000']}")
                else:
                    j["state"] = "eval_failed"; log(f"{job} eval failed (see eval_{job}.log)")
            if j["state"] == "converged_waiting" and sum(x["state"] == "evaluating" for x in st["jobs"].values()) < MAX_EVALS:
                start_eval(st, job, j["stop_step"])
            if j["state"] != "training":
                continue
            run, m, finished = ROOT / job, j["next_check"], not alive(j["pid"])
            if not ckpt_ready(run, m):
                if not finished:
                    continue
                steps = [int(x.name.split("_")[1]) for x in run.glob("checkpoint_*") if (x / "model.pth").exists()]
                if not steps or max(steps) < 1_000_000:
                    j["state"] = "failed"; log(f"{job} died early (see {job}.log)"); continue
                m = max(steps)
            ok, res = convergence(run, m)
            j["checks"].append({"step": m, "ok": ok, "res": res})
            log(f"{job} check @{m}: {'converged' if ok else 'not converged'} "
                + json.dumps({k.split('/')[-1]: (round(v.get('rel_change', -1), 3) if isinstance(v, dict) else v) for k, v in res.items()}))
            stuck = not res.get("took_off", True) and m >= STUCK_STOP
            if ok or finished or stuck or m + STEP_CHECK > MAX_STEPS:
                stop(j["pid"]); j.update(state="converged_waiting", stop_step=m, converged=ok)
            else:
                j["next_check"] = m + STEP_CHECK
        for cap, info in list(st["caps"].items()):
            for k, j in list(st["jobs"].items()):
                if (cap_of(k) == cap and j.get("eval") and j["eval"]["swa"]["reach_10000"][0] < 0.10
                        and not j.get("replaced") and info.get("extra", 0) < MAX_EXTRA):
                    seed = 10 + seed_of(k)
                    while f"cap{cap}_s{seed}" in st["jobs"] or f"cap{cap}_s{seed}" in st["pending"]:
                        seed += 10
                    edit_spec(cap, seed); st["pending"].append(f"cap{cap}_s{seed}")
                    info["extra"] = info.get("extra", 0) + 1; j["replaced"] = True
                    log(f"{k} weak (SWA reach 10k < 10 %) -> extra seed cap{cap}_s{seed}")
            first5 = [st["jobs"].get(f"cap{cap}_s{s}", {}).get("eval") for s in range(5)]
            if all(first5) and "result" not in info:
                info["result"] = statistics.fmean(e["swa"]["reach_10000"][0] for e in first5)
                log(f"cap {cap}: first-5 SWA reach-10k mean {info['result']:.3f} (target {TARGET})")
                nxt = str(max(int(c) for c in st["caps"]) + 5000)
                if info["result"] < TARGET and not any(c.get("result", 0) >= TARGET for c in st["caps"].values()):
                    st["caps"][nxt] = {"gpu": info["gpu"]}
                    for s in range(5):
                        edit_spec(nxt, s); st["pending"].append(f"cap{nxt}_s{s}")
                    log(f"cap {cap} below target -> queue cap {nxt} (5 seeds) on GPU {info['gpu']}")
        tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(st, indent=2)); tmp.replace(STATE)
        write_summary(st)
        if not st["pending"] and all(j["state"] in ("done", "failed", "eval_failed") for j in st["jobs"].values()):
            log("all done"); break
        time.sleep(300)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        eval_one(sys.argv[2], int(sys.argv[3]))
    else:
        main()
