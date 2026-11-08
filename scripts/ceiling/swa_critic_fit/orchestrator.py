"""Critic fit for the frozen cap-7000 SWA actors: launch 5 trainers (3 on GPU 0, 2 on GPU 1), check convergence of
losses/q_loss and charts/avg_episodic_return (< 10 % change between the halves of the last 1M steps) at 2M, 3M, ...,
touch the run's stop file when converged (trainer saves its final checkpoint and exits; 6M upper bound), then run
post.py (actor identity check, Q-vs-return check, copy into latest_models)."""
import json, os, statistics, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; ROOT = REPO / "runs/ceiling/swa_critic_fit"
GEN = REPO / "configs/ceiling/generated/swa_critic_fit"; PY = sys.executable
JOBS = {f"cap7000_s{s}": (0 if s < 3 else 1) for s in range(5)}
FIRST, STEP, MAX = 2_000_000, 1_000_000, 6_000_000; TOL = 0.10
TAGS = ("losses/q_loss",)  # agreed criterion: critic loss only (actor frozen, so the return is exploration noise)
LOG_TAGS = ("charts/avg_episodic_return",)
STATE = ROOT / "orchestrator_state.json"

def log(m):
    with open(ROOT / "orchestrator.log", "a") as f: f.write(f"{time.strftime('%F %T')} {m}\n")

def alive(pid):
    try:
        os.kill(pid, 0); return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError: return False

def scalars(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    pts = {t: [] for t in TAGS}
    for f in sorted(run.glob("events.out.tfevents*")):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0}); ea.Reload()
        for t in TAGS:
            if t in ea.Tags()["scalars"]: pts[t] += [(s.step, s.value) for s in ea.Scalars(t)]
    return pts

def convergence(pts, m):
    res = {}
    for t, p in pts.items():
        a = [v for s, v in p if m - 1_000_000 <= s < m - 500_000]; b = [v for s, v in p if m - 500_000 <= s <= m]
        if not a or not b: return False, {t: "no data"}
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        res[t] = {"first_half": ma, "second_half": mb, "rel_change": abs(mb - ma) / max(abs(ma), 1e-8)}
    return all(r["rel_change"] < TOL for r in res.values()), res

st = json.loads(STATE.read_text()) if STATE.exists() else {"jobs": {}}
for job, gpu in JOBS.items():
    if job in st["jobs"]: continue
    out = open(ROOT / f"{job}.log", "a")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1")
    p = subprocess.Popen([PY, "-m", "scripts.rma.train_base_policy", "--args-file", str(GEN / f"{job}.yaml")],
                         cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    st["jobs"][job] = {"state": "training", "gpu": gpu, "pid": p.pid, "next_check": FIRST, "checks": []}
    log(f"launch {job} on GPU {gpu} pid {p.pid}")
STATE.write_text(json.dumps(st, indent=1))

while True:
    for job, j in st["jobs"].items():
        run = ROOT / job
        if j["state"] == "training":
            if not alive(j["pid"]):
                ok = (run / "training_state.pth").exists()
                j["state"] = "trained" if ok else "failed"; log(f"{job} trainer exited ({j['state']})")
                continue
            if j["next_check"] <= MAX - STEP:
                pts = scalars(run)
                last = max((s for s, _ in pts["losses/q_loss"]), default=0)
                if last >= j["next_check"]:
                    conv, res = convergence(pts, j["next_check"])
                    j["checks"].append({"step": j["next_check"], "converged": conv, "res": res})
                    log(f"{job} check @{j['next_check']}: converged={conv} " + json.dumps(res))
                    if conv:
                        (ROOT / f"{job}.STOP").touch(); j["stop_at"] = j["next_check"]; log(f"{job} stop file written")
                    j["next_check"] += STEP
        elif j["state"] == "trained":
            j["state"] = "post"; log(f"{job} post-processing")
            r = subprocess.run([PY, str(Path(__file__).with_name("post.py")), job], cwd=REPO, capture_output=True, text=True)
            (ROOT / f"{job}.post.log").write_text(r.stdout + r.stderr)
            j["state"] = "done" if r.returncode == 0 else "post_failed"; log(f"{job} post exit {r.returncode}")
    STATE.write_text(json.dumps(st, indent=1))
    if all(j["state"] in ("done", "failed", "post_failed") for j in st["jobs"].values()):
        log("all done"); break
    time.sleep(120)
