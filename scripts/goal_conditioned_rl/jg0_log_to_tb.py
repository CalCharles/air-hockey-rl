"""Convert a train_jg0.py train.log (JSON lines) into TensorBoard scalars, x axis = env steps.

    .venv/bin/python -m scripts.goal_conditioned_rl.jg0_log_to_tb runs/crl/jg0_best_s0
    .venv/bin/tensorboard --logdir runs/crl/jg0_best_s0/tb
"""
import json
import shutil
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

EVAL = {"hit10": "within_10cm", "hit5": "within_5cm", "stop": "stopped_on_target", "dropped": "dropped",
        "ep_steps": "episode_steps", "min_dist": "min_dist_m", "frac_within": "frac_time_within",
        "launch0_hits": "launch0_hits_of_35"}
TRAIN = {"critic_loss": "loss/critic", "actor_loss": "loss/actor", "alpha": "loss/alpha", "entropy": "loss/entropy",
         "logsumexp": "loss/logsumexp", "q": "loss/q", "skipped_updates": "loss/skipped_updates",
         "collect_hit10": "collect/within_10cm", "collect_stop": "collect/stopped_on_target",
         "collect_min_dist": "collect/min_dist_m", "drop_frac": "collect/dropped", "ep_len": "collect/episode_len",
         "episodes": "collect/episodes_per_round", "t_collect": "time/collect_s", "t_train": "time/train_s",
         "hours": "time/hours"}


def main(run):
    run = Path(run)
    out = run / "tb"
    shutil.rmtree(out, ignore_errors=True)
    w = SummaryWriter(str(out))
    n_eval = n_train = 0
    for line in (run / "train.log").read_text().splitlines():
        try:
            d = json.loads(line[line.index("{"):])
        except ValueError:
            continue
        step = d.get("steps")
        if step is None or d.get("done"):
            continue
        if d.get("eval"):
            for k, name in EVAL.items():
                if isinstance(d.get(k), (int, float)):
                    w.add_scalar(f"eval/{name}", d[k], step)
            n_eval += 1
        elif "round" in d and "critic_loss" in d:
            for k, name in TRAIN.items():
                if isinstance(d.get(k), (int, float)):
                    w.add_scalar(name, d[k], step)
            n_train += 1
    w.close()
    print(f"{out}: {n_eval} evals, {n_train} rounds")


if __name__ == "__main__":
    main(sys.argv[1])
