"""Build each seed's init bundle: SWA actor in both actor slots, critics + targets from the seed's last raw checkpoint,
and the per-seed trainer args (actor frozen, no intermediate checkpoints, stop-file early stop)."""
import torch, yaml
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "runs/ceiling/swa_critic_fit"; GEN = REPO / "configs/ceiling/generated/swa_critic_fit"
RAW = {0: 5_000_000, 1: 7_000_000, 2: 13_000_000, 3: 9_000_000, 4: 7_000_000}
for s, step in RAW.items():
    swa_dir = REPO / f"latest_models/ceiling/newsim_scratch_cap7000_swa/seed{s}"
    raw = REPO / f"runs/ceiling/scratch_newsim/cap7000_s{s}/checkpoint_{step}/training_state.pth"
    st = torch.load(raw, map_location="cpu", weights_only=False)
    swa = torch.load(swa_dir / "model.pth", map_location="cpu")
    assert set(swa) == set(st["actor"]), "actor key mismatch"
    st["actor"] = {k: v.clone() for k, v in swa.items()}
    st["actor_target"] = {k: v.clone() for k, v in swa.items()}
    init = OUT / "init" / f"cap7000_s{s}_init.pth"; torch.save(st, init)
    a = yaml.safe_load(open(REPO / f"configs/ceiling/generated/scratch_newsim/cap7000_s{s}.yaml"))
    job = f"cap7000_s{s}"
    a.update(
        run_name=f"swa_critic_fit_{job}", log_parent_dir=f"runs/ceiling/swa_critic_fit/{job}",
        model_path=str(init.relative_to(REPO)), full_checkpoint_load="weights_only",
        actor_learning_starts=10**12,                    # actor (and so actor_target) never updated
        exploration_pre_learning_action_source="policy",  # buffer fills with SWA-actor data only
        exploration_primitive_chance_start=0.05, exploration_primitive_chance_anneal_steps=0,  # late-training exploration
        checkpoint_interval=10**12,                      # no intermediate checkpoints
        total_timesteps=6_000_000,                       # upper bound; orchestrator stops at convergence
        stop_file=f"runs/ceiling/swa_critic_fit/{job}.STOP",
    )
    yaml.safe_dump(a, open(GEN / f"{job}.yaml", "w"), sort_keys=False)
    print(job, "init", init.name, "from", raw.parent.name)
