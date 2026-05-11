# Using the Box2D simulator for your own training loop

This page is for someone bringing their **own RL algorithm** and wanting to train against the same Box2D juggling environment that the in-repo TD3 pipeline uses, with the same physics / observations / rewards. If you are running the in-repo TD3 trainer, use [`td3-configs.md`](td3-configs.md) instead.

---

## TL;DR — config file to use

| Item | Path |
|---|---|
| **Box2D sim config** (canonical) | `scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params_hist2.yaml` |
| Active TD3 args (reference only) | `scripts/smooth_policy/amp_history/configs/td3/td3_recommended_top50_hist2.yaml` |
| Reference training script | `scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py` (look at `make_env` and the `__main__` block) |

`sysid_best_params_hist2.yaml` is the **only file your training loop needs to load.** It contains the env, task, physics, sim-to-real-gap features (occlusion / observation delay / force attenuation), spawn distribution, and reward weights — everything the env constructor reads. The TD3 args YAML next to it is algorithm-specific and is not consumed by the env.

This config matches `latest_models/canonical/hist2_motion0/config.yaml` at the high level (see [`sim-env-configs.md`](sim-env-configs.md) for the small restitution / per-collision-randomization deltas).

> Older hist3 / hist4 / hist5 variants used by historical ablations have been moved to `configs/new_juggle/legacy/`. Don't use them for new work.

---

## Minimal usage

Two lines of setup:

```python
import yaml
from airhockey import AirHockeyEnv

with open("scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params_hist2.yaml") as f:
    cfg = yaml.safe_load(f)

env = AirHockeyEnv(cfg["air_hockey"])
```

`AirHockeyEnv` ([`airhockey/__init__.py`](../../../airhockey/__init__.py)) is a factory: it dispatches on `cfg["air_hockey"]["task"]` (here, `puck_juggle_upper_half_reward`) and returns a `gymnasium.Env`-compatible object. The env exposes the standard API:

```python
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

- **Observation** (`obs_type: history`): 30-dim vector — see [`environments/observation-action-spaces.md`](../environments/observation-action-spaces.md). The actor in the TD3 pipeline additionally concatenates the previous action (32 dims), but that's an *algorithm* choice your loop may or may not make.
- **Action**: `Box([-1, 1], shape=(2,))` normalised paddle-displacement target. One step ≈ `[0.26 m, 0.12 m]` in (x, y) at full magnitude. Fed to a PID controller inside Box2D.
- **Reward**: dense puck-juggling reward defined by `task: puck_juggle_upper_half_reward`. See [`reward-shaping.md`](reward-shaping.md). The TD3 pipeline splits this into task / motion heads; if you don't want that split, just use `info["task_reward"]` (or the scalar `reward`) directly.
- **Termination**: terminates on enemy goal, puck hit-bottom, or puck-pass-paddle (per the config flags).

The td3 reference script's `make_env` (`td3_training.py:643`) shows the exact same two-line pattern, plus a per-thunk seed override:

```python
def make_env(env_id):
    def _thunk():
        config["air_hockey"]["seed"] = random.randint(0, int(1e8))
        return AirHockeyEnv(config["air_hockey"])
    return _thunk
```

If you want vector envs, wrap N copies in `gym.vector.AsyncVectorEnv([make_env(i) for i in range(N)])` exactly as the reference script does (`td3_training.py:789`).

---

## What `sysid_best_params_hist2.yaml` bakes in

You don't need to set any of these yourself — they're already in the YAML — but it's worth knowing what's enabled, because they all affect what an external policy will see:

- **System-ID physics** (matches the real robot): `gravity: -0.661`, `puck_damping: 0.178`, `paddle_density: 3000`, `pid_kp: 9000`, `pid_kd: 50`. See [`environments/real-world/puck-system-id.md`](../environments/real-world/puck-system-id.md) and [`teleop-system-id.md`](../environments/real-world/teleop-system-id.md).
- **`hist_len: 2`** — the PID target is low-pass-filtered over 2 timesteps inside `_filter_update` ([`airhockey/sims/airhockey_box2d.py`](../../../airhockey/sims/airhockey_box2d.py)). This is also the env default.
- **Sim-to-real-gap features (on)**: puck position noise (σ=0.01 m), random occlusions (with near-paddle boost), observation delay (25 ms ± 25%), action force attenuation (30% chance of 25–75% scaling).
- **Per-collision randomization (on)**: paddle-puck strength + direction jitter, wall direction jitter — see the config's `enable_paddle_puck_*` and `enable_wall_direction_*` blocks.
- **Episode**: `max_timesteps: 250`, `puck_juggle_upper_half_reward` task, near-paddle puck spawn 15% of resets.

If you want to remove a feature for a controlled comparison, edit a copy of the YAML — don't add overrides scattered through your training code.

---

## Where to look for more

- Obs / action spaces in detail: [`environments/observation-action-spaces.md`](../environments/observation-action-spaces.md)
- Box2D internals (workspace clipping, delay model, coordinate frames): [`environments/box2d/simulator-essentials.md`](../environments/box2d/simulator-essentials.md)
- Reward shaping: [`reward-shaping.md`](reward-shaping.md)
- Sim config field reference: [`sim-env-configs.md`](sim-env-configs.md)
- Reference training loop end-to-end: `scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py`
