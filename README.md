# 2D Air Hockey Simulator (Box2D Engine)

## Requirements:
- `pip install Box2D`
- `pip install opencv-python`

## Optional
- `pip install stable-baselines3` (required for `play_trained_agent.py`, `sb_trainer.py`, `sb_eval.py`)

## How to Run
Most of the files use a configuration file (--cfg cmd argument), but is defaulted to one from `configs/`. Please see there to tune parameters for various scripts.
#### What the files do
- `airhockey2d.py`: base gym environment for air hockey
- `render.py`: renders the air hockey environment
- `demonstrate.py`: user plays a self-play air hockey environment using keyboard
- `sb_trainer.py`: trains an agent using self-play via stable-baselines3 PPO.
- `sb_eval.py`: run after training, this shows training evaluation plots and plays a live rendering of the trained agent playing via self-play.
- `play_trained_agent`: run after training, you can play against the trained agent
