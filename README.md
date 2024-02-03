# 2D Air Hockey Simulator (Box2D Engine)

This contains an air hockey simulation environment powered by Box2D. It is fast (C++ back-end), capable of self-play, 1v1 play, and easy goal-conditioned reinforcement learning, resulting in a rich testbed for various algorithms.


1v1 Play             |  Goal-Conditioned RL
:-------------------------:|:-------------------------:
![](assets/player_vs_ai.gif)  |  ![](assets/goal_conditioned.gif)

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
