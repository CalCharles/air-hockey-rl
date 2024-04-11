# if error, exit
set -e

# This should be called from the root of the project
python scripts/train.py --cfg configs/baseline_configs/puck_vel.yaml
python scripts/train.py --cfg configs/baseline_configs/move_block.yaml
python scripts/train.py --cfg configs/baseline_configs/puck_height.yaml
python scripts/train.py --cfg configs/baseline_configs/strike.yaml
python scripts/train.py --cfg configs/baseline_configs/strike_crowd.yaml
python scripts/train.py --cfg configs/baseline_configs/reach.yaml
python scripts/train.py --cfg configs/baseline_configs/reach_vel.yaml
python scripts/train.py --cfg configs/baseline_configs/puck_catch.yaml
