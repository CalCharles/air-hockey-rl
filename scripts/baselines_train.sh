# if error, exit
set -e

# This should be called from the root of the project
python -m scripts.train --cfg configs/baseline_configs/puck_vel.yaml
python -m scripts.train --cfg configs/baseline_configs/move_block.yaml
python -m scripts.train --cfg configs/baseline_configs/puck_height.yaml
python -m scripts.train --cfg configs/baseline_configs/strike.yaml
python -m scripts.train --cfg configs/baseline_configs/strike_crowd.yaml
python -m scripts.train --cfg configs/baseline_configs/reach.yaml
python -m scripts.train --cfg configs/baseline_configs/reach_vel.yaml
python -m scripts.train --cfg configs/baseline_configs/puck_catch.yaml
