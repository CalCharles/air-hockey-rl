python scripts/train.py --cfg scripts/domain_adaptation/optimal_tasks/puck_goal_pos.yaml --device cuda:0 &
python scripts/train.py --cfg scripts/domain_adaptation/optimal_tasks/puck_height.yaml --device cuda:1 &
python scripts/train.py --cfg scripts/domain_adaptation/optimal_tasks/puck_juggle.yaml --device cuda:2 &
python scripts/train.py --cfg scripts/domain_adaptation/optimal_tasks/puck_touch.yaml --device cuda:3 &
python scripts/train.py --cfg scripts/domain_adaptation/optimal_tasks/puck_vel.yaml --device cuda:4 &
python scripts/train.py --cfg scripts/domain_adaptation/optimal_tasks/strike.yaml --device cuda:5