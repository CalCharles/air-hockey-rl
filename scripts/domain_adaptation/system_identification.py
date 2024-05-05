import copy
import numpy as np
import yaml
import cma # TODO: using https://cma-es.github.io/ for CMA-ES
from scripts.domain_adaptation.encode_env_params import assign_values, extract_values
from airhockey import AirHockeyEnv

def sample_trajectories(data, num_samples=20, traj_length = 20):
    trajectories = list()
    for s in num_samples:
        traj = data[np.random.randint(len(data))]
        traj_start = np.random.randint(0, len(traj.states) - 20)
        traj.states, traj.actions, traj.dones = traj.states[traj_start:traj_start + 20], traj.actions[traj_start:traj_start + 20], traj.dones[traj_start:traj_start + 20]
        trajectories.append(traj)
    return trajectories

def get_value(param_vector, param_names, base_config, trajectories):
    assign_values(param_vector, param_names, base_config)
    eval_env = AirHockeyEnv(new_config)
    
    evaluated_states = list()
    for traj in trajectories:
        eval_env.reset(traj.states[0]) # TODO: need to be able to reset from state
        for act in traj.actions:
            evaluated_states.append(eval_env.step(act)[0])
    
    return compare_trajectories(evaluated_states, sum([t.states for t in trajectories], start=list()))

def compare_trajectories(comp_type, a_traj, b_traj):
    # TODO: Dynamic time warping for trajectory comparison
    if comp_type == "l2":
        return np.linalg.norm(np.array(a_traj) - np.array(b_traj))

def system_id(args, data, air_hockey_cfg, param_names):
    air_hockey_params = air_hockey_cfg['air_hockey']
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']
    
    if 'sac' == air_hockey_cfg['algorithm']:
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = True
        else:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = False
    else:
        air_hockey_cfg['air_hockey']['return_goal_obs'] = False
    
    air_hockey_params_cp = air_hockey_params.copy()
    air_hockey_params_cp['seed'] = 42
    air_hockey_params_cp['max_timesteps'] = 200


    initial_params = extract_value(param_names, air_hockey_params_cp)
    opts = cma.CMAOptions()
    opts['N'] = args.population
    es = cma.CMAEvolutionStrategy(initial_params, args.initial_sigma, opts = opts)
    while not es.stop():
        solutions = es.ask()
        trajectories = sample_trajectories(data, num_samples=args.num_samples, traj_length=args.traj_length)
        values = [get_value(s, param_names, air_hockey_params_cp, trajectories) for s in solutions]            
        es.tell(solutions, values)

    return es.result[0], es.result[int(args.population // 2)]    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--dataset_pth', type=str, default=None, help='Path to the dataset file.')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)
    param_names = list(air_hockey_cfg["simulator_params"].keys())
    param_names.sort()

    # TODO: write a loader for the dataset, which should load into a list of dicts with three keys: states, actions, dones
    # sorting is: trajectory->key->data
    data = load_dataset(args.dataset_pth) 
    best_params, mean_params = system_id(args, data, air_hockey_cfg, param_names)