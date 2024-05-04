class GroundedActionTransformation():
    def __init__(self, args, sim_env, real_env):
        self.sim_env = sim_env
        self.real_env = real_env
        self.policy = init_policy()
        self.forward_model = init_forward_model()
        self.inverse_model = init_inverse_model()
        # TODO: need to define the buffers, check with Michael Munje about how we are using buffers
        # for the current RL training code
        self.real_buffer = init_buffer()
        self.sim_buffer = init_buffer()

        # before we have trained anything, don't use the grounded transform
        self.first = True

    def grounded_transform(self, state, action):
        if self.first: return action
        return self.inverse_model(state, self.forward_model(state, action))
    
    def add_data(self, trajectories):
        # TODO: pretty sure you can't do this
        self.real_buffer.add(trajectories)

    def rollout_real(self, num_frames):
        obs = self.real_env.get_state()
        for i in range(num_frames):
            act = self.policy.act(obs)
            obs, rew, term, trunc, info = self.real_env.step()
            self.real_buffer.add((obs, act, rew, term, trunc, info))
        # TODO: might need to make this actually work
    
    def train_sim(self, num_iters):
        # TODO: try to utilize the same train function as other components
        # TODO: train should automatically add to self.sim_buffer, so we don't need to keep
        # the trajectories
        trajectories = train(self.policy, self.sim_buffer, self.grounded_transform, num_iters)

    def train_forward(self, num_iters):
        for i in range(num_iters):
            data = self.real_buffer.sample(self.inverse_batch_size)
            pred_next_state = self.forward_model(data.state, data.action)
            # TODO: define these functions generally so we can change them later
            loss = compute_loss(pred_next_state, data.next_state)
            self.forward_optimizer.step(loss.mean())

    def train_inverse(self, num_iters):
        for i in range(num_iters):
            data = self.sim_buffer.sample(self.inverse_batch_size)
            pred_action = self.inverse_model(data.state, data.next_state)
            # TODO: define these functions generally so we can change them later
            loss = compute_loss(pred_action, data.action)
            self.inverse_optimizer.step(loss.mean())

def train_GAT(args, data, sim_air_hockey_cfg, real_air_hockey_cfg):
    sim_env = AirHockeyEnv(sim_air_hockey_cfg)
    real_env = AirHockeyEnv(real_air_hockey_cfg)

    gat = GroundedActionTransformation(args, sim_env, real_env)
    gat.add_data(data)
    gat.train_sim(args.initial_rl_training)
    gat.train_inverse(args.initial_inverse_training)
    gat.train_forward(args.initial_forward_training)
    for i in range(args.num_real_sim_iters):
        gat.train_sim(args.rl_iters)
        gat.rollout_real(args.num_real_steps)
        gat.train_inverse(args.inverse_iters)
        gat.train_forward(args.forward_iters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--sim-cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--real-cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--dataset_pth', type=str, default=None, help='Path to the dataset file.')
    args = parser.parse_args()

    with open(args.sim_cfg, 'r') as f:
        sim_air_hockey_cfg = yaml.safe_load(f)

    with open(args.real_cfg, 'r') as f:
        real_air_hockey_cfg = yaml.safe_load(f)

    # TODO: write a loader for the dataset, which should load into a list of dicts with three keys: states, actions, dones
    # sorting is: trajectory->key->data
    data = load_dataset(args.dataset_pth) 
    best_params, mean_params = train_GAT(args, data, sim_air_hockey_cfg, real_air_hockey_cfg)