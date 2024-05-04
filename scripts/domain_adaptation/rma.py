from airhockey import AirHockeyEnv
from scripts.domain_adaptation.encode_env_params import assign_values, extract_values

def train(policy, env, z, num_null_iters):
    raise NotImplementedError # TODO: write this function, preferably using something already written

def rollout(policy, env, z, num_frames):
    raise NotImplementedError # TODO: write this function, just rolls out a policy

def train_identifier(identifier, trajectories, param_vector):
    raise NotImplementedError # TODO: write this function, trains the identifier to predict the param_vector from a fixed length sample of trajectories

def train_encoder_decoder(encode, decode, param_vector):
    raise NotImplementedError # TODO: write this function, leans an encoder/decoder of the param vector


class RapidMotorAdaptation():
    def __init__(self, args, param_names, base_config, param_tolerances):
        self.policy = init_policy() # TODO initialize a policy that takes in param_vector as an input
        self.encode = init_encode() # TODO: initialize an encoder for the param vector
        self.decode = init_decode() # TODO: initialize a decoder for the param vector
        self.identifier = init_identifier() # TODO: A model that identifies the param vector from a few trajectories 
        self.base_config = base_config
        self.num_train_iters = 0
        self.num_null_iters = 0 
        # TODO: figure out good tolerances for the parameters, and use as a dict of
        # name: [lower, upper]
        self.param_names = param_names
        self.param_tolerances = param_tolerances
    
    def generate_null_rollouts(self, param_vector):
        new_config = assign_values(param_vector, param_names, self.base_config)
        env = AirHockeyEnv(new_config)


    def train_environment(self, param_vector):
        new_config = assign_values(param_vector, param_names, self.base_config)
        env = AirHockeyEnv(new_config)
        # TODO: hook up to a train function
        # TODO: null training trains the policy  to be able to handle unknown parameters
        train(self.policy, env, np.zeros(param_vector.shape), self.num_null_iters)
        # TODO: then train with the (encoded) param vector
        trajectories = train(self.policy, env, self.encode(param_vector), self.num_train_iters)
        # TODO: train the identifier with trajectories to recover the param vector
        train_identifier(self.identifier, trajectories, param_vector)
        # TODO: it's not obvious how to train the encoder and decoder, so we don't need to do this
        train_encoder_decoder(self.encode, self.decode, param_vector)
        return trajectories
    
    def identify(self, trajectories):
        return self.decode(self.identifier(trajectories))

    def sample_param(self):
        param_vector = list()
        for name in self.param_names:
            if type(self.base_config["simulator_params"][name]) == list:
                for i in range(len(self.base_config["simulator_params"][name])):
                    param_vector.append(self.param_tolerances[name][i])
            else:
                param_vector.append(self.param_tolerances[name])
        return np.array(param_vector)

def train_RMA(args, data, air_hockey_cfg, param_names):
    # TODO: define param tolerances, probably inside the config?
    rma = RapidMotorAdaptation(args, param_names, air_hockey_cfg, param_tolerances)
    for i in range(args.num_env_samples):
        # train the RMA policy on one environment
        param_vector = rma.sample_param()
        rma.train_environment(param_vector)

        # evaluate the identifier
        param_vector = rma.sample_param()
        new_config = assign_values(param_vector, param_names, self.base_config)
        env = AirHockeyEnv(new_config)
        trajectories = rollout(rma.policy, env, np.zeros(param_vector.shape), self.num_eval_frames)
        predicted_param = rma.identify(trajectories)

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
    best_params, mean_params = train_RMA(args, data, air_hockey_cfg, param_names)