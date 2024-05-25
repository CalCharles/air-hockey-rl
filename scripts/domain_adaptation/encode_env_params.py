import copy

# TODO: modify so not all the params have to be simulator params
def extract_value(param_names, config):
    param_vector = list()
    for name in param_names:
        if type(config["simulator_params"][name]) == list:
            for i in range(len(config["simulator_params"][name])):
                param_vector.append(config["simulator_params"][name][i])
        else:
            param_vector.append(config["simulator_params"][name])
    return param_vector

def assign_values(param_vector, param_names, config):
    new_config = copy.deepcopy(config)
    at = 0
    for name in param_names:
        if type(config["simulator_params"][name]) == list:
            for i in range(len(config["simulator_params"][name])):
                new_config["simulator_params"][name][i] = param_vector[i + at] 
                at += 1
        else:
            # TODO: handle booleans and ints gracefully
            new_config["simulator_params"][name] = param_vector[at]
            at += 1
    return new_config
