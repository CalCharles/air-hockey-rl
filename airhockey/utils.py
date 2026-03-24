import numpy as np
import copy
from types import SimpleNamespace
import collections.abc
from .observation_homography import apply_plane_homography_xy

def dict_to_namespace(d):
    if isinstance(d, dict):
        namespace = SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return namespace
    elif isinstance(d, collections.abc.MutableMapping):
        namespace = SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return namespace
    else:
        return d

def _maybe_warp_xy(x, y, position_homography):
    if position_homography is None:
        return float(x), float(y)
    return apply_plane_homography_xy(x, y, position_homography)


def _warp_history_xy(history_array, position_homography):
    if position_homography is None:
        return np.array(history_array)
    history = np.array(history_array, dtype=float, copy=True)
    if history.size == 0:
        return history
    if history.ndim == 1:
        if history.size % 3 != 0:
            return history
        history = history.reshape(-1, 3)
    if history.shape[1] < 2:
        return history
    for idx in range(history.shape[0]):
        history[idx, 0], history[idx, 1] = _maybe_warp_xy(
            history[idx, 0], history[idx, 1], position_homography
        )
    return history


def get_observation_by_type(state_info, obs_type='vel', **kwargs):
    # TODO: check that this code is used properly
    position_homography = kwargs.get("position_homography")
    ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
    ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
    ego_paddle_x_pos, ego_paddle_y_pos = _maybe_warp_xy(
        ego_paddle_x_pos, ego_paddle_y_pos, position_homography
    )
    ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
    ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
    if obs_type == "paddle":
        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel])
        return obs
    elif obs_type == "negative_regions_paddle":
        reward_regions_states = [nrr for nrr in state_info['negative_regions']]

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel])
        return np.concatenate([obs] + reward_regions_states)
    elif obs_type == 'vel':
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_pos, puck_y_pos = _maybe_warp_xy(puck_x_pos, puck_y_pos, position_homography)
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1] 
        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return obs
    elif obs_type == 'pos':
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_pos, puck_y_pos = _maybe_warp_xy(puck_x_pos, puck_y_pos, position_homography)
        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, puck_x_pos, puck_y_pos])
        return obs
    elif obs_type == "history":        
        puck_hist_array = np.array(kwargs["puck_history"][-5:])
        paddle_hist_array = np.array(kwargs["paddle_history"][-5:])
        if puck_hist_array.size == 0:
            # Add 15 elements, every 3rd element is a 1, rest are 0
            puck_hist_array = np.zeros(15)
            puck_hist_array[2::3] = 1
        puck_hist_array = _warp_history_xy(puck_hist_array, position_homography)
        paddle_hist_array = _warp_history_xy(paddle_hist_array, position_homography)

        obs = np.concatenate([paddle_hist_array.flatten(), puck_hist_array.flatten()])
        return obs
    elif obs_type == "single_block_vel":
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_pos, puck_y_pos = _maybe_warp_xy(puck_x_pos, puck_y_pos, position_homography)
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       

        block_x_pos = state_info['blocks'][0]['current_position'][0]
        block_y_pos = state_info['blocks'][0]['current_position'][1]
        block_initial_x_pos = state_info['blocks'][0]['initial_position'][0]
        block_initial_y_pos = state_info['blocks'][0]['initial_position'][1]
        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel, block_x_pos, block_y_pos, block_initial_x_pos, block_initial_y_pos])
        return obs
    elif obs_type == "single_block_history":
        puck_hist = _warp_history_xy(kwargs["puck_history"][-5:], position_homography).flatten().tolist()

        block_x_pos = state_info['blocks'][0]['current_position'][0]
        block_y_pos = state_info['blocks'][0]['current_position'][1]
        block_initial_x_pos = state_info['blocks'][0]['initial_position'][0]
        block_initial_y_pos = state_info['blocks'][0]['initial_position'][1]
        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, block_x_pos, block_y_pos, block_initial_x_pos, block_initial_y_pos] + puck_hist)
        return obs
    elif obs_type == "many_blocks_vel":
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_pos, puck_y_pos = _maybe_warp_xy(puck_x_pos, puck_y_pos, position_homography)
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       

        blocks = state_info['blocks']
        block_initial_positions = []
        for block in blocks:
            block_initial_positions.append(block['initial_position'])
        block_initial_positions = np.array(block_initial_positions).flatten()
        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel] + block_initial_positions.tolist())
        return obs
    elif obs_type == "many_blocks_history":
        puck_hist = _warp_history_xy(kwargs["puck_history"][-5:], position_homography).flatten().tolist()

        blocks = state_info['blocks']
        block_initial_positions = []
        for block in blocks:
            block_initial_positions.append(block['initial_position'])
        block_initial_positions = np.array(block_initial_positions).flatten()
        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel] + block_initial_positions.tolist() + puck_hist)
        return obs
    elif obs_type == "negative_regions_puck_vel":
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_pos, puck_y_pos = _maybe_warp_xy(puck_x_pos, puck_y_pos, position_homography)
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]       
        reward_regions_states = [nrr for nrr in state_info['negative_regions']]

        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return np.concatenate([obs] + reward_regions_states)
    elif obs_type == "negative_regions_puck_history":
        puck_hist = _warp_history_xy(kwargs["puck_history"][-5:], position_homography).flatten().tolist()
        reward_regions_states = [nrr for nrr in state_info['negative_regions']]
        
        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel] + puck_hist)
        return np.concatenate([obs] + reward_regions_states)
    elif obs_type == 'paddle_acceleration_vel':
        ego_paddle_x_acc = state_info['paddles']['paddle_ego']['acceleration'][0]
        ego_paddle_y_acc = state_info['paddles']['paddle_ego']['acceleration'][1]
        
        paddle_forces_x = state_info['paddles']['paddle_ego']['force'][0]
        paddle_forces_y = state_info['paddles']['paddle_ego']['force'][1]

        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_pos, puck_y_pos = _maybe_warp_xy(puck_x_pos, puck_y_pos, position_homography)
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1] 
        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, ego_paddle_x_acc, ego_paddle_y_acc, paddle_forces_x, paddle_forces_y, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        return obs
    elif obs_type == 'paddle_acceleration_history':
        ego_paddle_x_acc = state_info['paddles']['paddle_ego']['acceleration'][0]
        ego_paddle_y_acc = state_info['paddles']['paddle_ego']['acceleration'][1]
        
        paddle_forces_x = state_info['paddles']['paddle_ego']['force'][0]
        paddle_forces_y = state_info['paddles']['paddle_ego']['force'][1]

        puck_hist = _warp_history_xy(kwargs["puck_history"][-5:], position_homography).flatten().tolist()
        obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, ego_paddle_x_acc, ego_paddle_y_acc, paddle_forces_x, paddle_forces_y] + puck_hist)
        return obs
    elif obs_type == "multipuck_vel":
        obs = [ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel]
        
        for puck in state_info['pucks']:
            puck_x_pos, puck_y_pos = _maybe_warp_xy(
                puck['position'][0], puck['position'][1], position_homography
            )
            obs.append(puck_x_pos)
            obs.append(puck_y_pos)
            obs.append(puck['velocity'][0])
            obs.append(puck['velocity'][1])
        
        obs = np.array(obs)
        return obs
    elif obs_type == "multipuck_history":
        obs = [ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel]
        
        puck_hist = _warp_history_xy(
            kwargs["puck_history"][-5 * len(state_info['pucks']):],
            position_homography,
        ).flatten().tolist()
        obs = np.array(obs + puck_hist)
        return obs

    raise  ValueError("obs type " + obs_type + " is not a defined observation type")
