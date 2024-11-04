import os
import h5py
from airhockey.sims.real.proprioceptive_state import slicer, flatten
from airhockey.sims.real.trajectory_merging import write_trajectory
from dataset_management.plot_trajectories import plot_trajectories
import numpy as np
from collections.abc import Iterable
import copy
import cv2

DATA_RANGES = [[0,1], [1,2], [2,3], [3,4], [4,10], [10,16], [16,22], [22,24], [24,32]]
DATA_NAMES = ["cur_time", "tidx", "i", "estop", "pose", "speed", "force", "acc", "desired_pose"]
def old_slicer(val):
    cur_time, tidx, i, estop, pose, speed, force, acc, desired_pose = val[...,DATA_RANGES[0][0]:DATA_RANGES[0][1]],\
        val[...,DATA_RANGES[1][0]:DATA_RANGES[1][1]], \
        val[...,DATA_RANGES[2][0]:DATA_RANGES[2][1]], \
        val[...,DATA_RANGES[3][0]:DATA_RANGES[3][1]], \
        val[...,DATA_RANGES[4][0]:DATA_RANGES[4][1]], \
        val[...,DATA_RANGES[5][0]:DATA_RANGES[5][1]], \
        val[...,DATA_RANGES[6][0]:DATA_RANGES[6][1]], \
        val[...,DATA_RANGES[7][0]:DATA_RANGES[7][1]], \
        val[...,DATA_RANGES[8][0]:DATA_RANGES[8][1]]
    res_dict = {
        "cur_time": cur_time, "tidx": tidx, "i": i, "pose": pose, "speed": speed, "force": force, 
        "acc": acc, "desired_pose": desired_pose, "estop": estop
    }
    return res_dict

offset_constant = 1.2 # TODO: don't manually specify the offset constant


def repair_old_real_data(data_dir, target_dir):
    # should only ever need to run this once, 
    trajectories = list()
    for file in os.listdir(data_dir):
        traj = list()
        with h5py.File(os.path.join(data_dir, file), 'r') as f:
            # try:
            data_dict = dict()
            dat_imgs = np.array(f['train_img'])
            measured_vals = np.array(f['train_vals'])
            for mv in measured_vals:
                # print(mv, len(mv))
                updating_dict = old_slicer(mv)
                updating_dict["safety"] = [1] # one means things are safe
                updating_dict["puck"] = np.array([-2 + offset_constant, 0, 1])
                vals = flatten(updating_dict)
                # print(vals.shape)
                traj.append(vals)

            # except Exception as e:
            #     print('Error in file:', file, e)
            #     continue
            print("added trajectory ", file)
        vals = np.stack(traj, axis=0)
        imgs = dat_imgs
        write_trajectory(target_dir, 0, imgs, vals, filename=file) # filename replaces tidx

        trajectories.append(traj)
    return trajectories

# repair_old_real_data("/datastor1/calebc/public/data/mouse/cleaned/", "/datastor1/calebc/public/data/mouse/updated_old/")
# after this, move from updated_old to cleaned_all

def read_new_real_data(data_dir, num_load=-1):
    values = list()
    images = list()
    dones = list()
    itr = 0
    for file in os.listdir(data_dir):
        traj = list()
        print(file)
        with h5py.File(os.path.join(data_dir, file), 'r') as f:
            try:
                data_dict = dict()
                dat_imgs = np.array(f['train_img'])
                measured_vals = np.array(f['train_vals'])
                for im, mv in zip(dat_imgs, measured_vals):
                    # print(mv, len(mv))
                    updating_dict = slicer(mv)
                    values.append(updating_dict)
                    images.append(im)
                    dones.append(0)
                    
                dones.pop(-1)
                dones.append(1)

            except Exception as e:
                print('Error in file:', file, e)
                continue
            print("added trajectory ", file)
        itr += 1
        if itr > num_load and num_load > 0: break
    return values, images, dones
# read_new_real_data("/datastor1/calebc/public/data/mouse/cleaned_new/")

'''
Expected output keys:
cur_time: wall time (1,)
tidx: trajectory index (1,)
i: timestep in trajectory index (1,)
estop: estop indicator, 1 is estopped (1,)
safety: safety indicator, 0 if unsafe (1,)
pose: robot end effector pose xyz rxryrx (6,)
speed: robot end effector speed xyz rxryrz (6,)
force: robot end effector force xyz rxryrz (6,)
acc: robt end effector acceleration xyz (6,)
desired_pose: `action` desired posed xyz rxryrz (6,)
puck: location of the puck, with occlusion 1 if occluded xy occluded (3,)
'''
def collect_new_state_data(data_dir, state_dir, target_dir, skip_start = False, skip_end=False):
    # offset_constant = 1.0 # TODO: don't manually specify the offset constant
    trajectories = dict()
    for file in os.listdir(data_dir):
        traj = list()
        traj_num = int(file[len("trajectory_data"):-5])
        # if traj_num > 430: continue
        puck_traj = list()
        total_masked = 0
        if len(state_dir) > 0:
            try:
                with h5py.File(os.path.join(state_dir, f"state_trajectory_data{traj_num}.hdf5"), 'r') as f:
                    puck_state = np.array(f["puck_state"]).T
                    masks = np.array(f["puck_state_nan_mask"]).T
                    last_state = np.array((-2 + offset_constant,0,1))
                    for state, mask in zip(puck_state, masks):
                        if mask[0]:
                            state = np.zeros(last_state.shape)
                            state[0] = last_state[0]
                            state[1] = last_state[1]
                            state[2] = 1
                            total_masked += 1
                        else:
                            state[0], state[1] = state[1] + offset_constant, state[0]
                            state = np.concatenate([state, [0]])
                        print(state)
                        last_state = state
                        puck_traj.append(state)
            except FileNotFoundError as e:
                print("No state file found, ", file, e)
                continue
        with h5py.File(os.path.join(data_dir, file), 'r') as f:
            # puck_vals = np.load(os.path.join(state_dir, "state_trajectory_data{traj_num}.npy"))
                
            try:
                measured_vals = np.array(f['train_vals'])
                images = np.array(f['train_img'])
                if len(puck_traj):
                    for pv, mv, im in zip(puck_traj, measured_vals, images):
                        # print(mv, len(mv))
                        updating_dict = slicer(mv)
                        updating_dict["puck"] = pv
                        updating_dict["image"] = im
                        updating_dict["action"] = updating_dict["desired_pose"][:2] - updating_dict["pose"][:2] 
                        updating_dict["paddle"] = copy.deepcopy(updating_dict["pose"])
                        updating_dict["paddle"][0] += offset_constant
                        print(updating_dict["action"], updating_dict["puck"], updating_dict["paddle"][:2], [(k, len(updating_dict[k])) for k in updating_dict.keys()])
                        traj.append(updating_dict)
                else:
                    last_puck = np.array([-2 + offset_constant,0,1]) # -2 + real_env.offset_constant
                    start = True and skip_start
                    counter = 0
                    first = measured_vals[10]
                    ud_first = slicer(first)
                    for mv, im in zip(measured_vals, images):
                        # print(mv, len(mv))
                        updating_dict = slicer(mv)
                        counter += 1
                        if counter < 10 and skip_start:
                            continue
                        if np.sum(np.abs(updating_dict["pose"] - ud_first["pose"])) < 1e-10 and start:
                            print("skipping start reset", ud_first["pose"])
                            continue
                        start = False
                        if "puck" not in updating_dict or len(updating_dict["puck"]) == 0:
                            updating_dict["puck"] = copy.deepcopy(last_puck)
                        if np.sum(np.abs(updating_dict["puck"] - np.array([0.2,0,1]))) < 1e-10:
                            updating_dict["puck"] = copy.deepcopy(last_puck)
                            updating_dict["puck"][-1] = 1
                        updating_dict["action"] = updating_dict["desired_pose"][:2] - updating_dict["pose"][:2] 
                        updating_dict["paddle"] = copy.deepcopy(updating_dict["pose"])
                        updating_dict["paddle"][0] += offset_constant
                            
                        last_puck = updating_dict["puck"]
                        print(updating_dict["paddle"][:2], updating_dict["action"])
                        updating_dict["image"] = im
                        traj.append(updating_dict)
            except Exception as e:
                print('Error in file:', file, e)
                continue
            print("added trajectory ", file, "total masked: ", total_masked, len(measured_vals), len(puck_traj))
        if len(traj) > 0:
            traj_dict = dict()
            for k in traj[0].keys():
                vals = list()
                for state_dict in traj:
                    vals.append(state_dict[k])
                traj_dict[k] = np.stack(vals)
            try:  
                os.makedirs(target_dir)
                print("made ", target_dir)
            except OSError as error:
                pass

            with h5py.File(os.path.join(target_dir, file), 'w') as hf:
                for k in traj_dict.keys():
                    hf.create_dataset(k,
                                    shape=traj_dict[k].shape,
                                    compression="gzip",
                                    compression_opts=9,
                                    data = traj_dict[k])
        else:
            print("SKIP", file)
    return trajectories

def read_real_data(data_dir, num_load=-1):
    data_dict = dict()
    itr = 0
    for file in os.listdir(data_dir):
        print(file)
        tidx = int(file[len("trajectory_data"):-5])
        try:
            os.makedirs(os.path.join(data_dir, "imgs"))
        except OSError as e:
            print(e)
        with h5py.File(os.path.join(data_dir, file), 'r') as f:
            try:
                for k in f.keys():
                    # print(k, np.array(f[k]))
                    # print(mv, len(mv))
                    if k in data_dict:
                        data_dict[k].append(np.array(f[k]))
                    else:
                        data_dict[k] = [np.array(f[k])]
                dones = np.zeros(len(f[k]))
                dones[-1] = 1
                if "done" in data_dict:
                    data_dict["done"].append(dones)
                else:
                    data_dict["done"] = [dones]
                for i in range(len(dones)):
                    cv2.imshow("frame", data_dict["image"][i])
                    cv2.waitKey(100)
                    print(tidx, data_dict["puck"][-1][i][:2], data_dict["paddle"][-1][i][:2], data_dict["action"][-1][i][:2], data_dict["desired_pose"][-1][i][:2] - data_dict["pose"][-1][i][:2])
                plot_trajectories(os.path.join(data_dir, "imgs"), tidx, data_dict["paddle"][-1][:,:2], data_dict["puck"][-1][:,:2])
            except Exception as e:
                print('Error in file:', file, e)
                continue
            print("added trajectory ", file)
        itr += 1
        if itr > num_load and num_load > 0: break
    for k in data_dict.keys():
        # print(k, data_dict[k])
        if k not in ["num_hits", "occlusions"]: data_dict[k] = np.concatenate(data_dict[k], axis=0)
        else: data_dict[k] = np.stack(data_dict[k])
        # print(k, [datav.shape for datav in data_dict[k]])
        # print(k, data_dict[k].shape)
    # for i in range(len(data_dict["done"])):
    #     print(list(data_dict.keys()))


    return data_dict
# read_new_real_data("/datastor1/calebc/public/data/mouse/cleaned_new/")

if __name__ == "__main__":
    # repair_old_real_data("/datastor1/calebc/public/data/mouse/cleaned/", "/datastor1/calebc/public/data/mouse/updated_old/")
    # after this, move from updated_old to cleaned_all
    # collect_new_state_data("/datastor1/calebc/public/data/mouse/cleaned_all/", "/datastor1/calebc/public/data/mouse/all_cleaned_state_trajectories_5-25-2024/", "/datastor1/calebc/public/data/mouse/state_data_all_new")
    
    # collect_new_state_data("/datastor1/calebc/public/data/mouse/expert_avoid_fixed_start_fixed_goal/", "", "/datastor1/calebc/public/data/mouse/expert_avoid_fixed_start_fixed_goal_all_new/")
    # collect_new_state_data("/datastor1/calebc/public/data/mouse/expert_avoid_random_start_fixed_goal/", "", "/datastor1/calebc/public/data/mouse/expert_avoid_random_start_fixed_goal_all_new/")
    # collect_new_state_data("/datastor1/calebc/public/data/mouse/expert_avoid_random_start_random_goal/", "", "/datastor1/calebc/public/data/mouse/expert_avoid_random_start_random_goal_all_new/")
    # collect_new_state_data("/datastor1/calebc/public/data/mouse/expert_no_avoid_random_start_random_goal/", "", "/datastor1/calebc/public/data/mouse/expert_no_avoid_random_start_random_goal_all_new/")
    
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/single_drop_expert/", "", "/datastor1/calebc/public/data/dilo/single_drop_expert_all_new/")
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/single_drop_random/", "/datastor1/calebc/public/data/dilo/single_drop_random_state_trajectories/", "/datastor1/calebc/public/data/dilo/single_drop_random_all_new/")
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/multi_drop_expert/", "", "/datastor1d/", "", "/datastor1/calebc/public/data/dilo/observe/observe_reaching_expert_fixed_all_new/", skip_start = True)
    
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/observe/observe_reaching_expert_random/", "", "/datastor1/calebc/public/data/dilo/observe/observe_reaching_expert_random_all_new/", skip_start = True)
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/observe/observe_reaching_expert_neg_sta/calebc/public/data/dilo/multi_drop_expert_all_new/")
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/observe/observe_reaching_expert_fixert/", "", "/datastor1/calebc/public/data/dilo/observe/observe_reaching_expert_neg_start_all_new/", skip_start = True)
    
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/stationary/hitting_expert/", "", "/datastor1/calebc/public/data/dilo/stationary/hitting_expert_all/")
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/stationary/hitting_random/", "", "/datastor1/calebc/public/data/dilo/stationary/hitting_random_all/")

    # collect_new_state_data("/datastor1/calebc/public/data/dilo/drawing/drawing_C/", "", "/datastor1/calebc/public/data/dilo/drawing/drawing_C_all/")
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/drawing/drawing_random/", "", "/datastor1/calebc/public/data/dilo/drawing/drawing_random_all/")
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/drawing/drawing_J/", "", "/datastor1/calebc/public/data/dilo/drawing/drawing_J_all/")
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/drawing/drawing_S/", "", "/datastor1/calebc/public/data/dilo/drawing/drawing_S_all/")
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/drawing/drawing_O/", "", "/datastor1/calebc/public/data/dilo/drawing/drawing_O_all/")
    # collect_new_state_data("/datastor1/calebc/public/data/dilo/drawing/drawing_W/", "", "/datastor1/calebc/public/data/dilo/drawing/drawing_W_all/")


    # read_real_data("/datastor1/calebc/public/data/dilo/single_drop_random_all_new/", 20)
    # read_real_data("/datastor1/calebc/public/data/mouse/expert_avoid_fixed_start_fixed_goal_all_new/")
    # read_real_data("/datastor1/calebc/public/data/mouse/expert_avoid_random_start_fixed_goal_all_new/")
    # read_real_data("/datastor1/calebc/public/data/mouse/expert_avoid_random_start_random_goal_all_new/")
    # read_real_data("/datastor1/calebc/public/data/mouse/expert_no_avoid_random_start_random_goal_all_new/")
    # read_real_data("/datastor1/calebc/public/data/mouse/state_data_all_new", num_load=5)
    read_real_data("/datastor1/calebc/public/data/dilo/stationary/hitting_expert_all", num_load=10)
