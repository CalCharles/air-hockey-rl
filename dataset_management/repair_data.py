import os
import h5py
from airhockey.sims.real.proprioceptive_state import slicer, flatten
from airhockey.sims.real.trajectory_merging import write_trajectory
import numpy as np

DATA_RANGES = [[0,1], [1,2], [2,3], [3,4], [4,9], [10,15], [16,21], [22,24], [25,31]]
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


def repair_old_real_data(data_dir, target_dir):
    # should only ever need to run this once, 
    trajectories = list()
    for file in os.listdir(data_dir):
        traj = list()
        with h5py.File(os.path.join(data_dir, file), 'r') as f:
            try:
                data_dict = dict()
                dat_imgs = np.array(f['train_img'])
                measured_vals = np.array(f['train_vals'])
                for mv in measured_vals:
                    # print(mv, len(mv))
                    updating_dict = old_slicer(mv)
                    updating_dict["safety"] = [1] # one means things are safe
                    traj.append(flatten(updating_dict))

            except Exception as e:
                print('Error in file:', file, e)
                continue
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
def collect_new_state_data(data_dir, state_dir, target_dir):
    trajectories = dict()
    for file in os.listdir(data_dir):
        traj = list()
        traj_num = int(file[len("trajectory_data"):-5])
        # if traj_num > 430: continue
        puck_traj = list()
        total_masked = 0
        try:
            with h5py.File(os.path.join(state_dir, f"state_trajectory_data{traj_num}.hdf5"), 'r') as f:
                puck_state = np.array(f["puck_state"]).T
                masks = np.array(f["puck_state_nan_mask"]).T
                last_state = np.array((-2,0,0))
                for state, mask in zip(puck_state, masks):
                    if mask[0]:
                        state = np.zeros(last_state.shape)
                        state[0] = last_state[0]
                        state[1] = last_state[1]
                        total_masked += 1
                    else:
                        state = np.concatenate([state, [1]])
                    puck_traj.append(state)
        except Exception as e:
            print("No state file found, ", file, e)
            continue
        with h5py.File(os.path.join(data_dir, file), 'r') as f:
            # puck_vals = np.load(os.path.join(state_dir, "state_trajectory_data{traj_num}.npy"))
            try:
                measured_vals = np.array(f['train_vals'])
                images = np.array(f['train_img'])
                # print(len(measured_vals))
                for pv, mv, im in zip(puck_traj, measured_vals, images):
                    # print(mv, len(mv))
                    updating_dict = slicer(mv)
                    updating_dict["puck"] = pv
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
        with h5py.File(os.path.join(data_dir, file), 'r') as f:
            try:
                for k in f.keys():
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

            except Exception as e:
                print('Error in file:', file, e)
                continue
            print("added trajectory ", file)
        itr += 1
        if itr > num_load and num_load > 0: break
    for k in data_dict.keys():
        data_dict[k] = np.concatenate(data_dict[k], axis=0)
    return data_dict
# read_new_real_data("/datastor1/calebc/public/data/mouse/cleaned_new/")

if __name__ == "__main__":
    collect_new_state_data("/datastor1/calebc/public/data/mouse/cleaned_all/", "/datastor1/calebc/public/data/mouse/all_cleaned_state_trajectories_5-25-2024/", "/datastor1/calebc/public/data/mouse/state_data_all")
