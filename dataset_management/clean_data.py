import h5py
import os
import cv2
import sys
import numpy as np



def homography_transform(image, from_save = False, get_save=True, rotate=False, Mimg = None):
    mousepos = (0,0,1)
    if Mimg is None: Mimg = np.load('assets/real/Mimg.npy')


    upscale_constant = 3
    original_size = np.array([640, 480])
    visual_downscale_constant = 2
    save_downscale_constant = 2
    offset_constants = np.array((2100, 500))

    image = cv2.rotate(image, cv2.ROTATE_180)
    save_image = None
    if from_save:
        image = cv2.rotate(image, cv2.ROTATE_180)
        image = cv2.resize(image, (int(640 * save_downscale_constant), int(480 *save_downscale_constant)))
    if get_save:
        save_image = cv2.resize(image, (int(640/save_downscale_constant), int(480/save_downscale_constant)))
    image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
                interpolation = cv2.INTER_LINEAR)
    dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)
    if rotate: 
        dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
        showdst = cv2.resize(dst, (int(480*upscale_constant / visual_downscale_constant), int(640*upscale_constant / visual_downscale_constant)), 
                interpolation = cv2.INTER_LINEAR)
    else:
        showdst = cv2.resize(dst, (int(640*upscale_constant / visual_downscale_constant), int(480*upscale_constant / visual_downscale_constant)), 
                interpolation = cv2.INTER_LINEAR)
    return showdst, save_image


def load_hdf5_to_dict(datapath):
    """
    Load a hdf5 dataset into a dictionary.

    :param datapath: Path to the hdf5 file.
    :return: Dictionary with the dataset contents.
    """
    data_dict = {}
    
    # Open the hdf5 file
    with h5py.File(datapath, 'r') as hdf:
        # Loop through groups and datasets
        def recursively_save_dict_contents_to_group(h5file, current_dict):
            """
            Recursively traverse the hdf5 file to save all contents to a Python dictionary.
            """
            for key, item in h5file.items():
                if isinstance(item, h5py.Dataset):  # if it's a dataset
                    current_dict[key] = item[()]  # load the dataset into the dictionary
                elif isinstance(item, h5py.Group):  # if it's a group (which can contain other groups or datasets)
                    current_dict[key] = {}
                    recursively_save_dict_contents_to_group(item, current_dict[key])

        # Start the recursive function
        recursively_save_dict_contents_to_group(hdf, data_dict)

    return data_dict

def write_trajectory(pth, tidx, imgs, vals, metadata=None):
    with h5py.File(os.path.join(pth, 'trajectory_data' + str(tidx) + '.hdf5'), 'w') as hf:
        hf.create_dataset("train_img",
                        shape=imgs.shape,
                        compression="gzip",
                        compression_opts=9,
                        data = imgs)
        print(vals)

        hf.create_dataset("train_vals",
                        shape=vals.shape,
                        compression="gzip",
                        compression_opts=9,
                        data = vals)
        print(tidx, hf)
        for key in metadata.keys():
            hf[key] = metadata[key]

def clean_data(pth, target, start_tidx=-1):
    filenames = list()
    for filename in os.listdir(pth):
        file_path = os.path.join(pth, filename)
        try:
            if os.path.isfile(file_path):
                filenames.append(filename)
        except Exception as e:
            print('Failed to recover %s. Reason: %s' % (file_path, e))

    target_filenames = list()
    for filename in os.listdir(target):
        file_path = os.path.join(pth, filename)
        try:
            if os.path.isfile(file_path):
                target_filenames.append(filename)
        except Exception as e:
            print('Failed to recover %s. Reason: %s' % (file_path, e))
    cleaned_trajectories = [int(fn[len("trajectory_data"):-5]) for fn in target_filenames]
    
    filenames.sort(key = lambda x: int(x[len("trajectory_data"):-5]))
    if start_tidx != -1: filenames = filenames[filenames.index("trajectory_data" + str(start_tidx) + ".hdf5"):]
    for fn in filenames:
        if fn.find("trajectory_data") == -1:
            continue
        tidx = int(fn[len("trajectory_data"):-5])
        if tidx in cleaned_trajectories:
            print("skipping", tidx)
            continue
        dataset_dict = load_hdf5_to_dict(os.path.join(pth, fn))
        idx = 0
        frames = dataset_dict['train_img']
        keep = False
        start, end = 0, 2000
        while True:
            frame = frames[idx]
            # cv2.imshow('frame',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow('frame',frame)
            key = cv2.waitKey(100)
            print("traj", tidx, "frame", idx, key)
            if key == 81: # left
                idx = max(idx - 1,0)
            if key == 83: # right
                idx = min(idx + 1, len(frames) - 1)
            if key == 82: # up
                idx = max(idx - 10,0)
            if key == 84: # down
                idx = min(idx + 10, len(frames) - 1)
            if key == 121: # y
                keep = True
                break
            if key == 110: # n
                keep = False
                break
            if key == 101: # e
                start = idx
            if key == 114: # r
                end = idx
        print("final statistics", tidx, keep, start, end)
        if keep: 
            nh = input("\nRecord the number of hits: ")
            occ = input("\nRecord 1 if occluded, 0 if not: ")
            write_trajectory(target, tidx, frames[start:end], dataset_dict['train_vals'][start:end], metadata={"num_hits": int(nh),"occlusions": int(occ)})

def save_frames(pth, target):
    import imageio
    filenames = list()
    for filename in os.listdir(pth):
        file_path = os.path.join(pth, filename)
        try:
            if os.path.isfile(file_path):
                filenames.append(filename)
        except Exception as e:
            print('Failed to recover %s. Reason: %s' % (file_path, e))
    
    filenames.sort(key = lambda x: int(x[len("trajectory_data"):-5]))
    for fn in filenames[0:]:
        if fn.find("trajectory_data") == -1:
            continue
        tidx = int(fn[len("trajectory_data"):-5])
        dataset_dict = load_hdf5_to_dict(os.path.join(pth, fn))
        idx = 0
        frames = dataset_dict['train_img']
        keep = False
        keep_indices = list()
        while True:
            frame = frames[idx]
            # cv2.imshow('frame',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow('frame',frame)
            key = cv2.waitKey(10)
            print("traj", tidx, "frame", idx, key)
            if key == 81:
                idx = max(idx - 1,0)
            if key == 83:
                idx = min(idx + 1, len(frames) - 1)
            if key == 121: # y
                keep = True
                break
            if key == 110: # 
                keep = False
                break
            if key == 101: # e
                j=keep_indices.append(idx)
        if keep:
            for fidx in keep_indices:
                imageio.imwrite(os.path.join(target, "traj" + str(tidx) + "_" + str(fidx) + ".png"), frames[fidx])

def rgb_frames(pth):
    import imageio
    filenames = list()
    for filename in os.listdir(pth):
        file_path = os.path.join(pth, filename)
        try:
            if os.path.isfile(file_path):
                filenames.append(filename)
        except Exception as e:
            print('Failed to recover %s. Reason: %s' % (file_path, e))
    
    for fn in filenames:
        pic = imageio.imread(os.path.join(pth, fn))
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pic = cv2.rotate(pic, cv2.ROTATE_90_CLOCKWISE)
        pic = pic[:,40:]
        imageio.imwrite(os.path.join(pth, fn + "_rgb.png"), pic)

def visualize_clean_data(target, tidx):
    dataset_dict = load_hdf5_to_dict(os.path.join(target, "trajectory_data" + str(tidx) + ".hdf5"))
    frames = dataset_dict['train_img']
    idx = 0
    while True:
        frame = frames[idx]
        # cv2.imshow('frame',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        cv2.resize(frame, (int(640), int(480)))        
        frame, sframe = homography_transform(frame)
        cv2.imshow('frame',frame)
        key = cv2.waitKey(100)
        print("traj", tidx, "frame", idx, key)
        if key == 81:
            idx = max(idx - 1,0)
        if key == 83:
            idx = min(idx + 1, len(frames) - 1)
        if key == 121: 
            break

def get_hits(pth):
    filenames = list()
    for filename in os.listdir(pth):
        file_path = os.path.join(pth, filename)
        try:
            if os.path.isfile(file_path):
                filenames.append(filename)
        except Exception as e:
            print('Failed to recover %s. Reason: %s' % (file_path, e))

    total_hits = list()
    for fn in filenames:
        dataset_dict = load_hdf5_to_dict(os.path.join(target, fn))
        if "num_hits" in dataset_dict:
            total_hits.append(dataset_dict["num_hits"])
            print(dataset_dict["num_hits"])
    total_hits = np.array(total_hits)
    total_hits[total_hits < 4] = 0 
    total_hits[total_hits >= 4] = 1 
    print(np.mean(total_hits))

def write_video(pth, target, tnums=[]):
    import imageio
    filenames = list()
    for filename in os.listdir(pth):
        file_path = os.path.join(pth, filename)
        try:
            if os.path.isfile(file_path):
                filenames.append(filename)
        except Exception as e:
            print('Failed to recover %s. Reason: %s' % (file_path, e))
    
    filenames.sort(key = lambda x: int(x[len("trajectory_data"):-5]))
    if len(tnums) > 0: 
        video = cv2.VideoWriter(os.path.join(target, "vid_" + "_".join(map(str, tnums)) + ".avi"), 0, 20, (320, 240))
    for fn in filenames[0:]:
        if fn.find("trajectory_data") == -1:
            continue
        tidx = int(fn[len("trajectory_data"):-5])
        if len(tnums) > 0 and tidx not in tnums:
            continue
        dataset_dict = load_hdf5_to_dict(os.path.join(pth, fn))
        idx = 0
        frames = dataset_dict['train_img']
        if len(tnums) <= 0: video = cv2.VideoWriter(os.path.join(target, "vid_" + str(tidx) + ".avi"), 0, 20, (320, 240))
        for frame in frames:
            video.write(frame)
        if len(tnums) <= 0: video.release() 
    if len(tnums) > 0: video.release()


# things to look for: start when the puck is dropped and the human hand is no longer over the white part of the table
    # invisible puck
    # end after the robot finishes the last hit and the puck is coming back down when the human misses the puck
    # the camera might say it is not responding, you don't need to close it
    # The script also asks for the number of hits and whether the puck is occluded (blocked) at any point in the cleaned segment,
        # so look out for those things
        

# usage: left, right advance and regress the frames respectively
    # n will reject the trajectory
    # y will keep the trajectory
    # e will set the start point
    # r will set the end point

if __name__ == '__main__':
# CHANGE THE FOLLOWING TO YOUR PATH AND TARGET
    pth = "../../../ut_airhockey_ur/data/mouse/trajectories/"
    target = "../../../ut_airhockey_ur/data/mouse/cleaned/"
    # clean_data(pth, target, -1)
    clean_data(pth, target, 764)
# UNCOMMENT and COMMENT OUT ABOVE if you just want to visualize some cleaned data
    # I left some examples in data/mouse/cleaned
    # press y to stop visualizing cleaned data
    tidx = 193
    # visualize_clean_data(pth, tidx)
# get hit mean
    # target = "/datastor1/calebc/public/data/mouse/cleaned"
    # get_hits(target)
    # pth = "/datastor1/calebc/public/data/IQL/trajectories/"
    # target = "/datastor1/calebc/public/data/select_frames/"
    # save_frames(pth, target)

    # pth = "/datastor1/calebc/public/data/select_frames/mouse/"
    # rgb_frames(pth)

    # pth = "/datastor1/calebc/public/data/mouse/cleaned/"
    # target = "/datastor1/calebc/public/data/mouse/videos/"
    # write_video(pth, target)