from dataset_management.repair_data import read_new_real_data
from airhockey.sims.real.control_parameters import homography_transform
import os
import numpy as np
import imageio
import cv2
import copy

def stack_frames(target_dir, data_dir, num_load=-1):
    values, images, dones, files = read_new_real_data(data_dir, num_load=num_load, show = False)

    traj_idxes = np.array(dones).nonzero()[0]
    print(traj_idxes)
    try:  
        os.makedirs(target_dir)
        print("made ", target_dir)
    except OSError as error:
        pass
    traj_ims = list()
    ltidx = 0
    stacked_images = list()
    for tidx, file in zip(traj_idxes, files):
        traj_ims = np.array([im.astype(float) for im in images[ltidx: tidx]])
        # stack = traj_ims[min(len(traj_ims) - 1, 10)]
        stack = np.ones(traj_ims[min(len(traj_ims) - 1, 10)].shape) * 255
        total = len(traj_ims[10:200])
        for i, timg in enumerate(traj_ims[10:200]):
            # time_decay_top = np.exp(-i / total * 100)
            time_decay_top = np.exp((i-total + 1) / total * 6.0)
            time_decay_bottom = np.exp((i-total + 1) / total * 4)
            # time_decay_top = i / total
            # timg =cv2.addWeighted(timg, time_decay_top, stack, 1 - time_decay_top, 0)
            timg[:,45:] =cv2.addWeighted(timg[:,45:], time_decay_bottom, stack[:,45:], 1 - time_decay_bottom, 0) # 43 for dice, 40 for bco, 45 for smodice
            timg[:,:45] =cv2.addWeighted(timg[:,:45], time_decay_top, stack[:,:45], 1 - time_decay_top, 0)
            stack[:,:,0] = np.minimum(stack[:,:,0], timg[:,:,0])
            stack[:,:,1] = np.minimum(stack[:,:,1], timg[:,:,1])
            stack[:,:,2] = np.minimum(stack[:,:,2], timg[:,:,2])
        stack = cv2.rotate(stack, cv2.ROTATE_180)

        stack, save_im = homography_transform(stack, rotate=True)
        # stack = np.sum(np.array(traj_ims), axis=0)
        # stack = (stack * (1/len(traj_ims))).astype(np.uint8)
        stacked_images.append(stack)
        cv2.imshow("frame", stack.astype(np.uint8))
        cv2.waitKey(1000)
        stack[...,2], stack[...,0], stack[...,1] = copy.deepcopy(stack[:,:,0]), copy.deepcopy(stack[:,:,2]), copy.deepcopy(stack[:,:,1])
        imageio.imsave(os.path.join(target_dir, "stack_" + file + ".png"), stack.astype(np.uint8))
        ltidx = tidx
    return stacked_images

def swap_colors(target_path, data_path):
    image = imageio.imread(data_path)
    image[...,2], image[...,0], image[...,1] = copy.deepcopy(image[:,:,0]), copy.deepcopy(image[:,:,2]), copy.deepcopy(image[:,:,1])
    imageio.imwrite(target_path, image)

def video_trajectory(target_dir, data_dir, num_load=-1):
    values, images, dones, files = read_new_real_data(data_dir, num_load=num_load, show = False)

    traj_idxes = np.array(dones).nonzero()[0]
    print(traj_idxes)
    try:  
        os.makedirs(target_dir)
        print("made ", target_dir)
    except OSError as error:
        pass
    traj_ims = list()
    ltidx = 0
    stacked_images = list()
    counter = 0
    for tidx, file in zip(traj_idxes, files):
        traj_ims = np.array([im.astype(float) for im in images[ltidx: tidx]])
        # stack = traj_ims[min(len(traj_ims) - 1, 10)]
        vid_images, untransformed_images = list(), list()
        for i, timg in enumerate(traj_ims[10:200]):
            timg = cv2.rotate(timg, cv2.ROTATE_180)

            vid_img, save_im = homography_transform(timg, rotate=True)
            vid_img = vid_img[...,::-1]
            vid_images.append(vid_img)
            save_im = save_im[...,::-1]
            untransformed_images.append(save_im)
        # stack = np.sum(np.array(traj_ims), axis=0)
        # stack = (stack * (1/len(traj_ims))).astype(np.uint8)
        vidwriter = imageio.get_writer(os.path.join(target_dir, "homography_trajectory" + str(counter) + ".mp4"), fps=20)

        for im in vid_images:
            vidwriter.append_data(im.astype(np.uint8))
        vidwriter.close()        
        untr_writer = imageio.get_writer(os.path.join(target_dir, "raw_trajectory" + str(counter) + ".mp4"), fps=20)

        for im in untransformed_images:
            untr_writer.append_data(im.astype(np.uint8))
        untr_writer.close()
        print("wrote ", tidx) 
        ltidx = tidx
        counter += 1
    return stacked_images



if __name__ == "__main__":
    # stack_frames("data/stack_images_dilo15/", "data/rollout/puck_hitting_dilo15", num_load=-1)
    # stack_frames("data/stack_images_bco15", "data/rollout/puck_hitting_bco15", num_load=-1)
    # stack_frames("data/stack_images_smodice15", "data/rollout/puck_hitting_smodice15", num_load=-1)
    # stack_frames("data/stack_images_expert", "data/mouse/multi_drop_expert", num_load=2)
    # swap_colors("data/observe/observe_keep/observe.jpg", "data/observe/observe_keep/img1717543861.8622344.jpg")
    # video_trajectory("data/videos/dilo15/", "data/rollout/puck_hitting_dilo15", num_load=-1)

    all_files = os.listdir("data/observe/observe_raws")
    all_files.sort(key = lambda x: float(x[len("img"):-5]))

    try:  
        os.makedirs("data/videos")
        print("made ", "data/videos")
    except OSError as error:
        pass
    traj_ims = list()
    ltidx = 0
    counter = 0
    for file in all_files:
        vid_img = cv2.imread(os.path.join("data/observe/observe_raws", file))
        # vid_img = vid_img[...,::-1]
        traj_ims.append(vid_img)
    vidwriter = imageio.get_writer(os.path.join("data/videos", "observed_trajectory" + str(counter) + ".mp4"), fps=20)

    for im in traj_ims:
        vidwriter.append_data(im.astype(np.uint8))
    vidwriter.close()        
    # video_trajectory("data/videos/bco15", "data/rollout/puck_hitting_bco15", num_load=-1)
    # video_trajectory("data/videos/smodice15", "data/rollout/puck_hitting_smodice15", num_load=-1)
