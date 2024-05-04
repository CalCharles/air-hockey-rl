import os, imageio, h5py, shutil
import numpy as np

def merge_trajectory(image_path, images, vals):

    if len(images) == 0:
        list_of_files = filter( lambda x: os.path.isfile 
                            (os.path.join(image_path, x)), 
                                os.listdir(image_path) ) 
        list_of_files = list(list_of_files)
        list_of_files.sort()

        vidx = 0
        imgs = list()
        for fil, nextfil in zip(list_of_files, list_of_files[1:]):
            tfil, tnextfil = float(fil[3:-4]), float(nextfil[3:-4])
            tcur = vals[vidx][0]
            if tfil < tcur < tnextfil:
                if np.abs(tfil-tcur) >= np.abs(tnextfil - tcur):
                    print(nextfil, tcur)
                    imgs.append(imageio.imread(os.path.join(image_path, nextfil)))
                else:
                    print(fil, tcur)
                    imgs.append(imageio.imread(os.path.join(image_path, fil)))
                vidx += 1
                # cv2.imshow('hsv',imgs[-1])
                # cv2.waitKey(1)
                if vidx == len(vals):
                    break
    else:
        imgs = images

    imgs = np.stack(imgs, axis=0)
    
    vals = np.stack(vals, axis=0)
    print(imgs.shape, vals.shape)
    if imgs.shape[0] != vals.shape[0]:
        print("MISALIGNED")
        return None, None

    return imgs, vals
    

def clear_images(folder='./temp/images/'):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def write_trajectory(pth, tidx, imgs, vals):
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

def get_trajectory_idx(save_path):
    try:  
        os.mkdir(save_path)
        print("made ", save_path)
    except OSError as error:
        print(error)
        pass
    list_of_files = filter( lambda x: os.path.isfile 
            (os.path.join(save_path, x)), 
                os.listdir(save_path) )
    list_of_files = list(list_of_files)
    list_of_files.sort(key = lambda x: int(x[len("trajectory_data"):-5]))
    if len(list_of_files) == 0:
        tstart = 0
    else:
        tstart = int(list_of_files[-1][len("trajectory_data"):-5]) + 1
    return tstart