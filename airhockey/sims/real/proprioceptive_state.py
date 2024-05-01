import numpy as np

DATA_RANGES = [[0,1], [1,2], [2,3], [3,4], [4,5],[5,10], [11,16], [17,22], [23,25], [26,32]]
DATA_NAMES = ["cur_time", "tidx", "i", "pose", "speed", "force", "acc", "desired_pose", "estop", "safety"]
def slicer(val):
    cur_time, tidx, i, pose, speed, force, acc, desired_pose, estop, safety = val[...,DATA_RANGES[0][0]:DATA_RANGES[0][1]],\
        val[...,DATA_RANGES[1][0]:DATA_RANGES[1][1]], \
        val[...,DATA_RANGES[2][0]:DATA_RANGES[2][1]], \
        val[...,DATA_RANGES[3][0]:DATA_RANGES[3][1]], \
        val[...,DATA_RANGES[4][0]:DATA_RANGES[4][1]], \
        val[...,DATA_RANGES[5][0]:DATA_RANGES[5][1]], \
        val[...,DATA_RANGES[6][0]:DATA_RANGES[6][1]], \
        val[...,DATA_RANGES[7][0]:DATA_RANGES[7][1]], \
        val[...,DATA_RANGES[8][0]:DATA_RANGES[8][1]], \
        val[...,DATA_RANGES[9][0]:DATA_RANGES[9][1]]
    res_dict = {
        "cur_time": cur_time, "tidx": tidx, "i": i, "pose": pose, "speed": speed, "force": force, 
        "acc": acc, "desired_pose": desired_pose, "estop": estop, "safety": safety
    }
    return cur_time, tidx, i, pose, speed, force, acc, desired_pose, estop, safety, res_dict

def get_state_array(cur_time, tidx, i, pose, speed, force, acc, desired_pose, estop, safety):
    # rcv = RTDEReceive("172.22.22.2")
    # ret, image = cap.read()
    # timestamp = time.time()
    # print(image.shape)
    # # cv2.imshow('capture',image)
    # # cv2.waitKey(1)
    # image = None

    # print(np.array([tidx]), np.array([i]), pose, speed, force, acc, desired_pose)

    # val = np.concatenate([np.array([cur_time]), # 0
    #                    np.array([tidx]), # 1 
    #                    np.array([i]), # 2 
    #                    np.array([estop]).astype(float), # 3
    #                    np.array(pose), # 4-9
    #                    np.array(speed), # 10-15
    #                    np.array(force), # 16-21
    #                    np.array(acc), # 22-24
    #                    np.array(desired_pose[0])]) # 25-31
    val = np.concatenate([np.array([cur_time]), # 0
                       np.array([tidx]), # 1 
                       np.array([i]), # 2 
                       np.array([estop]).astype(float), # 3
                       np.array([safety]), # 4
                       np.array(pose), # 5-10
                       np.array(speed), # 11-16
                       np.array(force), # 17-22
                       np.array(acc), # 23-25
                       np.array(desired_pose[0])]) # 26-32
    return val#, image
