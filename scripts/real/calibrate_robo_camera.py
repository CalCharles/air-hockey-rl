import cv2
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from airhockey.sims.real.robot_control import apply_negative_z_force
import imageio

def find_robo_pixel(cap, offset):
    pixels = list()
    for i in range(100):
        ret, image = cap.read()
        # imageio.imsave("temp/ar_frames/frame_" + str(i) +".png", image)
        px = find_red_dot(image, offset)
        if px is not None: pixels.append(px)
    return np.mean(np.array(pixels), axis=0)

def find_red_dot(image, offset):
    # Load the image
    # image = cv2.imread(image_path)
    image = cv2.rotate(image, cv2.ROTATE_180)

    # Convert to HSV color space
    image = cv2.resize(image, (int(image.shape[1]), int(image.shape[0])), 
                    interpolation = cv2.INTER_LINEAR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:,:int(300)] = 0
    hsv_image[int(350):,:] = 0

    # hsv_image[:,:120] = 0
    # hsv_image[:,200:] = 0
    # hsv_image[200:,:] = 0

    # Define the range of red color in HSV
    # These values might need adjustment depending on the image
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create a mask for red color
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = mask1 + mask2
    # cv2.imshow('hsv',hsv_image)
    # cv2.imshow('mask',mask)
    # cv2.waitKey(10)
    vals = np.where(mask > 0)
    if len(vals[0]) < 10:
        return None
    x, y = int(np.round(np.median(vals[0]))),int(np.round(np.median(vals[1])))

    # # Draw detected blobs as red circles
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # print(image.shape)
    
    # image_with_keypoints = cv2.drawKeypoints(image, [(x,y)], np.array([]), (0, 0, 255),
    #                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # py = int(keypoints[0].pt[0])
    # px = int(keypoints[0].pt[1])
    # width=100
    # x,y = x + 30, y + 40 # top far
    # x,y = x - 20, y + 40 # bot far
    # x,y = x - 15, y + 10 # bot far
    # x,y = x + 40, y + 12 # bot far
    x,y = x+offset[0], y + offset[1]
    image[x-3:x+3, y-3:y+3, :] = 0
    cv2.imshow('id-ed',image)
    cv2.waitKey(10)

    return x,y

def calibrate_homography(camera_id, save_homographies):
    rtde_frequency = 500.0
    ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
    rcv = RTDEReceive("172.22.22.2")
    # moves the robot to fixed positions, then aligns the homography pixels so that they match those of the robot
    cap = cv2.VideoCapture(camera_id)

    ret, image = cap.read()
    image = cv2.rotate(image, cv2.ROTATE_180)
    upscale_constant = 3
    visual_downscale_constant = 2
    image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
                interpolation = cv2.INTER_LINEAR)
    cv2.imshow('image',image)
    cv2.waitKey(1)

    original_size = np.array([640, 480])
    offset_constants = np.array((2100, 500))
    # Coordinates that you want to Perspective Transform,[450,255], [455,93]
    # pts1 = np.float32([[357,295],[361,53],[509,40],[499,306]])
    pts1 = np.float32([[357,280],[361,38],[509,25],[499,291]])
    offsets1 = np.array([[30,40], [-20, 40], [-15,10], [40,12]])
    # Size of the Transformed Image
    # pts2 = np.float32([[400,400],[400,100],[550,100],[550,400]]), [-548,-206], [-541, 259]
    # pts2 = np.float32([[-829,389],[-834,-337],[-408,-345],[-398,391]])
    # pts2 = np.float32([[-829,379],[-834,-327],[-408,-355],[-398,381]])
    pts2 = np.float32([[-767,336],[-775,-361],[-426,-361],[-407,352]])
    apply_negative_z_force(ctrl)
    pts1 = list()
    for offset, robo_pt in zip(offsets1, pts2):
        vel = 0.8 # velocity limit
        acc = 0.8 # acceleration limit 
        angle = [-0.00153677648744038, -3.0647520618606172, 0.]

        reset_pose = ([robo_pt[0] * 0.001, robo_pt[1] * 0.001, 0.33] + angle, vel, acc)
        high_reset_success = ctrl.moveL(reset_pose[0], reset_pose[1], reset_pose[2], False)
        pts1.append(find_robo_pixel(cap, offset))
    pts1 =  np.float32(pts1)
    pts1 *= upscale_constant
    Mrob = cv2.getPerspectiveTransform(pts1,pts2)
    for val in pts1:
        cv2.circle(image,(int(val[0]),int(val[1])),5,(0,255,0),-1)

    pts2 += offset_constants
    Mimg = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)

    for i in range(1000):
        image = cv2.resize(image, (int(640 * upscale_constant / visual_downscale_constant), int(480 * upscale_constant / visual_downscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        cv2.imshow('image',image)
        dst = cv2.resize(dst, (int(640 * upscale_constant / visual_downscale_constant), int(480 * upscale_constant / visual_downscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        cv2.imshow("transformed", dst)
        cv2.waitKey(10)
    # Save calibration data
    if save_homographies:
        np.save('Mimg.npy', Mimg)
        np.save('Mrob.npy', Mrob)

if __name__ == "__main__":
    calibrate_homography(0, False)