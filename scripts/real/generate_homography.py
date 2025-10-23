import cv2
import os
import numpy as np


# 0.5545621421040268, 0.48758756141253956
# 265, 466
# 1.0064265409081836, 0.4957963720907306
# 265, 582
# 0.5529227565159962, -0.4840302450683018
# 504, 464
# 0.9633642320739517, -0.4887553926441414
# 503, 577
def update_homography_robosuite(target=""):
    pts1 = np.float32([[466, 265 ],[582, 265],[464, 504],[577, 503]])
    # Size of the Transformed Image
    rl_shift, td_shift = -0.0,-0.0
    pts2 = np.float32([[0.5545621421040268 + td_shift, -0.48758756141253956 + rl_shift],
                       [1.0064265409081836 - td_shift, -0.4957963720907306 + rl_shift],
                       [0.5529227565159962 + td_shift, 0.4840302450683018 - rl_shift],
                       [0.9633642320739517 - td_shift, 0.4887553926441414 - rl_shift]])
    Mrob = cv2.getPerspectiveTransform(pts2,pts1)
    Mimg = cv2.getPerspectiveTransform(pts1,pts2)

    # Save calibration data
    if len(target) > 0:
        np.save(os.path.join(target, 'Mimg.npy'), Mimg)
        np.save(os.path.join(target, 'Mrob.npy'), Mrob)




def update_homography(target=""):
    cap = cv2.VideoCapture(1)

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
    pts1 *= upscale_constant
    # Size of the Transformed Image
    # pts2 = np.float32([[400,400],[400,100],[550,100],[550,400]]), [-548,-206], [-541, 259]
    # pts2 = np.float32([[-829,389],[-834,-337],[-408,-345],[-398,391]])
    # pts2 = np.float32([[-829,379],[-834,-327],[-408,-355],[-398,381]])
    pts2 = np.float32([[-829,359],[-834,-377],[-408,-385],[-398,351]])
    Mrob = cv2.getPerspectiveTransform(pts2,pts1)
    for val in pts1:
        cv2.circle(image,(int(val[0]),int(val[1])),5,(0,255,0),-1)

    pts2 += offset_constants
    Mimg = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)

    for i in range(100):
        image = cv2.resize(image, (int(640 * upscale_constant / visual_downscale_constant), int(480 * upscale_constant / visual_downscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        cv2.imshow('image',image)
        dst = cv2.resize(dst, (int(640 * upscale_constant / visual_downscale_constant), int(480 * upscale_constant / visual_downscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        cv2.imshow("transformed", dst)
        cv2.waitKey(10)
    # Save calibration data
    if len(target) > 0:
        np.save(os.path.join(target, 'Mimg.npy'), Mimg)
        np.save(os.path.join(target, 'Mrob.npy'), Mrob)

if __name__ == "__main__":
    update_homography_robosuite(os.path.join('assets', 'robosuite'))