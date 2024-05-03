import cv2
import numpy as np

def find_red_hockey_paddle(image):
    # Load the image
    # image = cv2.imread(image_path)

    # Convert to HSV color space
    # start = time.time()
    # print(image.shape)
    image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)), 
                    interpolation = cv2.INTER_LINEAR)
    # print(image.shape)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv_image[:,:int(540)] = 0
    # hsv_image[int(400):,:] = 0
    hsv_image[:,:int(540 / 4)] = 0
    hsv_image[int(500 / 4):,:] = 0

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
    # # cv2.imshow('mask',mask)
    # cv2.waitKey(1)



    # Blob detection parameters
#     params = cv2.SimpleBlobDetector_Params()
#     params.filterByColor = True
#     params.blobColor = 255  # Since the mask will be white where the puck is
#     params.minDistBetweenBlobs = 100


#    # Filter by Area.
#     params.filterByArea = False
#     params.minArea = 100  # Adjust based on the size of the puck in the image

#     # Filter by Circularity
#     params.filterByCircularity = False
#     params.minCircularity = 0.7  # Adjust to better match the puck's shape

#     # Filter by Convexity
#     params.filterByConvexity = False
#     params.minConvexity = 0.8

#     # Filter by Inertia
#     params.filterByInertia = True
#     params.minInertiaRatio = 0.5
    
    # Create a detector with the parameters
    # detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    # keypoints = detector.detect(mask)
    vals = np.where(mask > 0)
    x, y = int(np.round(np.median(vals[0]))),int(np.round(np.median(vals[1])))

    # # Draw detected blobs as red circles
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # print(image.shape)
    
    # image_with_keypoints = cv2.drawKeypoints(image, [(x,y)], np.array([]), (0, 0, 255),
    #                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # py = int(keypoints[0].pt[0])
    # px = int(keypoints[0].pt[1])
    # width=100
    image[x-3:x+3, y-3:y+3, :] = 0
    # Save the image with keypoints
    # cv2.imwrite('output.jpg', image_with_keypoints)
    # print("inrange", time.time()-start)
    return x*4,y*4,image



MIN_DETECT = 25
Mimg = np.load('assets/Mimg.npy')
upscale_constant = 3
original_size = np.array([640, 480])
visual_downscale_constant = 2
save_downscale_constant = 2
offset_constants = np.array((2100, 500))

def find_red_hockey_puck(image, puck_history):
    # hsv_alt should e a lit
    h, w, _ = image.shape
    # lower HSV: [110  25 119], upper HSV: [125 255 255]
    hsv_low = [  0, 137 ,  80]
    hsv_high = [ 8, 255, 255]
    # hsv_low = [0,100,140]
    # hsv_high=[50,255,255]
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)), 
                    interpolation = cv2.INTER_LINEAR)
    image[249:,:] = 0
    image[:9] = 0
    image[:,470:] = 0

    # Convert the left half of the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # We'll lower the saturation and value thresholds to possibly capture a darker green
    refined_lower = np.array(hsv_low)  # Lower saturation and value
    refined_upper = np.array(hsv_high)
    # print(hsv_image[:,:,2])
    # Create a mask for green color in the left half with the refined thresholds
    refined_mask = cv2.inRange(hsv_image, refined_lower, refined_upper)
    # remove_table_edges_mask = np.zeros((h,w), dtype=np.uint8)
    # remove_table_edges_mask[0:175, 30:290] = 1
    # refined_mask *= remove_table_edges_mask

    puck_idx = np.where(refined_mask)
    cv2.imshow('MASK',refined_mask)
    cv2.waitKey(1)

    if len(puck_idx[0]) < MIN_DETECT:
        return puck_history[-1][0], puck_history[-1][1], 1
    y, x = int(np.round(np.median(puck_idx[0]))),int(np.round(np.median(puck_idx[1])))
    image[y-3:y+3, x-3:x+3, :] = 0
    # homo_idx = (Mimg @ np.array([[x * upscale_constant,y * upscale_constant,1]]).T - offset_constants / 2) * 0.001
    homo_idx = (np.array([x*2 * 2,y*2 * 2]) - offset_constants) * 0.001
    cv2.imshow('detect',image)
    cv2.waitKey(1)
    print(image.shape, offset_constants, x,y, x*2 * 1.5,y*2 * 1.5, homo_idx)
    # print(h,w, x,y,homo_idx)
    return homo_idx[0], -homo_idx[1], 0


    # Detect blobs
    # keypoints = detector.detect(mask)
    while True:
        cv2.imshow('mask',hsv_image)
        cv2.imshow('mask',mask)
        cv2.waitKey(1)
    vals = np.where(mask > 0)
    x, y = int(np.round(np.median(vals[0]))),int(np.round(np.median(vals[1])))

    image[x-5:x+5, y-5:y+5, :] = 0
    # Save the image with keypoints
    # cv2.imwrite('output.jpg', image_with_keypoints)

    return x,y,image