import cv2
import apriltag

# Load the image
image_path = '/u/mrudolph/documents/air-hockey-rl/scripts/real/april_tag_detection_images/frame_44.png'  # Replace with the path to your image
image = cv2.imread(image_path)
import pdb; pdb.set_trace()
# Convert the image to grayscale (AprilTag detection requires grayscale images)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create the AprilTag detector
detector = apriltag.Detector()

# Detect AprilTags in the grayscale image
detections = detector.detect(gray)


# Draw detected AprilTags on the image
for detection in detections:
    # Get the corners of the AprilTag
    corners = detection.corners.astype(int)

    # Draw lines between the corners to outline the detected tag
    cv2.polylines(image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

    # Put the tag ID on the image
    tag_id = str(detection.tag_id)
    cv2.putText(image, tag_id, (corners[0][0], corners[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Display the image with AprilTag detection
# cv2.imshow('AprilTag Detection', image)

# save image with detection
cv2.imwrite('output.jpg', image)
