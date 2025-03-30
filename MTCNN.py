from mtcnn import MTCNN
import cv2  
import matplotlib.pyplot as plt  

# Load the image
image = cv2.imread("face.jpg")  
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

# Initialize MTCNN detector
detector = MTCNN()  

# Detect faces
faces = detector.detect_faces(image_rgb)  

# Draw bounding boxes and landmarks
for face in faces:
    x, y, width, height = face['box']
    keypoints = face['keypoints']

    # Draw rectangle around face
    cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Draw key points
    for point in keypoints.values():
        cv2.circle(image_rgb, point, 3, (0, 0, 255), -1)

# Show the image
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
