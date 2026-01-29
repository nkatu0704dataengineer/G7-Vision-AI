# run_train.py
import cv2
from cnn.extract_feature import extract_feature

# Test với ảnh file
img = cv2.imread("test.jpg")
feature = extract_feature(img)
print("Feature shape:", feature.shape)

# Test camera (optional)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    feature_cam = extract_feature(frame)
    print("Camera feature shape:", feature_cam.shape)

cap.release()
