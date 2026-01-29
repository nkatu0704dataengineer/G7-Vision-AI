# run_train.py
from cnn.train import train
from cnn.extract_feature import extract_feature
import cv2
import os

if __name__ == "__main__":
    # 1. Train nếu chưa có model
    if not os.path.exists("face_cnn.pth"):
        print("Training model...")
        train()

    # 2. Test inference với ảnh
    img = cv2.imread("test.jpg")
    if img is not None:
        feature = extract_feature(img)
        print("Feature shape:", feature.shape)
    else:
        print("Không tìm thấy test.jpg")
