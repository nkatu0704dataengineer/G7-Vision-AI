#cnn/config.py
import torch

class Config:
    # 📁 ĐƯỜNG DẪN DATASET (BẮT BUỘC PHẢI CÓ)
    DATASET_PATH = "data/train"

    # Thông số ảnh đầu vào
    IMAGE_SIZE = 224
    CHANNELS = 3

    # Thông số huấn luyện
    BATCH_SIZE = 32
    LR = 0.0001
    EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Thông số model
    EMBEDDING_DIM = 128
    NUM_CLASSES = 4   # đúng với số folder trong data/train

    # Output
    MODEL_PATH = "face_cnn.pth"
