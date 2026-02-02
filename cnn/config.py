#cnn/config.py
#Lưu toàn bộ tham số & cấu hình dùng chung cho toàn project
import torch

class Config:
    # 📁 ĐƯỜNG DẪN DATASET (BẮT BUỘC PHẢI CÓ)
    DATASET_PATH = "data/train"

    # Thông số ảnh đầu vào
    IMAGE_SIZE = 224
    CHANNELS = 3  #RGB

    # Thông số huấn luyện
    BATCH_SIZE = 32  # Nhóm 32 ảnh
    LR = 0.0001     # Tốc độ học
    EPOCHS = 50  # 1 epoch = CNN học hết toàn bộ dataset 1 lần
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Nếu có GPU NVIDIA → dùng GPU 
                                                            # Không có → dùng CPU

    # Thông số model
    EMBEDDING_DIM = 128 # Vector
    NUM_CLASSES = 4   # đúng với số folder trong data/train (Nhãn phân loại)

    # Output
    MODEL_PATH = "face_cnn.pth"
