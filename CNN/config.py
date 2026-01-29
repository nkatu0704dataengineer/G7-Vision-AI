#cnn/config.py
import torch

class Config:
    # Thông số ảnh đầu vào
    IMAGE_SIZE = 224
    CHANNELS = 3
    
    # Thông số huấn luyện
    BATCH_SIZE = 32
    LR = 0.0001
    EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    EMBEDDING_DIM = 128  # Vector đặc trưng 128 chiều
    NUM_CLASSES = 4      # tu tung 2minh quan 
    
    MODEL_PATH = "face_cnn.pth"

