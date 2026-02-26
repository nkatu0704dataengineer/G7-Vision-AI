import torch
import os

class CNNConfig:
    """
    Module cấu hình trung tâm cho hệ thống nhận diện khuôn mặt CNN.
    Các tham số được thiết lập dựa trên các tiêu chuẩn nghiên cứu (SOTA).
    """
    DATA_DIR = "data/train"
    # --- Thông số hình ảnh đầu vào ---
    # Kích thước phổ biến: 224x224 (VGGFace/ResNet) hoặc 112x112 (ArcFace)
    INPUT_SIZE = (224, 224)
    CHANNELS = 3
    
    # --- Siêu tham số huấn luyện (Training Hyperparameters) ---
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 80
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    
    # --- Cấu hình mô hình ---
    BACKBONE_NAME = 'ResNet50'  # ResNet50 là tiêu chuẩn công nghiệp cho độ chính xác cao
    EMBEDDING_SIZE = 512        # Kích thước vector đặc trưng (embedding)
    
    # --- Tham số Metric Learning (ví dụ: ArcFace) ---
    ARC_SCALE = 64.0            # Tham số s (scaling factor)
    ARC_MARGIN = 0.5            # Tham số m (angular margin penalty)
    
    # --- Thiết bị tính toán ---
    # Tự động chọn GPU nếu có, ngược lại dùng CPU
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- Đường dẫn hệ thống ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(BASE_DIR, 'data')
    
    # Đường dẫn lưu trữ mô hình và log
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Tên file mô hình lưu trữ (.pth hoặc .checkpoint)
    MODEL_NAME = "cnn_face_recognition_model.pth"
    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    # Đảm bảo các thư thư mục tồn tại
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    NUM_WORKERS = 2
    PIN_MEMORY = True

# Khởi tạo một đối tượng cấu hình để sử dụng toàn cục
config = CNNConfig()