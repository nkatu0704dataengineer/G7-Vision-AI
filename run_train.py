import torch
from cnn.config import config as cfg
from cnn.dataset import FaceDataset
from cnn.model import CNNBackbone
from cnn.train import train_model


def run_pipeline():
    """
    Điểm vào trung tâm (Entry Point) để điều phối luồng huấn luyện 
    hệ thống trích xuất đặc trưng khuôn mặt CNN.
    """
    
    # 1. Tải các tham số cấu hình hệ thống
    print(f"Đang khởi tạo quy trình huấn luyện trên thiết bị: {cfg.DEVICE}")

    # 2. Khởi tạo tập dữ liệu (Dataset)
    # Dữ liệu được tổ chức theo cấu trúc danh tính (Identity-based folders)
    train_dataset = FaceDataset(
        root_dir=cfg.DATA_DIR,
        input_size=cfg.INPUT_SIZE
    )

    
    print(f"Số lượng danh tính phát hiện: {train_dataset.get_num_classes()}")
    print(f"Tổng số mẫu ảnh huấn luyện: {len(train_dataset)}")

    # 3. Khởi tạo kiến trúc mô hình CNN Backbone
    # Sử dụng ResNet-50 làm trích xuất đặc trưng cơ sở (Backbone)
    face_backbone = CNNBackbone(
        embedding_size=cfg.EMBEDDING_SIZE,
        pretrained=True
    )

    # 4. Kích hoạt quy trình huấn luyện điều phối
    # Hàm này thực hiện vòng lặp huấn luyện, tính toán tối ưu hóa và lưu trữ model
    try:
        trained_model = train_model(
            backbone=face_backbone,
            dataset=train_dataset,
            config=cfg
        )
        
        print("-" * 50)
        print("KẾT QUẢ: Quá trình huấn luyện CNN hoàn tất thành công.")
        print(f"Mô hình backbone đã được lưu trữ tại: {cfg.MODEL_PATH}")
        print("-" * 50)
        
    except Exception as e:
        print(f"LỖI: Quá trình huấn luyện bị gián đoạn. Chi tiết: {str(e)}")

if __name__ == "__main__":
    # Thực thi script huấn luyện hệ thống
    run_pipeline()