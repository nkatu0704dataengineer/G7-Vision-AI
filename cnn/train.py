import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

def train_model(backbone, dataset, config):
    """
    Hàm thực hiện huấn luyện CNN backbone dựa trên PyTorch [1], [2].
    Sử dụng kiến trúc phân cấp để tối ưu hóa vector đặc trưng (embeddings) [3], [4].
    """
    # Khởi tạo DataLoader để quản lý việc nạp dữ liệu theo lô (batch) [5], [6].
    train_loader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    ) # [6].

    # Chuyển mô hình backbone sang thiết bị tính toán đã cấu hình (CPU/GPU) [7], [8].
    device = config.DEVICE
    backbone = backbone.to(device)

    # Do backbone trả về embeddings, cần thêm một lớp tuyến tính (head) để tính loss phân loại [9], [10].
    # Số lượng lớp (classes) được lấy từ thuộc tính của tập dữ liệu [6].
    num_classes = dataset.get_num_classes()
    classifier_head = nn.Linear(config.EMBEDDING_SIZE, num_classes).to(device)

    # Định nghĩa hàm mất mát (Loss Function) [11], [6].
    # CrossEntropyLoss thường được sử dụng làm baseline cho bài toán nhận diện [12], [13].
    criterion = nn.CrossEntropyLoss()

    # Định nghĩa thuật toán tối ưu (Optimizer) [14], [15].
    # Sử dụng SGD với momentum hoặc Adam tùy theo cấu hình hệ thống [16], [13].
    optimizer = optim.SGD(
        list(backbone.parameters()) + list(classifier_head.parameters()),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    ) # [14].

    print(f"Bắt đầu huấn luyện trên thiết bị: {device}") # [17].

    # Vòng lặp huấn luyện theo số epoch đã định nghĩa [18], [19].
    for epoch in range(config.NUM_EPOCHS):
        backbone.train()
        classifier_head.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Thực hiện các bước huấn luyện PyTorch tiêu chuẩn [20], [21].
            optimizer.zero_grad() # Xóa gradient cũ [20].
            
            # Forward pass: Trích xuất đặc trưng và tính toán logits [20], [21].
            embeddings = backbone(inputs)
            logits = classifier_head(embeddings)
            
            # Tính toán độ lỗi (Loss) [20], [21].
            loss = criterion(logits, labels)
            
            # Backward pass: Tính toán gradient và cập nhật trọng số [20], [21].
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Hiển thị thống kê sau mỗi epoch [22], [23].
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    # Lưu trọng số của mô hình backbone sau khi hoàn tất huấn luyện [24], [25].
    # Đảm bảo thư mục lưu trữ tồn tại trước khi ghi file [24].
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    torch.save(backbone.state_dict(), config.MODEL_PATH)
    print(f"Đã lưu mô hình huấn luyện tại: {config.MODEL_PATH}") # [24].

    return backbone