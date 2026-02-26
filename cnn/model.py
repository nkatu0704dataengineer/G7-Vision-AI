import torch
import torch.nn as nn
import torchvision.models as models

class CNNBackbone(nn.Module):
    """
    Module CNN đóng vai trò trích xuất đặc trưng (feature extractor) từ ảnh khuôn mặt [1-3].
    Kiến trúc dựa trên ResNet-50, một tiêu chuẩn trong nhận diện khuôn mặt [4-6].
    """
    def __init__(self, embedding_size=512, pretrained=True):
        super(CNNBackbone, self).__init__()
        # Sử dụng ResNet-50 làm kiến trúc cốt lõi trích xuất các đặc trưng phân cấp [4, 6].
        # Gồm các lớp Convolution, Activation (ReLU) và Pooling (Global Average Pooling) [7-9].
        self.model = models.resnet50(pretrained=pretrained)
        
        # Lấy kích thước đầu vào của lớp Fully Connected gốc (thường là 2048) [5, 10].
        num_features = self.model.fc.in_features
        
        # Thay thế lớp phân loại cuối cùng (Softmax head) bằng Identity [10-12].
        # Điều này giúp mô hình trả về đặc trưng thay vì nhãn phân loại lớp [10, 13].
        self.model.fc = nn.Identity()
        
        # Thêm lớp chiếu để xuất ra vector embedding có kích thước cố định (ví dụ: 512) [5, 6, 14].
        # Sử dụng BatchNorm để ổn định các đặc trưng phục vụ cho SVM và KNN [15, 16].
        self.embedding_layer = nn.Sequential(
            nn.Linear(num_features, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        """
        Đầu vào: Ảnh khuôn mặt đã căn chỉnh (Batch, 3, 224, 224) [7, 17, 18].
        Đầu ra: Vector đặc trưng (embedding) đã được chuẩn hóa L2 [19-21].
        """
        # Trích xuất đặc trưng thô từ xương sống CNN [2, 22].
        features = self.model(x)
        
        # Chuyển đổi thành vector đặc trưng (embedding) cố định [1, 2].
        embeddings = self.embedding_layer(features)
        
        # Chuẩn hóa L2 để đưa vector về không gian Euclidean/Hyperspherical [19, 21, 23].
        # Việc chuẩn hóa là bắt buộc để tích hợp hiệu quả với SVM và KNN [19, 24, 25].
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return normalized_embeddings