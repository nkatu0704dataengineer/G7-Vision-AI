#cnn/model.py (Định nghĩa kiến trúc - Bản "Chay" & "Tối ưu")

import torch
import torch.nn as nn
import torchvision.models as models

class FaceCNN(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=4):
        super(FaceCNN, self).__init__()
        
        # 1. Kế thừa VGG16 (Backbone trích xuất đặc trưng)
        vgg = models.vgg16(pretrained=True) # vgg16 đã chứa các lớp trong cnn
                                            #Conv → ReLU → Conv → ReLU → MaxPool . . .
                                            # Cụ thể gồm 13 Conv2D và 5 MaxPooling
        self.feature_extractor = vgg.features # Giữ lại các lớp Convolution( Giữ lại các đặc trưng)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # 2. Lớp chuyển đổi sang Vector Embedding (Phần "chay" chi tiết)
        self.flatten = nn.Flatten()
        self.fc_embedding = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),    # Học 512 đặc trưng quan trọng nhất
            nn.ReLU(),  # học phi tuyến, phân biệt người A và B tốt hơn
            nn.Dropout(0.4),    # chống học vẹt
            nn.Linear(512, embedding_dim) # Đây là Vector 128D cho SVM/KNN
        )
        
        # 3. Lớp phân loại (Dùng để huấn luyện CNN ban đầu)
        self.classifier = nn.Linear(embedding_dim, num_classes)     #Dự đoán
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))     # cố định kích thước ảnh đầu ra

    def forward(self, x, return_embedding=False):
        # Trích xuất đặc trưng không gian
        x = self.feature_extractor(x)   # Trích đặc trưng không gian
        x = self.avgpool(x)     # Chuẩn hóa kích thước
        x = self.flatten(x)
        
        # Tạo vector embedding
        embedding = self.fc_embedding(x)
        
        if return_embedding:
            return embedding # Trả về vector để Người 2 dùng SVM/Logic mờ
            
        # Trả về kết quả phân loại (Softmax) để tính Loss khi train
        output = self.classifier(embedding)
        return output