#cnn/model.py (Định nghĩa kiến trúc - Bản "Chay" & "Tối ưu")

import torch
import torch.nn as nn
import torchvision.models as models

class FaceCNN(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=4):
        super(FaceCNN, self).__init__()
        
        # 1. Kế thừa VGG16 (Backbone trích xuất đặc trưng)
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = vgg.features # Giữ lại các lớp Convolution
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # 2. Lớp chuyển đổi sang Vector Embedding (Phần "chay" chi tiết)
        self.flatten = nn.Flatten()
        self.fc_embedding = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, embedding_dim) # Đây là Vector 128D cho SVM/KNN
        )
        
        # 3. Lớp phân loại (Dùng để huấn luyện CNN ban đầu)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x, return_embedding=False):
        # Trích xuất đặc trưng không gian
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        
        # Tạo vector embedding
        embedding = self.fc_embedding(x)
        
        if return_embedding:
            return embedding # Trả về vector để Người 2 dùng SVM/Logic mờ
            
        # Trả về kết quả phân loại (Softmax) để tính Loss khi train
        output = self.classifier(embedding)
        return output