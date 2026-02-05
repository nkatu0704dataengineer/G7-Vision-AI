import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from config import config
from model import CNNBackbone

class FeatureExtractor:
    """
    Module chuyên dụng để trích xuất vector đặc trưng (embeddings) từ ảnh khuôn mặt.
    Sử dụng mô hình CNN đã huấn luyện để phục vụ suy luận (inference).
    """
    def __init__(self, model_path=None):
        self.device = config.DEVICE
        self.model_path = model_path if model_path else config.MODEL_PATH
        
        # Khởi tạo kiến trúc backbone
        self.model = CNNBackbone(embedding_size=config.EMBEDDING_SIZE)
        
        # Tải trọng số đã huấn luyện
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Đã tải thành công mô hình từ: {self.model_path}")
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            
        self.model.to(self.device)
        
        # Thiết lập chế độ đánh giá (Evaluation mode)
        # Vô hiệu hóa Dropout và BatchNorm updates để đảm bảo tính ổn định của embedding [1, 2]
        self.model.eval()
        
        # Định nghĩa quy trình tiền xử lý ảnh đầu vào chuẩn hóa cho backbone [3, 4]
        self.transform = transforms.Compose([
            transforms.Resize(config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_embedding(self, image_source):
        """
        Trích xuất vector embedding từ một ảnh đầu vào.
        Đầu vào: Đường dẫn ảnh (str) hoặc đối tượng ảnh PIL.
        Đầu ra: Numpy array (1D) đại diện cho đặc trưng khuôn mặt phục vụ SVM/KNN.
        """
        # Xử lý nguồn dữ liệu ảnh
        if isinstance(image_source, str):
            image = Image.open(image_source).convert('RGB')
        else:
            image = image_source

        # Tiền xử lý và chuyển thành tensor, thêm batch dimension
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Trích xuất đặc trưng trong chế độ no_grad để tối ưu bộ nhớ và tốc độ [5, 6]
        with torch.no_grad():
            embedding = self.model(img_tensor)

        # Chuyển kết quả về numpy array để dễ dàng tích hợp với các thư viện ML truyền thống (Scikit-Learn)
        return embedding.cpu().numpy().flatten()

    def get_embeddings_batch(self, image_list):
        """
        Trích xuất đặc trưng cho một danh sách các ảnh (batch processing).
        """
        embeddings = []
        for img in image_list:
            embeddings.append(self.get_embedding(img))
        return np.array(embeddings)

# Ví dụ sử dụng trong NCKH
if __name__ == "__main__":
    extractor = FeatureExtractor()
    # feature_vector = extractor.get_embedding("path/to/face.jpg")
    # print(f"Kích thước vector đặc trưng: {feature_vector.shape}")