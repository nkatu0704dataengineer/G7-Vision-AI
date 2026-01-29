import numpy as np
import pickle
import os

class FaceEmbeddingDatabase:
    """
    Module quản lý lưu trữ vector đặc trưng (embeddings) phục vụ nghiên cứu nhận diện khuôn mặt [1].
    Sử dụng NumPy để quản lý dữ liệu số và Pickle để đóng gói dữ liệu định danh [2, 3].
    """
    def __init__(self, storage_path="face_db.pkl"):
        self.storage_path = storage_path
        self.embeddings = []  # Danh sách các vector NumPy (CNN output) [4]
        self.labels = []      # Danh sách nhãn danh tính tương ứng [3]
        self._initialize_db()

    def _initialize_db(self):
        """Khởi tạo hoặc tải dữ liệu hiện có từ file [5]."""
        if os.path.exists(self.storage_path):
            self.load_database()
        else:
            self.embeddings = []
            self.labels = []

    def enroll(self, embedding, identity_label):
        """
        Đăng ký (Enroll) vector embedding mới vào cơ sở dữ liệu [6].
        Embedding được đảm bảo ở dạng mảng NumPy 1D phẳng [7].
        """
        # Chuyển đổi và làm phẳng vector embedding từ CNN [8]
        feat_vector = np.array(embedding).flatten()
        
        self.embeddings.append(feat_vector)
        self.labels.append(identity_label)
        
        # Lưu thay đổi ngay lập tức để bảo toàn dữ liệu nghiên cứu [1]
        self.save_database()

    def save_database(self):
        """Lưu trữ toàn bộ cấu trúc dữ liệu xuống file nhị phân [9, 10]."""
        data_to_save = {
            "embeddings": self.embeddings,
            "labels": self.labels
        }
        with open(self.storage_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load_database(self):
        """Tải dữ liệu từ file để phục vụ truy vấn [11, 12]."""
        with open(self.storage_path, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data.get("embeddings", [])
            self.labels = data.get("labels", [])

    def get_all_embeddings(self):
        """
        Truy xuất toàn bộ dữ liệu dưới dạng ma trận NumPy (N, D).
        Phục vụ thuật toán KNN thực hiện quét tuyến tính (Linear Scan) [13, 14].
        """
        if not self.embeddings:
            return np.array([]), np.array([])
            
        # Chuyển đổi danh sách vector thành ma trận 2D để tối ưu tính toán KNN [12, 15]
        return np.array(self.embeddings), np.array(self.labels)

    def clear_database(self):
        """Xóa sạch dữ liệu trong database phục vụ thử nghiệm mới."""
        self.embeddings = []
        self.labels = []
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
