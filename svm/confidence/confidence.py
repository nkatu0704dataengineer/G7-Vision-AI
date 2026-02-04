import numpy as np
from sklearn.svm import SVC

class DiscriminativeConfidenceModule:
    """
    Mô-đun tính toán độ tin cậy phân biệt dựa trên hình học siêu phẳng.
    TUYỆT ĐỐI KHÔNG sử dụng Platt Scaling, Sigmoid hoặc đầu ra xác suất. [Chỉ thị]
    """
    def __init__(self, kernel='linear', C=1.0):
        # Khởi tạo SVM với tham số mặc định không xác suất
        self.svm = SVC(kernel=kernel, C=C, probability=False)
        self.is_fitted = False

    def learn_boundaries(self, X, y):
        """
        Học ranh giới giữa các định danh để xác định các siêu phẳng tối ưu.
        y chỉ dùng để xác định cấu trúc hình học, không dùng để dự đoán nhãn.
        """
        self.svm.fit(X, y)
        self.is_fitted = True

    def compute_confidence_scores(self, X):
        """
        Tính toán chỉ số tin cậy dựa trên khoảng cách hình học.
        Trả về điểm số định chuẩn trong đoạn [1].
        """
        if not self.is_fitted:
            raise RuntimeError("Mô-đun cần được học ranh giới trước khi tính toán.")

        # 1. Tính giá trị hàm quyết định (Signed Distances)
        # f(x) = sum(alpha_i * y_i * K(x_i, x)) + b [Tài liệu 115, 122]
        signed_distances = self.svm.decision_function(X)

        # 2. Tính Khoảng cách tuyệt đối đến siêu phẳng
        # Trong không gian đặc trưng, khoảng cách càng lớn, đặc trưng càng ít nhiễu [Tài liệu 531, 532]
        abs_distances = np.abs(signed_distances)

        # 3. Định chuẩn hóa (Normalization) sang thang đo [1]
        # Sử dụng Min-Max scaling để tạo ra điểm tin cậy tương đối [Tài liệu 211]
        # Lưu ý: Đây là phép biến đổi tuyến tính, không phải hàm Sigmoid phi tuyến.
        min_d = np.min(abs_distances, axis=0)
        max_d = np.max(abs_distances, axis=0)
        
        # Tránh chia cho 0 nếu tất cả khoảng cách bằng nhau
        denom = (max_d - min_d)
        denom[denom == 0] = 1.0
        
        confidence_scores = (abs_distances - min_d) / denom

        return {
            'signed_distances': signed_distances,
            'absolute_distances': abs_distances,
            'confidence_scores': confidence_scores
        }