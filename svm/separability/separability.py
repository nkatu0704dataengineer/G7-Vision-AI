import numpy as np
from sklearn.svm import SVC

class FeatureSeparabilityModule:
    """
    Mô-đun củng cố đặc trưng bằng SVM.
    Chỉ trích xuất các chỉ số hình học, KHÔNG phân loại hoặc dự đoán nhãn.
    """
    def __init__(self, kernel='linear', C=1.0):
        # Sử dụng SVC nhưng chỉ để truy cập vào các tham số hình học của siêu phẳng
        self.svm = SVC(kernel=kernel, C=C, probability=False)
        self.margin_width_ = None
        self.support_vector_stats_ = {}
        self.is_fitted = False

    def reinforce_features(self, X, y):
        """
        Huấn luyện SVM để tìm các biên phân tách tối ưu giữa các định danh.
        Dữ liệu nhãn (y) chỉ được dùng để xác định ranh giới hình học.
        """
        self.svm.fit(X, y)
        self.is_fitted = True
        
        # 1. Tính toán Chiều rộng Biên (Margin Width) [1-3]
        # Trong trường hợp tuyến tính, biên = 2 / ||w||
        if self.svm.kernel == 'linear':
            # Với đa lớp, sklearn sử dụng One-Vs-One, ta lấy trung bình các biên
            weights = self.svm.coef_
            margins = 2.0 / np.linalg.norm(weights, axis=1)
            self.margin_width_ = np.mean(margins)
        else:
            # Với RBF/Phi tuyến, margin không có giá trị vô hướng đơn giản như tuyến tính
            self.margin_width_ = None 

        # 2. Thống kê Support Vectors [4-6]
        self.support_vector_stats_ = {
            'total_sv': len(self.svm.support_),
            'sv_per_class': self.svm.n_support_.tolist(),
            'sv_indices': self.svm.support_
        }

        return self.get_separability_indicators(X)

    def get_separability_indicators(self, X):
        """
        Trả về các chỉ số hình học trung gian để chuyển tiếp cho stage so khớp.
        TUYỆT ĐỐI KHÔNG sử dụng svm.predict() [Chỉ thị].
        """
        if not self.is_fitted:
            raise RuntimeError("Mô-đun chưa được huấn luyện trên không gian đặc trưng.")

        # 3. Khoảng cách đến siêu phẳng (Discriminative Confidence Scores) [7, 8]
        # decision_function trả về khoảng cách có dấu từ vector đến siêu phẳng
        # Giá trị tuyệt đối càng lớn, đặc trưng càng nằm xa biên (độ tin cậy cao)
        confidence_distances = self.svm.decision_function(X)

        return {
            'margin_width': self.margin_width_,
            'support_vector_info': self.support_vector_stats_,
            'discriminative_confidence': confidence_distances
        }