import numpy as np
from sklearn.svm import SVC

class FeatureConsistencyModule:
    """
    Mô-đun củng cố tính nhất quán đặc trưng bằng cách phân tích 
    biến bù (slack) và khoảng cách hình học nội lớp.
    """
    def __init__(self, kernel='linear', C=1.0):
        self.svm = SVC(kernel=kernel, C=C, probability=False)
        self.is_fitted = False

    def analyze_consistency(self, X, y):
        """
        Phân tích tính ổn định đặc trưng của từng định danh (y).
        Sử dụng y để học ranh giới, không dùng để dự đoán nhãn [Chỉ thị].
        """
        self.svm.fit(X, y)
        self.is_fitted = True
        
        # 1. Lấy giá trị hàm quyết định f(x) = w*x + b
        # Đối với SVM đa lớp (ovo), decision_function trả về (n_samples, n_classes*(n_classes-1)/2)
        decision_values = self.svm.decision_function(X)
        
        # 2. Tính toán Khoảng cách Hình học (Geometric Distance)
        if self.svm.kernel == 'linear':
            norm_w = np.linalg.norm(self.svm.coef_, axis=1)
            # Chuẩn hóa khoảng cách hình học: d = |f(x)| / ||w|| [Tài liệu 134, 428]
            # Lưu ý: Với đa lớp, việc tính khoảng cách hình học chính xác phức tạp hơn, 
            # ở đây ta lấy trung bình các khoảng cách đến các siêu phẳng liên quan.
            geom_distances = np.abs(decision_values) / np.mean(norm_w)
        else:
            geom_distances = np.abs(decision_values) # RBF không có ||w|| đơn giản [Tài liệu 84]

        # 3. Tính toán Biến bù (Slack Variables xi)
        # xi = max(0, 1 - yi*f(xi)). Các điểm vi phạm biên có xi > 0 [Tài liệu 59, 76]
        # Ta ước lượng mức độ vi phạm biên dựa trên decision values.
        slack_estimates = np.maximum(0, 1 - np.abs(decision_values))

        results = {}
        unique_identities = np.unique(y)
        
        for identity in unique_identities:
            idx = (y == identity)
            class_distances = geom_distances[idx]
            class_slacks = slack_estimates[idx]
            
            # Tính toán các chỉ số nhất quán
            results[identity] = {
                # Phương sai khoảng cách: Độ dao động của đặc trưng quanh siêu phẳng [Chỉ thị]
                'distance_variance': np.var(class_distances),
                
                # Tỷ lệ vi phạm biên: Mức độ chồng lấn/nhiễu của identity này [Tài liệu 421, 435]
                'margin_violation_rate': np.mean(class_slacks > 0),
                
                # Xác định outlier: Các đặc trưng có biến bù cao (lệch xa phân bố ổn định) [Tài liệu 421, 509]
                'outlier_indices': np.where((y == identity) & (np.max(class_slacks, axis=1) > 1.0)).tolist(),
                
                # Độ nén nội lớp (Compactness): Nghịch đảo của tổng biến bù
                'compactness_score': 1.0 / (1.0 + np.sum(class_slacks))
            }
            
        return results