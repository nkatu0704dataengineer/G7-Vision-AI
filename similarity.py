import numpy as np

class FaceSimilarity:
    """
    Module cung cấp các phép đo độ tương đồng và khoảng cách cho vector embedding khuôn mặt.
    Thiết kế phục vụ tính toán KNN trong không gian vector cao chiều.
    """

    @staticmethod
    def l2_normalize(vectors):
        """
        Thực hiện chuẩn hóa L2 cho một hoặc nhiều vector.
        Giúp đưa các vector về cùng một mặt cầu đơn vị (radius=1).
        """
        # Đảm bảo đầu vào là mảng numpy
        vectors = np.array(vectors)
        
        # Tính chuẩn L2 dọc theo trục cuối cùng (embedding dimension)
        norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
        
        # Chia cho norm (thêm epsilon 1e-10 để tránh chia cho 0)
        return vectors / (norm + 1e-10)

    @staticmethod
    def cosine_similarity(query_vector, gallery_matrix):
        """
        Tính toán độ tương đồng Cosine giữa 1 query vector và danh sách gallery vectors.
        Kết quả trong khoảng [-1, 1], giá trị càng cao càng tương đồng.
        """
        # Nếu đã chuẩn hóa L2, Cosine Similarity tương đương với tích vô hướng (dot product)
        # Để đảm bảo tính tổng quát, module thực hiện tính toán đầy đủ
        dot_product = np.dot(gallery_matrix, query_vector)
        
        query_norm = np.linalg.norm(query_vector)
        gallery_norms = np.linalg.norm(gallery_matrix, axis=1)
        
        # Cosine similarity formula: (A . B) / (||A|| * ||B||)
        return dot_product / (query_norm * gallery_norms + 1e-10)

    @staticmethod
    def euclidean_distance(query_vector, gallery_matrix):
        """
        Tính toán khoảng cách Euclidean (L2 distance) giữa query và gallery.
        Giá trị càng thấp, hai khuôn mặt càng gần nhau trong không gian đặc trưng.
        """
        # Euclidean distance formula: sqrt(sum((xi - yi)^2))
        return np.linalg.norm(gallery_matrix - query_vector, axis=1)

    @staticmethod
    def compute_all_metrics(query_vector, gallery_matrix, normalize_first=True):
        """
        Hàm tiện ích trả về cả hai phép đo sau khi đã chuẩn hóa (nếu cần).
        """
        if normalize_first:
            q = FaceSimilarity.l2_normalize(query_vector)
            g = FaceSimilarity.l2_normalize(gallery_matrix)
        else:
            q, g = query_vector, gallery_matrix

        return {
            "cosine": FaceSimilarity.cosine_similarity(q, g),
            "euclidean": FaceSimilarity.euclidean_distance(q, g)
        }
