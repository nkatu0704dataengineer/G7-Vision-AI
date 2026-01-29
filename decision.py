import numpy as np
from similarity import FaceSimilarity


class FaceDecisionKNN:
    """
    Module thực hiện quyết định nhận dạng cuối cùng bằng thuật toán KNN.
    Hỗ trợ chiến lược bỏ phiếu đa số (Majority) và trọng số (Weighted),
    tích hợp cơ chế từ chối (Reject Option).
    """

    def __init__(self, k=5, threshold=0.5, strategy='majority'):
        self.k = k
        self.threshold = threshold
        self.strategy = strategy

    def identify(self, query_embedding, gallery_embeddings, labels, metric='cosine'):
        """
        Thực hiện so khớp query với toàn bộ gallery (Linear Scan).
        Trả về: identity_label, confidence_score
        """
        if len(gallery_embeddings) == 0:
            return "Unknown", 0.0

        # ===== BƯỚC 1: TÍNH SIMILARITY / DISTANCE =====
        if metric == 'cosine':
            # Giá trị càng lớn càng giống
            scores = FaceSimilarity.cosine_similarity(
                query_embedding, gallery_embeddings
            )
            top_indices = np.argsort(scores)[::-1][:self.k]
        else:
            # Giá trị càng nhỏ càng gần
            scores = FaceSimilarity.euclidean_distance(
                query_embedding, gallery_embeddings
            )
            top_indices = np.argsort(scores)[:self.k]

        neighbor_labels = labels[top_indices]
        neighbor_scores = scores[top_indices]

        # ===== BƯỚC 2: REJECT OPTION (FIX BUG) =====
        # Lấy láng giềng gần nhất / giống nhất
        best_match_score = neighbor_scores[0]

        if metric == 'cosine':
            if best_match_score < self.threshold:
                return "Unknown", float(best_match_score)
        else:
            if best_match_score > self.threshold:
                return "Unknown", float(best_match_score)

        # ===== BƯỚC 3: DECISION STRATEGY =====
        if self.strategy == 'weighted':
            return self._weighted_voting(
                neighbor_labels, neighbor_scores, metric
            )
        else:
            return self._majority_voting(
                neighbor_labels, neighbor_scores
            )

    def _majority_voting(self, neighbor_labels, neighbor_scores):
        """
        Majority Voting + Mean Similarity
        Confidence = độ tương đồng trung bình của nhãn thắng
        """
        unique_labels, counts = np.unique(
            neighbor_labels, return_counts=True
        )
        winner_idx = np.argmax(counts)
        winner_label = unique_labels[winner_idx]

        # Chỉ lấy score của các neighbor cùng nhãn
        mask = neighbor_labels == winner_label
        confidence = float(np.mean(neighbor_scores[mask]))

        return winner_label, confidence

    def _weighted_voting(self, neighbor_labels, neighbor_scores, metric):
        """
        Weighted Voting:
        - Euclidean: trọng số = 1 / distance
        - Cosine: dùng trực tiếp similarity
        """
        if metric == 'euclidean':
            weights = 1.0 / (neighbor_scores + 1e-10)
        else:
            weights = neighbor_scores

        label_weight_map = {}
        for label, w in zip(neighbor_labels, weights):
            label_weight_map[label] = label_weight_map.get(label, 0.0) + w

        winner_label = max(label_weight_map, key=label_weight_map.get)
        total_weight = sum(weights)
        confidence = label_weight_map[winner_label] / total_weight

        return winner_label, float(confidence)

    def update_parameters(self, k=None, threshold=None, strategy=None):
        """Cập nhật tham số phục vụ thực nghiệm."""
        if k is not None:
            self.k = k
        if threshold is not None:
            self.threshold = threshold
        if strategy is not None:
            self.strategy = strategy

