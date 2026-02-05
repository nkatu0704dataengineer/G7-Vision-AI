import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class BiometricConfidenceModel:
    def __init__(self):

        # =============================
        # 1. UNIVERSE DEFINITIONS
        # =============================
        similarity = ctrl.Antecedent(np.arange(0, 101, 1), 'similarity')      # 0-100 cosine similarity
        svm_margin = ctrl.Antecedent(np.arange(0, 11, 0.1), 'svm_margin')     # SVM margin thường nhỏ
        knn_dist = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'knn_dist')      # KNN distance 0-1
        quality = ctrl.Antecedent(np.arange(0, 101, 1), 'quality')            # Quality score từ Model 1

        confidence = ctrl.Consequent(np.arange(0, 101, 1), 'confidence')      # Final trust score

        # =============================
        # 2. MEMBERSHIP FUNCTIONS
        # =============================

        # ---- Similarity ----
        similarity['low'] = fuzz.trapmf(similarity.universe, [0, 0, 30, 50])
        similarity['medium'] = fuzz.trimf(similarity.universe, [40, 60, 80])
        similarity['high'] = fuzz.trapmf(similarity.universe, [70, 85, 100, 100])

        # ---- SVM Margin ----
        svm_margin['weak'] = fuzz.trapmf(svm_margin.universe, [0, 0, 2, 4])
        svm_margin['medium'] = fuzz.trimf(svm_margin.universe, [3, 5.5, 8])
        svm_margin['strong'] = fuzz.trapmf(svm_margin.universe, [7, 9, 10, 10])

        # ---- KNN Distance ----
        knn_dist['close'] = fuzz.trapmf(knn_dist.universe, [0, 0, 0.2, 0.4])
        knn_dist['medium'] = fuzz.trimf(knn_dist.universe, [0.3, 0.5, 0.7])
        knn_dist['far'] = fuzz.trapmf(knn_dist.universe, [0.6, 0.8, 1, 1])

        # ---- Image Quality ----
        quality['poor'] = fuzz.trapmf(quality.universe, [0, 0, 30, 50])
        quality['acceptable'] = fuzz.trimf(quality.universe, [40, 60, 80])
        quality['good'] = fuzz.trapmf(quality.universe, [70, 85, 100, 100])

        # ---- Confidence Output ----
        confidence['reject'] = fuzz.trapmf(confidence.universe, [0, 0, 25, 40])
        confidence['low'] = fuzz.trimf(confidence.universe, [30, 45, 60])
        confidence['medium'] = fuzz.trimf(confidence.universe, [55, 70, 85])
        confidence['high'] = fuzz.trapmf(confidence.universe, [80, 90, 100, 100])

        # =============================
        # 3. RULE BASE
        # =============================

        rules = [

            # Strong positive case
            ctrl.Rule(similarity['high'] & quality['good'] & svm_margin['strong'], confidence['high']),

            # Good similarity but bad quality → downgrade
            ctrl.Rule(similarity['high'] & quality['poor'], confidence['low']),

            # KNN very far → reject
            ctrl.Rule(knn_dist['far'], confidence['reject']),

            # Medium similarity + good quality
            ctrl.Rule(similarity['medium'] & quality['good'], confidence['medium']),

            # Weak classifier + low similarity
            ctrl.Rule(svm_margin['weak'] & similarity['low'], confidence['reject']),

            # Good similarity but weak SVM
            ctrl.Rule(similarity['high'] & svm_margin['weak'], confidence['medium']),

            # Medium everything
            ctrl.Rule(similarity['medium'] & quality['acceptable'] & svm_margin['medium'], confidence['medium']),

            # Low similarity no matter what
            ctrl.Rule(similarity['low'], confidence['reject']),

            # Good quality but medium similarity
            ctrl.Rule(quality['good'] & similarity['medium'], confidence['medium']),

            # Close KNN + good similarity
            ctrl.Rule(knn_dist['close'] & similarity['high'], confidence['high'])
        ]

        # =============================
        # 4. BUILD CONTROL SYSTEM
        # =============================
        self.conf_ctrl = ctrl.ControlSystem(rules)
        self.conf_sim = ctrl.ControlSystemSimulation(self.conf_ctrl)

    # =============================
    # 5. INFERENCE FUNCTION
    # =============================
    def compute_confidence(self, sim, svm, knn, q_score):

        # Clip để tránh crash khi input out-range
        self.conf_sim.input['similarity'] = np.clip(sim * 100, 0, 100)   # Convert 0-1 → 0-100
        self.conf_sim.input['svm_margin'] = np.clip(svm, 0, 10)
        self.conf_sim.input['knn_dist'] = np.clip(knn, 0, 1)
        self.conf_sim.input['quality'] = np.clip(q_score, 0, 100)

        self.conf_sim.compute()
        score = self.conf_sim.output['confidence']

        # Label mapping
        if score >= 85:
            label = "High Trust"
        elif score >= 60:
            label = "Medium Trust"
        elif score >= 40:
            label = "Low Trust"
        else:
            label = "Reject"

        return {
            "confidence_score": round(score, 2),
            "confidence_label": label
        }
