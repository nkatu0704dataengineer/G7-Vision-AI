# clear_image_model.py
import numpy as np

def cnn_recognition(image):
    """
    CNN trích xuất đặc trưng và dự đoán
    (giả lập xác suất nhận diện)
    """
    return np.random.uniform(0.75, 1.0)

def svm_recognition(feature):
    """
    SVM phân loại dựa trên đặc trưng
    """
    return np.random.uniform(0.7, 0.95)

def knn_recognition(feature):
    """
    KNN dựa trên khoảng cách
    """
    return np.random.uniform(0.65, 0.9)

def clear_image_pipeline(image):
    """
    Pipeline xử lý ảnh rõ:
    CNN → SVM → KNN (ensemble đơn giản)
    """
    cnn_score = cnn_recognition(image)

    # Giả sử feature đã được trích xuất từ CNN
    feature = cnn_score

    svm_score = svm_recognition(feature)
    knn_score = knn_recognition(feature)

    # Kết hợp điểm (trung bình)
    final_score = (cnn_score + svm_score + knn_score) / 3
    return final_score
