# image_checker.py
import cv2

def is_blurry(image, threshold=100.0):
    """
    Kiểm tra ảnh có bị mờ hay không bằng Laplacian
    Trả về:
    - True  : ảnh mờ
    - False : ảnh rõ
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_score < threshold
