import torch
import os
from cnn.model import FaceCNN
from cnn.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weight_path="face_cnn.pth"):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            "Model chưa được train. Hãy train trước để tạo face_cnn.pth"
        )

    model = FaceCNN().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


def extract_feature(image, model=None):
    """
    input: ảnh (PIL hoặc OpenCV)
    output: vector embedding
    """
    if model is None:
        model = load_model()

    with torch.no_grad():
        embedding = model(image)

    return embedding
