# extract_feature.py

import torch
import os
import cv2
import numpy as np
from PIL import Image

from cnn.model import FaceCNN
from cnn.config import Config
from cnn.dataset import get_train_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weight_path=Config.MODEL_PATH):  # nạp CNN đã train
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            "Model chưa được train. Hãy train trước để tạo face_cnn.pth"
        )

    model = FaceCNN(Config.EMBEDDING_DIM, Config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))  # Load trọng số
    model.eval()
    return model


def extract_feature(image, model=None):
    """
    input: ảnh (PIL hoặc OpenCV)
    output: vector embedding (128D)
    """
    if model is None:
        model = load_model()

    transform = get_train_transform()

    # Nếu ảnh từ OpenCV → BGR → RGB
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

    image = transform(image)                # (3, 224, 224)
    image = image.unsqueeze(0).to(device)   # (1, 3, 224, 224)

    with torch.no_grad():
        embedding = model(image, return_embedding=True)

    return embedding
