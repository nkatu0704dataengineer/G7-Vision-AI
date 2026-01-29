# cnn/extract_feature.py
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

from cnn.model import FaceCNN
from cnn.config import Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = FaceCNN(
    embedding_dim=Config.EMBEDDING_DIM,
    num_classes=Config.NUM_CLASSES
)
model.load_state_dict(torch.load("face_cnn.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


def extract_feature(image):
    """
    input: image (PIL.Image or OpenCV image)
    output: 1D numpy array (embedding)
    """

    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image, return_embedding=True)

    return embedding.cpu().numpy().flatten()
