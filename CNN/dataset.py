# cnn/dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        self.class_to_idx = {
            name: idx for idx, name in enumerate(sorted(os.listdir(root_dir)))
        }

        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                self.samples.append(
                    (os.path.join(class_dir, img_name), label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label