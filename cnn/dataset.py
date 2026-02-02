# cnn/dataset.py
# Cầu nối giữa ảnh & CNN
import os
from torch.utils.data import Dataset
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform      # Dataset ghi nhớ cách xử lý ảnh để áp dụng khi cần
        self.samples = []   #Chứa ảnh

        self.class_to_idx = {   # Gán label
            name: idx for idx, name in enumerate(sorted(os.listdir(root_dir)))
        }

        for class_name, label in self.class_to_idx.items(): # Phân loại ảnh
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                self.samples.append(
                    (os.path.join(class_dir, img_name), label)
                )

    def __len__(self):  #Số lượng ảnh
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]     # Lấy đường dẫn + label
        image = Image.open(img_path).convert("RGB")     # Mở ảnh

        if self.transform:      # Chuyển ảnh thành ma trận số
            image = self.transform(image)

        return image, label
from torchvision import transforms

def get_train_transform():      
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])