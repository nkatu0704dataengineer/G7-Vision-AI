import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataset(Dataset):
    """
    Lớp Dataset tùy chỉnh để quản lý dữ liệu ảnh khuôn mặt cho pipeline huấn luyện CNN.
    Hỗ trợ cấu trúc thư mục dạng: root_dir/identity_name/image.jpg
    """
    def __init__(self, root_dir, input_size=224, is_train=True):
        self.root_dir = root_dir
        self.input_size = input_size
        
        # Tự động trích xuất danh sách danh tính từ tên các thư mục con
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_path, img_name), self.class_to_idx[cls_name]))
        
        # Định nghĩa các bước tiền xử lý ảnh (Resize, Augmentation, Normalization)
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        """Trả về tổng số lượng mẫu ảnh có trong tập dữ liệu."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Đọc ảnh từ đĩa, thực hiện tiền xử lý và trả về dưới dạng tensor.
        Returns:
            image (Tensor): Ảnh đã xử lý.
            label (Tensor): Nhãn định danh tương ứng.
        """
        img_path, label = self.samples[idx]
        
        # Đọc ảnh và chuyển đổi sang không gian màu RGB
        image = Image.open(img_path).convert('RGB')
        
        # Áp dụng pipeline biến đổi dữ liệu
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
        """Trả về tổng số lượng danh tính (classes) trong tập dữ liệu."""
        return len(self.classes)