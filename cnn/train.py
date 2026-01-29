import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from cnn.model import FaceCNN
from cnn.config import Config
from cnn.dataset import FaceDataset, get_train_transform

def train_model(train_loader, val_loader):
    device = torch.device(Config.DEVICE)
    model = FaceCNN(Config.EMBEDDING_DIM, Config.NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{Config.EPOCHS}, "
              f"Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"Đã lưu mô hình tại {Config.MODEL_PATH}")

# ✅ HÀM NÀY ĐỂ run_train.py GỌI
def train():
    train_dataset = FaceDataset(
        Config.DATASET_PATH,
        transform=get_train_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = None  # tạm thời

    train_model(train_loader, val_loader)

if __name__ == "__main__":
    train()
