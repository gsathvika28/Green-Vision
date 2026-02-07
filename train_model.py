import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ---------------- Paths ----------------
DATA_DIR = r"C:\PROJECT HACKATHOn\BPLD Dataset"
train_dir = os.path.join(DATA_DIR, "train")
valid_dir = os.path.join(DATA_DIR, "valid")

# ---------------- Transforms ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ---------------- DataLoaders ----------------
train_data = datasets.ImageFolder(train_dir, transform=transform)
valid_data = datasets.ImageFolder(valid_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False)

class_names = train_data.classes
num_classes = len(class_names)
print("Classes found:", class_names)

# ---------------- Model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights="DEFAULT")
for p in model.parameters():
    p.requires_grad = False
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# ---------------- Training ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.3f}")

# ---------------- Save Model ----------------
torch.save({
    "model_state": model.state_dict(),
    "class_names": class_names
}, "blackgram_multidisease.pth")

print("Training complete. Model saved as blackgram_multidisease.pth")
