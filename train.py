import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_DIR = "data/dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 32
EPOCHS = 20
IMG = 224

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=test_tf)

train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_ld  = DataLoader(test_ds,  batch_size=BATCH)

classes = train_ds.classes
print("Classes:", classes)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, len(classes))
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Menurunkan Learning Rate setiap 7 epoch agar lebih teliti
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Di dalam loop epoch, panggil:
# scheduler.step()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for x, y in train_ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_ld):.4f}")

torch.save({"model": model.state_dict(), "classes": classes}, "model.pth")
print("✅ Model saved -> model.pth")

# TESTING
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = torch.argmax(model(x), 1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n✅ TEST ACCURACY:", round(acc, 4))
print("Confusion Matrix:\n", cm)
