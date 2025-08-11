import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import json  # Pour sauvegarder les métriques
import os

# Config
data_dir = 'data/train'  # dossier train
val_dir = 'data/val'     # dossier val
batch_size = 32
num_epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset & Dataloader
train_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Pour stocker les métriques
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    history["train_loss"].append(epoch_loss)
    history["train_acc"].append(epoch_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data).item()
            val_total += inputs.size(0)

    val_epoch_loss = val_loss / val_total
    val_epoch_acc = val_corrects / val_total

    history["val_loss"].append(val_epoch_loss)
    history["val_acc"].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train loss: {epoch_loss:.4f}, Train acc: {epoch_acc:.4f} - "
          f"Val loss: {val_epoch_loss:.4f}, Val acc: {val_epoch_acc:.4f}")

# Sauvegarde du modèle
torch.save(model.state_dict(), 'cats_vs_dogs_resnet18.pth')
print("Modèle sauvegardé.")

# Sauvegarde des métriques dans un fichier JSON
with open('training_history.json', 'w') as f:
    json.dump(history, f)

print("Historique d'entraînement sauvegardé.")