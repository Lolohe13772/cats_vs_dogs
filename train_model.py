import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import json
import os

# Config
train_dir = 'data/train'
val_dir = 'data/val'
batch_size = 32
num_epochs = 2
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def filter_corrupted_images(dataset):
    valid_samples = []
    for path, label in dataset.samples:
        try:
            img = Image.open(path)
            img.verify()  # Vérifie si l'image est valide
            valid_samples.append((path, label))
        except (UnidentifiedImageError, IOError, SyntaxError):
            print(f"Fichier corrompu ignoré : {path}")
    dataset.samples = valid_samples
    dataset.targets = [s[1] for s in valid_samples]
    return dataset

# Chargement datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataset = filter_corrupted_images(train_dataset)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Classes
classes = train_dataset.classes

# Modèle
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
model = model.to(device)

# Critère et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Pour sauvegarder métriques
metrics = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for inputs, labels in loop:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total += inputs.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total
    metrics["train_loss"].append(epoch_loss)
    metrics["train_acc"].append(epoch_acc.item())

    # Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            val_total += inputs.size(0)

    val_epoch_loss = val_loss / val_total
    val_epoch_acc = val_corrects.double() / val_total
    metrics["val_loss"].append(val_epoch_loss)
    metrics["val_acc"].append(val_epoch_acc.item())

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train loss: {epoch_loss:.4f}, Train acc: {epoch_acc:.4f} | "
          f"Val loss: {val_epoch_loss:.4f}, Val acc: {val_epoch_acc:.4f}")

# Sauvegarde du modèle entraîné
torch.save(model.state_dict(), 'cats_vs_dogs_resnet18.pth')
print("Modèle sauvegardé.")

# Sauvegarde des métriques dans un fichier JSON
os.makedirs('outputs', exist_ok=True)
with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Métriques sauvegardées dans outputs/metrics.json")
