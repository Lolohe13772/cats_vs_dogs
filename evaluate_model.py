import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Config
data_dir = 'data/val'  # dossier validation
model_path = 'cats_vs_dogs_resnet18.pth'
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations (même que pour val)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset et loader
val_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Classes
classes = val_dataset.classes

# Charge modèle (idem que pour training)
from torchvision.models import resnet18, ResNet18_Weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Matrice de confusion et rapport classification
cm = confusion_matrix(all_labels, all_preds)
cr = classification_report(all_labels, all_preds, target_names=classes)
accuracy = accuracy_score(all_labels, all_preds)

print(f"Accuracy globale: {accuracy:.4f}")
print("\nMatrice de confusion:")
print(cm)
print("\nRapport de classification:")
print(cr)

# Plot matrice de confusion
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.show()

# Sauvegarde rapport classification dans un fichier
with open('classification_report.txt', 'w') as f:
    f.write(f"Accuracy globale: {accuracy:.4f}\n\n")
    f.write("Matrice de confusion:\n")
    f.write(np.array2string(cm))
    f.write("\n\nRapport de classification:\n")
    f.write(cr)
