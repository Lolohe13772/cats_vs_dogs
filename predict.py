import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
import sys

# Config
model_path = 'cats_vs_dogs_resnet18.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations (mêmes que pour val)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Chargement du modèle
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes : cats et dogs
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

classes = ['Cat', 'Dog']

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    with torch.no_grad():
        out = model(batch_t)
        _, pred = torch.max(out, 1)
        prob = torch.nn.functional.softmax(out, dim=1)[0][pred].item()
    
    print(f"Image: {image_path} -> Prediction: {classes[pred]} (probabilité: {prob:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <image_path> OR <folder_path>")
        sys.exit(1)

    path = sys.argv[1]
    if os.path.isdir(path):
        for img_file in os.listdir(path):
            full_path = os.path.join(path, img_file)
            try:
                predict_image(full_path)
            except Exception as e:
                print(f"Erreur avec l'image {full_path}: {e}")
    elif os.path.isfile(path):
        predict_image(path)
    else:
        print(f"Le chemin {path} n'existe pas.")
