import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import os

# Config
model_path = 'cats_vs_dogs_resnet18.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
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
    return image, classes[pred], prob

def display_predictions(folder_path, max_images=10):
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    img_files = img_files[:max_images]

    plt.figure(figsize=(15, 5))
    for i, img_file in enumerate(img_files, 1):
        img_path = os.path.join(folder_path, img_file)
        try:
            image, pred_class, prob = predict_image(img_path)
            plt.subplot(1, max_images, i)
            plt.imshow(image)
            plt.title(f"{pred_class}\n({prob:.2f})")
            plt.axis('off')
        except Exception as e:
            print(f"Erreur sur {img_file}: {e}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 predict_batch.py <folder_path>")
    else:
        display_predictions(sys.argv[1])
