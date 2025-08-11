import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()  # <-- ajouté pour convertir en tensor
])

data_dir = 'data/train'
dataset = datasets.ImageFolder(data_dir, transform=transform)
class_names = dataset.classes

fig = plt.figure(figsize=(10, 5))
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    img, label = dataset[i]
    img = img.permute(1, 2, 0)  # CHW to HWC for matplotlib
    plt.imshow(img)
    ax.set_title(class_names[label])
    ax.axis('off')

plt.show()
