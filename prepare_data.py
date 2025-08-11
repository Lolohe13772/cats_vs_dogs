import os
import shutil
import random

# Chemin vers le dossier PetImages
source_dir = 'PetImages'

# Dossier de destination pour organiser train/val/test
base_dir = 'data'

# Sous-dossiers à créer
folders = ['train/cats', 'train/dogs', 'val/cats', 'val/dogs', 'test/cats', 'test/dogs']

# Crée les dossiers s'ils n'existent pas
for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

# Fonction pour faire le split et copier les images
def split_data(source_folder, train_folder, val_folder, test_folder, split_ratio=(0.8, 0.1, 0.1)):
    images = os.listdir(source_folder)
    images = [img for img in images if img.endswith('.jpg') or img.endswith('.png')]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    for i, img in enumerate(images):
        src = os.path.join(source_folder, img)
        if i < train_end:
            dst = os.path.join(train_folder, img)
        elif i < val_end:
            dst = os.path.join(val_folder, img)
        else:
            dst = os.path.join(test_folder, img)
        shutil.copyfile(src, dst)

# Appliquer la fonction pour cats et dogs
split_data(os.path.join(source_dir, 'Cats'),
           os.path.join(base_dir, 'train/cats'),
           os.path.join(base_dir, 'val/cats'),
           os.path.join(base_dir, 'test/cats'))

split_data(os.path.join(source_dir, 'Dogs'),
           os.path.join(base_dir, 'train/dogs'),
           os.path.join(base_dir, 'val/dogs'),
           os.path.join(base_dir, 'test/dogs'))

print("Organisation terminée !")

