# Cats vs Dogs Classification avec PyTorch

## Description du projet
Ce projet consiste à entraîner un modèle de classification d’images capable de distinguer les chats des chiens à partir d’un dataset d’images.  
Nous utilisons un modèle ResNet18 pré-entraîné sur ImageNet, adapté à notre tâche via fine-tuning.  
Ce projet illustre les concepts de transfert learning, traitement d’images, et évaluation de modèles en deep learning.

## Contenu du dépôt
- `train_model.py` : script d’entraînement du modèle.  
- `evaluate_model.py` : évalue les performances sur un jeu de validation et affiche la matrice de confusion et le rapport de classification.  
- `predict.py` : permet de faire des prédictions sur des images individuelles ou un dossier d’images.  
- `plot_metrics.py` : trace les courbes de loss et d’accuracy pendant l’entraînement à partir des métriques sauvegardées.  
- `cats_vs_dogs_resnet18.pth` : poids sauvegardés du modèle entraîné.  
- `metrics.json` : métriques enregistrées pendant l’entraînement (loss, accuracy).  
- `classification_report.txt` : rapport de classification généré après évaluation.  

## Prérequis
- Python 3.7 ou supérieur  
- GPU recommandé mais pas obligatoire  
- Packages Python (installés via `requirements.txt` ou manuellement) :  
  `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `Pillow`

## Installation
1. Cloner ce dépôt :  
   ```bash
   git clone https://github.com/Lolohe13772/cats_vs_dogs.git
   cd cats_vs_dogs
### Installer les dépendances :
pip install torch torchvision scikit-learn matplotlib seaborn tqdm Pillow
Utilisation
Entraînement du modèle
python3 train_model.py
Évaluation sur le jeu de validation
python3 evaluate_model.py
Prédiction sur image(s)
Prédire sur une image individuelle :
python3 predict.py path/to/image.jpg
Prédire sur un dossier d’images :
python3 predict.py path/to/folder/
Exemple de sortie :
Image: data/val/cats/2668.jpg -> Prediction: Cat (probabilité: 0.82)
Visualisation des métriques
python3 plot_metrics.py
Méthodologie
Fine-tuning du modèle ResNet18 pré-entraîné sur ImageNet.
Prétraitement uniforme des images (redimensionnement, normalisation).
Sauvegarde du modèle et des métriques pendant l’entraînement.
Évaluation via matrice de confusion et rapport détaillé (précision, rappel, F1-score).
Prédictions sur des images pour vérification qualitative.
Résultats
Accuracy globale atteinte : ~95.42% après 2 epochs.
Bon compromis entre précision pour les chats et les chiens.
Rapport détaillé disponible dans classification_report.txt.
Remarques
Pour améliorer les performances, augmenter le nombre d’epochs dans train_model.py.
Le script predict.py permet des tests rapides et manuels.
plot_metrics.py offre une visualisation claire de la progression de l’entraînement.
Auteur
Laurent Henon