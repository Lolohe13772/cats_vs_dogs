Cats vs Dogs Classification avec PyTorch

Description du projet

Ce projet consiste à entraîner un modèle de classification d’images capable de distinguer les chats des chiens à partir d’un dataset d’images. Nous utilisons un modèle ResNet18 pré-entraîné sur ImageNet, adapté à notre tâche via fine-tuning.
Contenu du dépôt
train_model.py : script d’entraînement du modèle.
evaluate_model.py : évalue les performances sur un jeu de validation et affiche la matrice de confusion et le rapport de classification.
predict.py : permet de faire des prédictions sur des images individuelles ou un dossier d’images.
plot_metrics.py : trace les courbes de loss et d’accuracy pendant l’entraînement à partir des métriques sauvegardées.
cats_vs_dogs_resnet18.pth : poids sauvegardés du modèle entraîné.
metrics.json : métriques enregistrées pendant l’entraînement (loss, accuracy).
classification_report.txt : rapport de classification généré après évaluation.
Installation
Cloner le dépôt
Installer les dépendances :
pip install torch torchvision scikit-learn matplotlib seaborn tqdm
Utilisation
Entraînement
python3 train_model.py
Évaluation
python3 evaluate_model.py
Prédiction sur image(s)
python3 predict.py <chemin_vers_image_ou_dossier>
Visualisation des métriques
python3 plot_metrics.py
Méthodologie
Fine-tuning de ResNet18 pré-entraîné sur ImageNet.
Prétraitement identique pour entraînement et validation (redimensionnement, normalisation).
Sauvegarde du modèle et des métriques pendant l’entraînement.
Évaluation avec matrice de confusion, rapport de classification (precision, recall, f1-score).
Prédictions sur des images individuelles pour validation qualitative.
Résultats
Accuracy globale autour de XX% (ajuster selon ton résultat).
Bon compromis entre précision pour les chats et les chiens.
Les métriques détaillées sont disponibles dans classification_report.txt.
Remarques
Pour des entraînements plus longs, ajuster le nombre d’epochs dans train_model.py.
Le script predict.py permet un test rapide sur des images « à la main ».
plot_metrics.py visualise l’évolution des performances pendant l’entraînement.


## Auteur
Laurent Henon
