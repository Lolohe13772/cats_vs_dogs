import matplotlib.pyplot as plt
import json
import os

metrics_path = "outputs/metrics.json"

if not os.path.exists(metrics_path):
    print(f"Le fichier {metrics_path} n'existe pas encore. Lance d'abord l'entraînement pour générer les métriques.")
    exit()

with open(metrics_path, "r") as f:
    metrics = json.load(f)

epochs = range(1, len(metrics["train_loss"]) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, metrics["train_loss"], label="Train Loss")
plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
plt.title("Loss par époque")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, metrics["train_acc"], label="Train Accuracy")
plt.plot(epochs, metrics["val_acc"], label="Validation Accuracy")
plt.title("Accuracy par époque")
plt.xlabel("Époque")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/metrics_plot.png")
plt.show()
