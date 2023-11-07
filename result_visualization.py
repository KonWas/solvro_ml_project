import re
import matplotlib.pyplot as plt

# Inicjalizacja pustych list
train_losses = []
val_losses = []
val_accuracies = []

# Otwarcie pliku z wynikami trenowania
with open('50_epochs_result_training.txt', 'r') as file:
    lines = file.readlines()

# Wyrażenia regularne do wyodrębnienia informacji o błędach i dokładności
train_loss_pattern = re.compile(r'Train Epoch: \d+.*Loss: (\d+\.\d+)')
val_loss_pattern = re.compile(r'Validation set: Average loss: (\d+\.\d+)')
val_accuracy_pattern = re.compile(r'Validation set: Average loss: \d+\.\d+, Accuracy: (\d+)/(\d+)')

# Przechodzenie przez linie w pliku i wyodrębnianie danych
for line in lines:
    train_loss_match = train_loss_pattern.search(line)
    val_loss_match = val_loss_pattern.search(line)
    val_accuracy_match = val_accuracy_pattern.search(line)

    if train_loss_match:
        train_loss = float(train_loss_match.group(1))
        train_losses.append(train_loss)
    if val_loss_match:
        val_loss = float(val_loss_match.group(1))
        val_losses.append(val_loss)
    if val_accuracy_match:
        val_accuracy = int(val_accuracy_match.group(1))
        val_total = int(val_accuracy_match.group(2))
        val_accuracy_percent = (val_accuracy / val_total) * 100  # Konwersja na procenty
        val_accuracies.append(val_accuracy_percent)

# Tworzenie listy epok
epochs = list(range(1, 51))  # Zakładam 50 epok

# Tworzenie wykresów
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses[:50], label='Train Loss')  # Bierze tylko pierwsze 50 wyników
plt.plot(epochs, val_losses[:50], label='Validation Loss')  # Bierze tylko pierwsze 50 wyników
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies[:50], label='Validation Accuracy', color='green')  # Bierze tylko pierwsze 50 wyników
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()