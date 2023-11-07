import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from model import Net

# Wczytaj dane testowe
X_test = np.load('data/X_test.npy')
X_test = torch.tensor(X_test, dtype=torch.float32)
test_dataset = TensorDataset(X_test)

batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Wczytaj wytrenowany model
model = Net(784, 26).to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Tworzenie mapowania między klasami a literami
class_to_letter = {i: chr(65 + i) for i in range(26)}

# Inicjalizacja list do zapisu wyników
results = []

# Przetestowanie modelu na zbiorze testowym i zapis wyników
for i in range(len(X_test)):
    data = X_test[i].view(1, -1).to(device)
    output = model(data)

    predicted_class = output.argmax().item()
    predicted_letter = class_to_letter[predicted_class]

    # Zapisz wynik (numer indeksu, przewidywaną literę jako int)
    results.append([i, predicted_class])

# Zapisz wyniki do pliku CSV
results_df = pd.DataFrame(results, columns=["index", "class"])
results_df.to_csv("submission.csv", index=False)

print("Wyniki zostały zapisane do submission.csv.")