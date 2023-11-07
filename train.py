import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Net

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')
X_test = np.load('data/X_test.npy')

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(784, 26).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 50

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        data = data.view(data.size(0), -1)

        optimizer.zero_grad()
        output = model(data)

        target = target.float()

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

def validate():
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target_onehot in val_loader:
            data = data.to(device)
            target = torch.argmax(target_onehot, dim=1).to(device)

            data = data.view(data.size(0), -1)

            output = model(data)
            val_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(val_loss, correct, len(val_loader.dataset),
                  100. * correct / len(val_loader.dataset)))


for epoch in range(1, epochs + 1):
    train(epoch)
    validate()

torch.save(model.state_dict(), "model.pt")


import random

model.eval()

fig, axs = plt.subplots(1, 5, figsize=(12, 3))
class_to_letter = {i: chr(65 + i) for i in range(26)}

for i in range(5):
    example_idx = random.randint(0, len(X_test) - 1)
    data = X_test[example_idx].view(1, -1).to(device)
    output = model(data)

    predicted_class = output.argmax()
    predicted_letter = class_to_letter[predicted_class.item()]

    axs[i].imshow(data.view(28, 28).cpu().numpy(), cmap='gray')
    axs[i].set_title(f'Predicted: {predicted_letter}')

plt.show()


