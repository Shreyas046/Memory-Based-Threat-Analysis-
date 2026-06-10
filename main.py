import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_data
from mlp_model import MLP
from rf_model import train_rf

# -------- LOAD DATA --------
X_train, X_test, y_train, y_test = load_data("data/malmem.csv")

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

input_size = X_train.shape[1]
num_classes = len(set(y_train.numpy()))

# -------- MODEL --------
model = MLP(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------- TRAIN --------
epochs = 10

for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------- TEST --------
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == y_test).float().mean()
    print(f"MLP Accuracy: {accuracy * 100:.2f}%")

# -------- SAVE MODEL --------
torch.save(model.state_dict(), "mlp_model.pth")

# -------- RANDOM FOREST --------
train_rf(X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy())

print("Model saved successfully")