import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available and set PyTorch to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate some synthetic data
n_samples = 1000
n_features = 20
X = torch.randn(n_samples, n_features, device=device)  # Random input data
W_true = torch.randn(n_features, 1, device=device)  # Random weights
b_true = torch.randn(1, device=device)  # Random bias
y = X @ W_true + b_true + torch.randn(n_samples, 1, device=device) * 0.1  # Linear relation with some noise


# Define a simple linear model
class LinearModel(nn.Module):
    def __init__(self, in_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


# Initialize the model, loss function, and optimizer
model = LinearModel(n_features).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 1000
for epoch in range(n_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradient buffers
    output = model(X)  # Get the model's predictions
    loss = criterion(output, y)  # Calculate the loss
    loss.backward()  # Backpropagate the errors
    optimizer.step()  # Update the weights

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{n_epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predictions = model(X)
    loss = criterion(predictions, y)
    print(f'Final Loss: {loss.item()}')
