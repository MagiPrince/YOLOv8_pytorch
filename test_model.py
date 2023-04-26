import torch

# Define a basic PyTorch model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(194*254, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 194*254)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model and define the loss function and optimizer
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Generate some sample data
inputs = torch.randn(32, 194, 254)
labels = torch.randint(0, 10, (32,))

# Train the model for 10 epochs
for epoch in range(10):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print the loss
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")

# Save the trained weights to a file
torch.save(model.state_dict(), "my_model_weights.pth")
