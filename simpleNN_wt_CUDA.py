import torch
import torch.nn as nn
import torch.optim as optim
from custom_autograd import custom_softmax  # Import the custom autograd softmax
from custom_autograd import custom_relu  # Import the custom autograd relu
from custom_autograd import CustomLinear
import time
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear1 = CustomLinear(input_size, hidden_size)  # First layer
        self.linear2 = CustomLinear(hidden_size, output_size) # Second layer

    def forward(self, x):
        x = self.linear1(x)
        x = custom_relu(x)  # Use custom CUDA ReLU
        x = self.linear2(x)
        x = custom_softmax(x)
        return x

# Define model parameters
input_size = 10  
hidden_size = 32  
output_size = 3  

# Instantiate and move the model to GPU
model = SimpleNN(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy data for training (Move to GPU)
x_train = torch.randn(100, input_size).to(device)
y_train = torch.randint(0, output_size, (100,)).to(device)

start_time = time.time()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Zero out gradients
    outputs = model(x_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

end_time = time.time()

# Test the model on a single input (Ensure it's on GPU)
x_test = torch.randn(1, input_size).to(device)
output = model(x_test)
predicted_class = torch.argmax(output).item()
print(f"Predicted class: {predicted_class}")


elapsed_time1 = end_time - start_time
print(f"Elapsed time1: {elapsed_time1} sec")


