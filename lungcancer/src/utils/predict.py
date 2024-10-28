# predict.py
import sys
import torch
import torch.nn as nn

# Define the same model architecture as used during training
class LungCancerRiskModel(nn.Module):
    def __init__(self):
        super(LungCancerRiskModel, self).__init__()
        self.fc1 = nn.Linear(21, 64)   # Adjust input size as needed
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 3)    # Output for three classes: Low, Medium, High

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load the model
model = LungCancerRiskModel()
model.load_state_dict(torch.load(r'C:\Users\monti\OneDrive\Área de Trabalho\UNI\Lung_Cancer_Prediction\lungcancer\src\utils\model.pth'))
model.eval()

# Get inputs from command-line arguments
input_data = list(map(float, sys.argv[1:22]))  # Adjust to the number of inputs
inputs = torch.tensor([input_data], dtype=torch.float32)

# Make prediction
with torch.no_grad():
    output = model(inputs)
    prediction = torch.argmax(output, dim=1).item()

# Map output to risk levels
risk_levels = {0: "Low", 1: "Medium", 2: "High"}
print(risk_levels[prediction])
