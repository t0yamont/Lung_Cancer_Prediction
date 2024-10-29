# // src/ utils/ predict.py 

import sys
import json
import torch
import torch.nn as nn

class LungCancerRiskModel(nn.Module):
    def __init__(self):
        super(LungCancerRiskModel, self).__init__()
        self.fc1 = nn.Linear(22, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = LungCancerRiskModel()
model.load_state_dict(torch.load(r'C:\Users\monti\OneDrive\Área de Trabalho\UNI\Lung_Cancer_Prediction\lungcancer\src\utils\model.pth'))
model.eval()

# Read JSON input
inputs = json.loads(sys.argv[1])
input_data = torch.tensor([inputs], dtype=torch.float32)

# Make prediction
with torch.no_grad():
    output = model(input_data)
    prediction = torch.argmax(output, dim=1).item()

risk_levels = {0: "Low", 1: "Medium", 2: "High"}
print(risk_levels[prediction])

