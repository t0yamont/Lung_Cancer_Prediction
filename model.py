import torch
import torch.nn as nn
import numpy as np

# Define the neural network structure (must match the trained model)
class LungCancerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LungCancerModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# Load the saved model
input_size = 23  # Number of features
num_classes = 3  # 'Low', 'Medium', 'High'
model = LungCancerModel(input_size, num_classes)
model.load_state_dict(torch.load('best_lung_cancer_model.pth'))
model.eval()

# Mappings for categorical variables
gender_mapping = {'Female': 0, 'Male': 1}

# Mapping from model output to risk level
level_mapping = {0: 'High', 1: 'Low', 2: 'Medium'}  # Based on label encoding

# Collect user input
def get_user_input():
    print("Please enter the following information:")
    Age = float(input("Age: "))
    Gender = input("Gender (Male/Female): ").capitalize()
    Gender = gender_mapping.get(Gender, 0)

    # List of features
    features = [
        'Air Pollution',
        'Alcohol use',
        'Dust Allergy',
        'OccuPational Hazards',
        'Genetic Risk',
        'chronic Lung Disease',
        'Balanced Diet',
        'Obesity',
        'Smoking',
        'Passive Smoker',
        'Chest Pain',
        'Coughing of Blood',
        'Fatigue',
        'Weight Loss',
        'Shortness of Breath',
        'Wheezing',
        'Swallowing Difficulty',
        'Clubbing of Finger Nails',
        'Frequent Cold',
        'Dry Cough',
        'Snoring'
    ]

    inputs = [Age, Gender]
    for feature in features:
        while True:
            try:
                value = float(input(f"{feature} (1-8): "))
                if 1 <= value <= 8:
                    inputs.append(value)
                    break
                else:
                    print("Please enter a value between 1 and 8.")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    return inputs

user_input = get_user_input()
user_input = np.array(user_input, dtype=np.float32)
user_input = torch.tensor(user_input).unsqueeze(0)  # Add batch dimension

# Make prediction
with torch.no_grad():
    output = model(user_input)
    _, predicted = torch.max(output.data, 1)
    predicted_level = level_mapping.get(predicted.item(), "Unknown")

print(f"\nThe predicted chance of contracting lung cancer is: {predicted_level}")