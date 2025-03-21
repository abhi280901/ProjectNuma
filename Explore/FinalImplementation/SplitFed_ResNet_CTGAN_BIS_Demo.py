import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch                              # PyTorch for tensor operations and deep learning
from torch import nn                      # Neural network module
from torch.utils.data import DataLoader, Dataset  # Data handling utilities
import torch.nn.functional as F           # Functional interface for activation functions, etc.
import numpy as np                        # For numerical operations
import pandas as pd                       # For data manipulation
import random                             # For random seeding
import math                               # For mathematical operations (e.g. sqrt)
import copy                               # For deep copying models
import os                                 # For file operations


# For additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix,accuracy_score
import time
import warnings 
warnings.filterwarnings('ignore') 

input_dim = 5        # Number of features (should match your dataset)
hidden_dim = 128      # Hidden representation dimension
output_dim = 2        # Number of output classes (binary classification)
num_blocks = 5        # Number of residual blocks
weight_decay = 1e-4

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x                        # Save input for skip connection
        out = self.block(x)                 # Pass through sequential layers
        out += residual                     # Add residual connection
        out = self.relu(out)                # Apply ReLU activation
        return out


# =============================================================================
# 2. Define the Client-Side FraudResNet Model
# This model consists of the initial projection and residual blocks.
# It takes the tabular input (of dimension input_dim) and outputs a hidden feature vector.
# =============================================================================
class FraudResNetClient(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks=3):
        super(FraudResNetClient, self).__init__()
        # Initial projection layer: maps input to hidden_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        # Residual blocks: stack of num_blocks residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
    def forward(self, x):
        x = self.input_layer(x)             # Project input to hidden_dim space
        x = self.res_blocks(x)              # Apply residual blocks
        return x     


# =============================================================================
# 3. Define the Server-Side FraudResNet Model
# This model consists solely of the final classification layer.
# It receives the hidden feature vector from the client model.
# =============================================================================
class FraudResNetServer(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(FraudResNetServer, self).__init__()
        self.output_layer = nn.Linear(hidden_dim, output_dim)  # Classification layer
    def forward(self, x):
        x = self.output_layer(x)            # Compute logits
        return x


# Define a dataset that includes features and labels
class NewDataDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        # Return both features and label
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# Load your new data (ensure that the data has the same number of features as the training data)
new_data_df = pd.read_csv("demo_data.csv")
# Assuming new_data_df includes both features and label column
# For example, if 'label' is the column name for the ground truth:
X_new = new_data_df.drop('laundering_schema_type', axis=1).values  
y_new = new_data_df['laundering_schema_type'].values

# Create dataset and dataloader
new_dataset = NewDataDataset(X_new, y_new)
new_loader = DataLoader(new_dataset, batch_size=128, shuffle=False)

# Re-instantiate models with the same architecture
net_client = FraudResNetClient(input_dim, hidden_dim, num_blocks=num_blocks)
net_server = FraudResNetServer(hidden_dim, output_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net_client.to(device)
net_server.to(device)

# Load the best model weights
net_client.load_state_dict(
    torch.load("best_client_model_CTGAN_ResNet_V2_wo_drop_128_005.pth", map_location=torch.device('cpu'))
)
net_server.load_state_dict(
    torch.load("best_server_model_CTGAN_ResNet_V2_wo_drop_128_005.pth", map_location=torch.device('cpu'))
)

# Set models to evaluation mode
net_client.eval()
net_server.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for features, labels in new_loader:
        features = features.to(device)
        labels = labels.to(device)
        # Pass through client model to get features
        client_out = net_client(features)
        # Pass the features through the server model to get logits
        logits = net_server(client_out)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate Accuracy and print a classification report
acc = accuracy_score(all_labels, all_preds)
print("Accuracy on new data: {:.2f}%".format(acc * 100))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("\nPredictions:")
random_indices = random.sample(range(len(all_preds)), min(5, len(all_preds)))
for i, idx in enumerate(random_indices):
    label = "Fraud" if all_preds[idx] == 1 else "Legit"
    print(f"Transaction {idx+1}: {label}")