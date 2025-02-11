# client.py
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import flwr as fl

# ---------------------------
# 1. Environment and Device Setup
# ---------------------------
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if len(sys.argv) < 3:
    print("Usage: python client.py <client_id> <data_file>")
    sys.exit(1)

client_id = sys.argv[1]
data_file = sys.argv[2]
print(f"Client {client_id} using data file: {data_file}")

# ---------------------------
# 2. Data Loading and Preprocessing
# ---------------------------
# Expected columns: ['step', 'amount', 'type_cat', 'account_numeric_orig', 'account_numeric_dest', 'isFraud']
df = pd.read_csv(data_file)
print(df.head())

# Save the original account columns for graph construction.
accounts_orig = df['account_numeric_orig'].values
accounts_dest = df['account_numeric_dest'].values

# One-hot encode the categorical variable 'type_cat'
# df = pd.get_dummies(df, columns=['type_cat'], drop_first=True)

# Define feature columns (all except the label 'isFraud')
feature_cols = [col for col in df.columns if col != 'isFraud']
X = df[feature_cols].values
y = df['isFraud'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert features and labels to tensors and send to device.
x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)
num_nodes = x_tensor.shape[0]

# ---------------------------
# 3. Graph Construction
# ---------------------------
# Build edges by connecting transactions (nodes) that share the same originating or destination account.
edge_index_list = []

# Shared originating accounts
orig_dict = defaultdict(list)
for i, account in enumerate(accounts_orig):
    orig_dict[account].append(i)
for indices in orig_dict.values():
    if len(indices) > 1:
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                edge_index_list.append((indices[i], indices[j]))
                edge_index_list.append((indices[j], indices[i]))

# Shared destination accounts
dest_dict = defaultdict(list)
for i, account in enumerate(accounts_dest):
    dest_dict[account].append(i)
for indices in dest_dict.values():
    if len(indices) > 1:
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                edge_index_list.append((indices[i], indices[j]))
                edge_index_list.append((indices[j], indices[i]))

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)

# Create train, validation, and test masks (e.g., 60%, 20%, 20%)
indices = torch.randperm(num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_end = int(0.7 * num_nodes)
val_end = int(0.15 * num_nodes)
train_mask[indices[:train_end]] = True
val_mask[indices[train_end:val_end]] = True
test_mask[indices[val_end:]] = True

# Create a PyTorch Geometric Data object.
data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# ---------------------------
# 4. Define the Split GCN Model
# ---------------------------
# Client-side model: processes graph data (node features and structure) via GCN layers.
class ClientGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.5):
        super(ClientGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        # First layer: in_channels -> hidden_channels.
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        # Additional layers: hidden_channels -> hidden_channels.
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # This is the "smashed data" sent to the server.

# Server-side model: receives the smashed embeddings and performs classification.
class ServerGCN(nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout=0.5):
        super(ServerGCN, self).__init__()
        # For simplicity, we use a fully connected network on the smashed data.
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x

# Set Parameters
learning_rate = 0.005
weight_decay = 1e-4
num_layers = 2
gamma = 2
alpha = 0.75
dropout = 0.2
epoch = 50

# Set dimensions.
in_channels = x_tensor.shape[1]
hidden_channels = 64
out_channels = 2  # Binary classification: non-fraud vs. fraud

client_model = ClientGCN(in_channels, hidden_channels, num_layers=num_layers, dropout=dropout).to(device)
server_model = ServerGCN(hidden_channels, out_channels, dropout=dropout).to(device)


# ---------------------------
# 5. Define Focal Loss
# ---------------------------
# Focal loss down-weights well-classified examples to focus training on hard examples.
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.7, reduction='mean'):
        """
        Args:
            gamma (float): Focusing parameter (default 2).
            alpha (float or list): If a single float is provided, then the weight vector is [1 - alpha, alpha].
                                     Here, setting alpha=0.75 means the negative class gets weight 0.25 and the
                                     positive class gets weight 0.75.
            reduction (str): 'mean' or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Compute per-sample cross-entropy loss (without reduction).
        logpt = -F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)
        
        # If alpha is provided, create a weight vector.
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # For binary classification, weight vector: [1 - alpha, alpha]
                alpha = torch.tensor([1 - self.alpha, self.alpha]).to(inputs.device)
            elif isinstance(self.alpha, list):
                alpha = torch.tensor(self.alpha).to(inputs.device)
            at = alpha.gather(0, targets.data)
            logpt = logpt * at
        
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Here we choose alpha so that the positive class (fraud) receives more weight.
# You may need to tune these parameters further.
# For example, if the imbalance is severe, you might choose alpha=0.7 (meaning 70% weight for positives).
focal_loss = FocalLoss(gamma=gamma, alpha=alpha, reduction='mean')



# # Compute class weights from the training nodes to help with imbalanced data.
# train_y = data.y[data.train_mask]
# num_pos = (train_y == 1).sum().item()
# num_neg = (train_y == 0).sum().item()
# print(f"Training set - Positives: {num_pos}, Negatives: {num_neg}")
# class_weight = torch.tensor([1.0, float(num_neg) / float(num_pos)]).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weight)

optimizer = optim.Adam(
    list(client_model.parameters()) + list(server_model.parameters()),
    lr=learning_rate,
    weight_decay=weight_decay
)



# ---------------------------
# 5. Define Local Training and Evaluation Functions
# ---------------------------
def train_local(epochs=10):
    client_model.train()
    server_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass: client-side.
        smashed = client_model(data.x, data.edge_index)
        # Forward pass: server-side.
        outputs = server_model(smashed)
        # loss = criterion(outputs[data.train_mask], data.y[data.train_mask])
        loss = focal_loss(outputs[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f"Local Epoch {epoch+1}, Loss: {loss.item():.4f}")


def test_local():
    client_model.eval()
    server_model.eval()
    with torch.no_grad():
        smashed = client_model(data.x, data.edge_index)
        outputs = server_model(smashed)
        # loss = criterion(outputs[data.test_mask], data.y[data.test_mask]).item()
        loss = focal_loss(outputs[data.test_mask], data.y[data.test_mask]).item()
        preds = outputs.argmax(dim=1)
        correct = preds[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        total = int(data.test_mask.sum())
        accuracy = correct / total
        # Additional metrics.
        all_labels = data.y[data.test_mask].cpu().numpy()
        all_preds = preds[data.test_mask].cpu().numpy()
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        print("Classification Report:")
        print(classification_report(all_labels, all_preds))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
    return loss, accuracy, precision, recall, f1

# ---------------------------
# 6. Implement the Flower Client for Split Federated Learning
# ---------------------------
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # Return parameters from both client and server models.
        client_params = [val.detach().cpu().numpy() for val in client_model.parameters()]
        server_params = [val.detach().cpu().numpy() for val in server_model.parameters()]
        return client_params + server_params

    def set_parameters(self, parameters):
        num_client_params = len(list(client_model.parameters()))
        client_params = parameters[:num_client_params]
        server_params = parameters[num_client_params:]
        for param, new_param in zip(client_model.parameters(), client_params):
            param.data = torch.tensor(new_param, device=device)
        for param, new_param in zip(server_model.parameters(), server_params):
            param.data = torch.tensor(new_param, device=device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_local(epochs=epoch)
        return self.get_parameters(config={}), int(data.train_mask.sum()), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, precision, recall, f1 = test_local()
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        return float(loss), int(data.test_mask.sum()), metrics

# ---------------------------
# 7. Start the Flower Client
# ---------------------------
if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
