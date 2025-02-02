# import sys
# import flwr as fl
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from torch_geometric.data import Data, DataLoader  # PyTorch Geometric Data and DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # You may need to install PyTorch Geometric and its dependencies.
# # See: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

# #######################################
# # 1. Data Processing: Create Graphs from CSV
# #######################################

# def load_csv_data(csv_path):
#     """
#     Loads CSV file.
#     The CSV is assumed to contain:
#       - Several feature columns (numeric)
#       - A 'graph_id' column to group transactions (for graph construction)
#       - An 'illicit' column for the target label (0/1)
#     """
#     df = pd.read_csv(csv_path)
#     return df

# def preprocess_group(df_group, feature_cols):
#     """
#     Given a group (a DataFrame corresponding to one graph), create a graph.
#     For simplicity, we construct a chain graph: node i connected to node i+1 (bidirectionally).
#     """
#     # Extract node features and scale them
#     X = df_group[feature_cols].values
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     x = torch.tensor(X, dtype=torch.float)
    
#     num_nodes = x.size(0)
#     if num_nodes < 2:
#         # If only one node, no edge.
#         edge_index = torch.empty((2, 0), dtype=torch.long)
#     else:
#         # Create chain edges: 0->1, 1->2, ... and the reverse edges
#         src = torch.arange(0, num_nodes - 1, dtype=torch.long)
#         dst = torch.arange(1, num_nodes, dtype=torch.long)
#         edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    
#     # For the label, we assume the graph label is defined for the group.
#     # For example, we can take the max of the illicit column (if any transaction is illicit, label the graph as illicit).
#     y_val = df_group["illicit"].max()  # 0 or 1
#     y = torch.tensor([y_val], dtype=torch.float)
    
#     return Data(x=x, edge_index=edge_index, y=y)

# def create_graph_dataset(csv_path):
#     """
#     Reads the CSV and groups rows by the 'graph_id' column.
#     Returns a list of torch_geometric.data.Data objects (one per graph).
#     """
#     df = load_csv_data(csv_path)
#     if "transaction_id" not in df.columns:
#         raise ValueError("CSV file must contain a 'transaction_id' column for grouping.")
    
#     # Identify feature columns (exclude 'graph_id' and 'illicit')
#     feature_cols = [col for col in df.columns if col not in ["transaction_id", "illicit"]]
    
#     graphs = []
#     for graph_id, group in df.groupby("transaction_id"):
#         graph = preprocess_group(group, feature_cols)
#         graphs.append(graph)
#     return graphs

# def split_graph_dataset(graphs, client_id, test_size=0.2):
#     """
#     Splits the list of graphs into train and test sets.
#     We use client_id as the random seed to create some variation across clients.
#     """
#     np.random.seed(client_id)
#     indices = np.arange(len(graphs))
#     np.random.shuffle(indices)
#     split = int(len(graphs) * (1 - test_size))
#     train_indices = indices[:split]
#     test_indices = indices[split:]
#     train_graphs = [graphs[i] for i in train_indices]
#     test_graphs = [graphs[i] for i in test_indices]
#     return train_graphs, test_graphs

# #######################################
# # 2. Define the Split Models (Front and Back)
# #######################################

# # Import PyTorch Geometric layers
# from torch_geometric.nn import GCNConv, global_mean_pool

# # Front Model: GNN encoder
# class FrontGNN(nn.Module):
#     def __init__(self, num_features, hidden_dim=64, out_dim=128):
#         """
#         num_features: dimension of node features.
#         hidden_dim: hidden dimension for GCN layers.
#         out_dim: dimension of the intermediate representation.
#         """
#         super(FrontGNN, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.pool = global_mean_pool  # Global mean pooling for graph-level representation.
#         self.fc = nn.Linear(hidden_dim, out_dim)
#         self.relu = nn.ReLU()
    
#     def forward(self, data):
#         # data is a torch_geometric.data.Data or a batch of Data objects.
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.conv1(x, edge_index)
#         x = self.relu(x)
#         x = self.conv2(x, edge_index)
#         x = self.relu(x)
#         x = self.pool(x, batch)  # shape: (batch_size, hidden_dim)
#         x = self.fc(x)
#         x = self.relu(x)
#         return x  # Intermediate representation (batch_size, out_dim)

# # Back Model: Fully connected classifier
# class BackFC(nn.Module):
#     def __init__(self, in_dim=128):
#         super(BackFC, self).__init__()
#         self.fc1 = nn.Linear(in_dim, 64)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(64, 1)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x

# #######################################
# # 3. Parameter Handling for SplitFed
# #######################################

# def get_combined_parameters(front_model, back_model, front_keys, back_keys):
#     front_state = front_model.state_dict()
#     back_state = back_model.state_dict()
#     params = [front_state[k].cpu().numpy() for k in front_keys] + \
#              [back_state[k].cpu().numpy() for k in back_keys]
#     return params

# def set_combined_parameters(front_model, back_model, parameters, front_keys, back_keys):
#     front_state = front_model.state_dict()
#     back_state = back_model.state_dict()
#     num_front = len(front_keys)
#     front_params = parameters[:num_front]
#     back_params = parameters[num_front:]
#     for key, param in zip(front_keys, front_params):
#         front_state[key] = torch.tensor(param)
#     for key, param in zip(back_keys, back_params):
#         back_state[key] = torch.tensor(param)
#     front_model.load_state_dict(front_state)
#     back_model.load_state_dict(back_state)

# #######################################
# # 4. Define the Flower Client for SplitFed GNN
# #######################################

# class TransactionGNNClient(fl.client.NumPyClient):
#     def __init__(self, cid: int, csv_path: str):
#         self.cid = cid
#         self.device = torch.device("cpu")
        
#         # Create the graph dataset from CSV
#         graphs = create_graph_dataset(csv_path)
#         train_graphs, test_graphs = split_graph_dataset(graphs, client_id=cid)
        
#         # Create PyTorch Geometric DataLoaders.
#         self.train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
#         self.test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
        
#         # Determine number of node features from the first graph.
#         num_features = graphs[0].x.size(1)
#         self.front_model = FrontGNN(num_features=num_features).to(self.device)
#         self.back_model = BackFC(in_dim=128).to(self.device)
        
#         # Optimizers for front and back models.
#         self.front_optimizer = optim.Adam(self.front_model.parameters(), lr=0.01)
#         self.back_optimizer = optim.Adam(self.back_model.parameters(), lr=0.01)
#         self.criterion = nn.BCELoss()
        
#         # Save parameter keys order (alphabetically sorted) for reproducible aggregation.
#         self.front_keys = sorted(self.front_model.state_dict().keys())
#         self.back_keys = sorted(self.back_model.state_dict().keys())
    
#     def get_parameters(self, config):
#         return get_combined_parameters(self.front_model, self.back_model, self.front_keys, self.back_keys)
    
#     def set_parameters(self, parameters):
#         set_combined_parameters(self.front_model, self.back_model, parameters, self.front_keys, self.back_keys)
    
#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         self.train_one_epoch()
#         return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, accuracy = self.test_model()
#         return float(loss), len(self.test_loader.dataset), {"accuracy": accuracy}
    
#     def train_one_epoch(self):
#         self.front_model.train()
#         self.back_model.train()
#         for data in self.train_loader:
#             data = data.to(self.device)
#             # Forward pass (Front GNN)
#             front_output = self.front_model(data)
#             front_output.retain_grad()  # Retain gradient for split learning
            
#             # Forward pass (Back FC)
#             output = self.back_model(front_output)
#             # data.y is of shape (batch_size, 1); squeeze to match output shape
#             loss = self.criterion(output.squeeze(), data.y.squeeze().float())
            
#             # Backward pass for the back model first.
#             self.back_optimizer.zero_grad()
#             loss.backward(retain_graph=True)
#             front_grad = front_output.grad.clone()
#             self.back_optimizer.step()
            
#             # Backward pass for the front model.
#             self.front_optimizer.zero_grad()
#             front_output.backward(front_grad)
#             self.front_optimizer.step()
    
#     def test_model(self):
#         self.front_model.eval()
#         self.back_model.eval()
#         total_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for data in self.test_loader:
#                 data = data.to(self.device)
#                 front_output = self.front_model(data)
#                 output = self.back_model(front_output)
#                 loss = self.criterion(output.squeeze(), data.y.squeeze().float())
#                 total_loss += loss.item() * data.num_graphs
#                 preds = (output.squeeze() > 0.5).float()
#                 correct += (preds == data.y.squeeze().float()).sum().item()
#                 total += data.num_graphs
#         return total_loss / total, correct / total

# #######################################
# # 5. Start the Flower Client
# #######################################

# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python client.py <client_id> <path_to_transaction_data.csv>")
#         sys.exit(1)
#     cid = int(sys.argv[1])
#     csv_path = sys.argv[2]
#     client = TransactionGNNClient(cid, csv_path)
#     fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader  # PyTorch Geometric Data and DataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys
import copy

# ---------------------------
# 1. Environment and Device Setup
# ---------------------------
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 2. Data Processing Functions for Graphs
# ---------------------------
def load_csv_data(csv_path):
    """
    Load the CSV file containing transaction data.
    Assumes the CSV has feature columns, a "graph_id" column, and an "illicit" column.
    """
    df = pd.read_csv(csv_path)
    return df

def preprocess_group(df_group, feature_cols):
    """
    Given a group (DataFrame for one graph), create a graph.
    Here we create a simple chain graph: node i connected to node i+1 (bidirectionally).
    """
    # Extract node features and scale them
    X = df_group[feature_cols].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x = torch.tensor(X, dtype=torch.float)
    
    num_nodes = x.size(0)
    if num_nodes < 2:
        # If only one node, no edge.
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        src = torch.arange(0, num_nodes - 1, dtype=torch.long)
        dst = torch.arange(1, num_nodes, dtype=torch.long)
        # Create bidirectional edges
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    
    # For the label, we use the maximum of the "illicit" column (if any transaction is illicit, label the graph as illicit)
    y_val = df_group["illicit"].max()
    y = torch.tensor([y_val], dtype=torch.long)  # For CrossEntropyLoss, target is LongTensor
    return Data(x=x, edge_index=edge_index, y=y)

def create_graph_dataset(csv_path):
    """
    Reads the CSV and groups rows by the "graph_id" column.
    Returns a list of torch_geometric.data.Data objects (one per graph).
    """
    df = load_csv_data(csv_path)
    if "transaction_id" not in df.columns:
        raise ValueError("CSV file must contain a 'transaction_id' column for grouping.")
    feature_cols = [col for col in df.columns if col not in ["transaction_id", "illicit"]]
    graphs = []
    for _, group in df.groupby("transaction_id"):
        graph = preprocess_group(group, feature_cols)
        graphs.append(graph)
    return graphs

def split_graph_dataset(graphs, client_id, test_size=0.2):
    """
    Splits the list of graphs into train and test sets.
    Uses client_id as seed variation.
    """
    np.random.seed(client_id)
    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    split = int(len(graphs) * (1 - test_size))
    train_graphs = [graphs[i] for i in indices[:split]]
    test_graphs = [graphs[i] for i in indices[split:]]
    return train_graphs, test_graphs

# ---------------------------
# 3. Model Definitions (Split Models)
# ---------------------------
# Front Model: GNN Encoder
class FrontGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, out_dim=128):
        """
        num_node_features: Dimension of node features.
        hidden_dim: Hidden dimension for GCN layers.
        out_dim: Dimension of intermediate representation.
        """
        super(FrontGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
    def forward(self, data):
        # data is a batch of graphs.
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # Global pooling to obtain graph-level representation
        x = global_mean_pool(x, batch)  # shape: (batch_size, hidden_dim)
        x = self.fc(x)
        x = self.relu(x)
        return x  # shape: (batch_size, out_dim)

# Back Model: Fully Connected Classifier
class BackFC(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=64, num_classes=2):
        """
        in_dim: Dimension of intermediate representation from front model.
        hidden_dim: Hidden dimension in FC layers.
        num_classes: Number of output classes (binary: 2).
        """
        super(BackFC, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # raw logits for CrossEntropyLoss

# ---------------------------
# 4. Helper Functions for Parameter Handling
# ---------------------------
def get_combined_parameters(front_model, back_model, front_keys, back_keys):
    front_state = front_model.state_dict()
    back_state = back_model.state_dict()
    params = [front_state[k].cpu().numpy() for k in front_keys] + \
             [back_state[k].cpu().numpy() for k in back_keys]
    return params

def set_combined_parameters(front_model, back_model, parameters, front_keys, back_keys):
    front_state = front_model.state_dict()
    back_state = back_model.state_dict()
    num_front = len(front_keys)
    front_params = parameters[:num_front]
    back_params = parameters[num_front:]
    for key, param in zip(front_keys, front_params):
        front_state[key] = torch.tensor(param)
    for key, param in zip(back_keys, back_params):
        back_state[key] = torch.tensor(param)
    front_model.load_state_dict(front_state)
    back_model.load_state_dict(back_state)

# ---------------------------
# 5. Flower Client Class for SplitFed with GNN
# ---------------------------
class SFLGNNClient(fl.client.NumPyClient):
    def __init__(self, cid: int, csv_path: str):
        self.cid = cid
        self.device = device
        # Load graphs from CSV and split into train and test
        graphs = create_graph_dataset(csv_path)
        train_graphs, test_graphs = split_graph_dataset(graphs, client_id=cid, test_size=0.2)
        self.train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
        # Determine number of node features from first graph
        num_node_features = graphs[0].x.size(1)
        # Initialize split models
        self.front_model = FrontGNN(num_node_features=num_node_features).to(self.device)
        self.back_model = BackFC(in_dim=128, hidden_dim=64, num_classes=2).to(self.device)
        # Optimizers
        self.front_optimizer = optim.Adam(self.front_model.parameters(), lr=0.0001)
        self.back_optimizer = optim.Adam(self.back_model.parameters(), lr=0.0001)
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        # Local epochs: 5 per round
        self.local_epochs = 5
        # Save sorted parameter keys
        self.front_keys = sorted(self.front_model.state_dict().keys())
        self.back_keys = sorted(self.back_model.state_dict().keys())

    def get_parameters(self, config):
        return get_combined_parameters(self.front_model, self.back_model, self.front_keys, self.back_keys)

    def set_parameters(self, parameters):
        set_combined_parameters(self.front_model, self.back_model, parameters, self.front_keys, self.back_keys)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Run training for local_epochs (5 epochs per round)
        for _ in range(self.local_epochs):
            self.train_one_epoch()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test_model()
        return float(loss), len(self.test_loader.dataset), {"accuracy": accuracy}

    def train_one_epoch(self):
        self.front_model.train()
        self.back_model.train()
        for data in self.train_loader:
            data = data.to(self.device)
            self.front_optimizer.zero_grad()
            self.back_optimizer.zero_grad()
            # Client-side forward pass (GNN encoder)
            front_output = self.front_model(data)
            # Retain gradient on the activation (simulate sending to server)
            front_output_retain = front_output.clone().detach().requires_grad_(True)
            # Server-side forward pass (classifier)
            logits = self.back_model(front_output_retain)
            loss = self.criterion(logits, data.y.squeeze())  # data.y shape: [batch_size, 1]
            # Backward pass on server side
            loss.backward(retain_graph=True)
            grad_from_server = front_output_retain.grad.clone()
            self.back_optimizer.step()
            # Backward pass on client side using the gradient from the server
            self.front_optimizer.zero_grad()
            front_output.backward(grad_from_server)
            self.front_optimizer.step()

    def test_model(self):
        self.front_model.eval()
        self.back_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                front_output = self.front_model(data)
                logits = self.back_model(front_output)
                loss = self.criterion(logits, data.y.squeeze())
                total_loss += loss.item() * data.num_graphs
                preds = logits.argmax(dim=1)
                correct += (preds == data.y.squeeze()).sum().item()
                total += data.num_graphs
        return total_loss / total, 100.0 * correct / total

# ---------------------------
# 6. Start Flower Client
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python client.py <client_id> <path_to_transaction_data.csv>")
        sys.exit(1)
    cid = int(sys.argv[1])
    csv_path = sys.argv[2]
    client = SFLGNNClient(cid, csv_path)
    # Start Flower client connecting to the Flower server.
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
