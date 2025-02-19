# =============================================================================
# SplitFedV1_FraudResNet_Fraud_Undersampling:
# SplitFed Learning using a FraudResNet for Binary Fraud Detection
# (Now recording precision, recall, and F1-score for both training and testing)
# =============================================================================

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

# For data splitting, scaling, and undersampling
from sklearn.model_selection import train_test_split  # Split data into training and test sets
from sklearn.preprocessing import StandardScaler        # Scale features
from imblearn.over_sampling import SMOTE                # For oversampling (if needed) – here we undersample
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality


# For additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import time
import warnings 
warnings.filterwarnings('ignore') 

# ------------------ Set seeds for reproducibility -------------------------
start_time = time.time()
SEED = 1234                                  # Define seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device

# ------------------ Program Information and Hyperparameters ---------------
program = "SplitFed_ResNet_CTGAN_ep10_lep_2_lr1e-3_drop"
print(f"---------{program}----------")

num_users = 5         # Number of simulated clients
epochs = 10           # Number of global rounds
frac = 1              # Fraction of clients participating each round
lr = 0.001            # Learning rate

# ------------------ Model Parameters --------------------------
# For tabular data we assume input_dim features (e.g., 30). 
# The provided FraudResNet design uses an initial projection, residual blocks, and final classification.
input_dim = 5        # Number of features (should match your dataset)
hidden_dim = 64      # Hidden representation dimension
output_dim = 2        # Number of output classes (binary classification)
num_blocks = 4        # Number of residual blocks
weight_decay = 1e-4
dropout = 0.2

# =============================================================================
# Global variables to collect metrics across rounds
# =============================================================================
acc_train_collect = []      
loss_train_collect = []
precision_train_collect = []
recall_train_collect = []
f1_train_collect = []

acc_test_collect = []
loss_test_collect = []
precision_test_collect = []
recall_test_collect = []
f1_test_collect = []

acc_train_collect_user = []
loss_train_collect_user = []
precision_train_collect_user = []
recall_train_collect_user = []
f1_train_collect_user = []

acc_test_collect_user = []
loss_test_collect_user = []
precision_test_collect_user = []
recall_test_collect_user = []
f1_test_collect_user = []

batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

train_all_preds = []
train_all_labels = []
test_all_preds = []
test_all_labels = []

criterion = nn.CrossEntropyLoss()  # Loss function

# =============================================================================
# 1. Define the ResidualBlock as provided
# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Added dropout
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
    def __init__(self, input_dim, hidden_dim, num_blocks=3, dropout_rate=0.2):
        super(FraudResNetClient, self).__init__()
        # Initial projection layer: maps input to hidden_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Residual blocks: stack of num_blocks residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
    def forward(self, x):
        x = self.input_layer(x)             # Project input to hidden_dim space
        x = self.res_blocks(x)              # Apply residual blocks
        return x                          # Return the hidden representation

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

# =============================================================================
# 4. Federated Averaging Function (unchanged)
# =============================================================================
def FedAvg(w_list):
    w_avg = copy.deepcopy(w_list[0])
    for key in w_avg.keys():
        for i in range(1, len(w_list)):
            w_avg[key] += w_list[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w_list))
    return w_avg

# =============================================================================
# 5. Utility Function for Calculating Accuracy (unchanged)
# =============================================================================
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.0 * correct.float() / preds.shape[0]
    return acc

# =============================================================================
# 6. Global Variables for Server-Side Federation
# =============================================================================
w_glob_server = None  # Will be set after initializing the server model
w_locals_server = []
idx_collect = []
l_epoch_check = False
fed_check = False

# For simulating a copy of the server model for each client; will be set later
net_model_server = None
net_server = None

# =============================================================================
# 7. Initialize Global Models using FraudResNetClient and FraudResNetServer
# =============================================================================
net_glob_client = FraudResNetClient(input_dim, hidden_dim, num_blocks=num_blocks, dropout_rate=dropout)
if torch.cuda.device_count() > 1:
    net_glob_client = nn.DataParallel(net_glob_client)
net_glob_client.to(device)
print("Client model:")
print(net_glob_client)

net_glob_server = FraudResNetServer(hidden_dim, output_dim)
if torch.cuda.device_count() > 1:
    net_glob_server = nn.DataParallel(net_glob_server)
net_glob_server.to(device)
print("Server model:")
print(net_glob_server)

# Set global server state and duplicate for simulation purposes
w_glob_server = net_glob_server.state_dict()
net_model_server = [w_glob_server for _ in range(num_users)]
net_server = copy.deepcopy(net_model_server[0])
net_server = net_glob_server  # Alternatively, use the global server model as a starting point

# =============================================================================
# 8. Server-Side Training Function (using new FraudResNet models)
# =============================================================================
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server, criterion, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global w_locals_server, w_glob_server, net_server, lr
    global train_all_preds, train_all_labels
    global acc_train_collect_user, loss_train_collect_user
    global precision_train_collect_user, recall_train_collect_user, f1_train_collect_user
    global acc_avg_all_user_train, loss_avg_all_user_train, precision_avg_all_user_train, recall_avg_all_user_train, f1_avg_all_user_train

    # Get a copy of the server model for the given client index
    net_server = copy.deepcopy(net_model_server[idx])
    net_server = net_glob_server  # Ensure consistency; here we use global server model
    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=weight_decay)
    
    optimizer_server.zero_grad()
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    # Forward pass through the server model
    fx_server = net_server(fx_client)
    loss = criterion(fx_server, y)
    acc = calculate_accuracy(fx_server, y)
    
    preds_train = fx_server.argmax(dim=1)
    train_all_preds.extend(preds_train.cpu().numpy())
    train_all_labels.extend(y.cpu().numpy())
    
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    
    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())
    
    # Update the simulated server model for this client
    net_model_server[idx] = copy.deepcopy(net_server.state_dict())
    
    if len(batch_acc_train) == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)
        
        precision_train = precision_score(train_all_labels, train_all_preds, average='binary', zero_division=0)
        recall_train = recall_score(train_all_labels, train_all_preds, average='binary', zero_division=0)
        f1_train = f1_score(train_all_labels, train_all_preds, average='binary', zero_division=0)
        
        print(f'Client {idx} Train => Local Epoch: {l_epoch_count} Acc: {acc_avg_train:.3f} Loss: {loss_avg_train:.4f} Precision: {precision_train:.3f} Recall: {recall_train:.3f} F1: {f1_train:.3f}')
        
        del train_all_preds[:]
        del train_all_labels[:]
        batch_acc_train.clear()
        batch_loss_train.clear()
        
        if l_epoch_count == l_epoch - 1:
            global l_epoch_check
            l_epoch_check = True
            w_locals_server.append(copy.deepcopy(net_server.state_dict()))
            acc_train_collect_user.append(acc_avg_train)
            loss_train_collect_user.append(loss_avg_train)
            precision_train_collect_user.append(precision_train)
            recall_train_collect_user.append(recall_train)
            f1_train_collect_user.append(f1_train)
            if idx not in idx_collect:
                idx_collect.append(idx)
                
        if len(idx_collect) == num_users:
            global fed_check
            fed_check = True
            w_glob_server = FedAvg(w_locals_server)
            net_glob_server.load_state_dict(w_glob_server)
            net_model_server = [w_glob_server for _ in range(num_users)]
            
            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)
            precision_avg_all_user_train = sum(precision_train_collect_user) / len(precision_train_collect_user)
            recall_avg_all_user_train = sum(recall_train_collect_user) / len(recall_train_collect_user)
            f1_avg_all_user_train = sum(f1_train_collect_user) / len(f1_train_collect_user)
            
            acc_train_collect.append(acc_avg_all_user_train)
            loss_train_collect.append(loss_avg_all_user_train)
            precision_train_collect.append(precision_avg_all_user_train)
            recall_train_collect.append(recall_avg_all_user_train)
            f1_train_collect.append(f1_avg_all_user_train)
            
            acc_train_collect_user.clear()
            loss_train_collect_user.clear()
            precision_train_collect_user.clear()
            recall_train_collect_user.clear()
            f1_train_collect_user.clear()
            idx_collect.clear()
            w_locals_server.clear()
    
    return dfx_client

# =============================================================================
# 9. Server-Side Evaluation Function (using new FraudResNet models)
# =============================================================================
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_model_server, criterion, batch_acc_test, batch_loss_test, net_glob_server 
    global acc_avg_all_user_train, loss_avg_all_user_train, precision_avg_all_user_train, recall_avg_all_user_train, f1_avg_all_user_train
    global acc_test_collect_user, loss_test_collect_user
    global precision_test_collect_user, recall_test_collect_user, f1_test_collect_user
    global test_all_preds, test_all_labels

    net = copy.deepcopy(net_model_server[idx])
    net = net_glob_server  # Use global server model for evaluation
    net.eval()
    
    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        fx_server = net(fx_client)
        loss = criterion(fx_server, y)
        acc = calculate_accuracy(fx_server, y)
        
        preds = fx_server.argmax(dim=1)
        test_all_preds.extend(preds.cpu().numpy())
        test_all_labels.extend(y.cpu().numpy())
        
        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())
        
        if len(batch_acc_test) == len_batch:
            acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)
            
            precision_test = precision_score(test_all_labels, test_all_preds, average='binary', zero_division=0)
            recall_test = recall_score(test_all_labels, test_all_preds, average='binary', zero_division=0)
            f1_test = f1_score(test_all_labels, test_all_preds, average='binary', zero_division=0)
            
            print(f'Client {idx} Test => Acc: {acc_avg_test:.3f} Loss: {loss_avg_test:.4f} Precision: {precision_test:.3f} Recall: {recall_test:.3f} F1: {f1_test:.3f}')
            
            print("---------------------------------------------------")
            print("Classification Report: \n")
            print(classification_report(test_all_labels, test_all_preds))
            print("Confusion matrix: \n")
            print(confusion_matrix(test_all_labels, test_all_preds))
            print("---------------------------------------------------")
            
            del test_all_preds[:]
            del test_all_labels[:]
            batch_acc_test.clear()
            batch_loss_test.clear()
            
            acc_test_collect_user.append(acc_avg_test)
            loss_test_collect_user.append(loss_avg_test)
            precision_test_collect_user.append(precision_test)
            recall_test_collect_user.append(recall_test)
            f1_test_collect_user.append(f1_test)
            
            global fed_check, l_epoch_check
            if l_epoch_check:
                l_epoch_check = False
            if fed_check:
                fed_check = False
                print("------------------------------------------------")
                print("Federation process at Server-Side")
                print("------------------------------------------------")
                acc_avg_all_user_test = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all_user_test = sum(loss_test_collect_user) / len(loss_test_collect_user)
                precision_avg_all_user_test = sum(precision_test_collect_user) / len(precision_test_collect_user)
                recall_avg_all_user_test = sum(recall_test_collect_user) / len(recall_test_collect_user)
                f1_avg_all_user_test = sum(f1_test_collect_user) / len(f1_test_collect_user)
                
                acc_test_collect.append(acc_avg_all_user_test)
                loss_test_collect.append(loss_avg_all_user_test)
                precision_test_collect.append(precision_avg_all_user_test)
                recall_test_collect.append(recall_avg_all_user_test)
                f1_test_collect.append(f1_avg_all_user_test)
                
                acc_test_collect_user.clear()
                loss_test_collect_user.clear()
                precision_test_collect_user.clear()
                recall_test_collect_user.clear()
                f1_test_collect_user.clear()
                print(f'Round {ell} Train => Avg Acc: {acc_avg_all_user_train:.3f} Avg Loss: {loss_avg_all_user_train:.3f} Avg F1: {f1_avg_all_user_train:.3f} Avg Precision: {precision_avg_all_user_train:.3f}  Avg Recall: {recall_avg_all_user_train:.3f}')
                print(f'Round {ell} Test  => Avg Acc: {acc_avg_all_user_test:.3f} Avg Loss: {loss_avg_all_user_test:.3f} Avg F1: {f1_avg_all_user_test:.3f} Avg Precision: {precision_avg_all_user_test:.3f}  Avg Recall: {recall_avg_all_user_test:.3f}')
    return

# =============================================================================
# 10. DatasetSplit Class for Federated Learning (unchanged)
# =============================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]

# =============================================================================
# 11. Client Class (now uses FraudResNetClient as the local model)
# =============================================================================
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None, idxs_test=None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1   # Number of local epochs per round
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=128, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=128, shuffle=True)
    
    def train(self, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=weight_decay)
        global train_all_preds, train_all_labels
        train_all_preds = []
        train_all_labels = []
        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (features, labels) in enumerate(self.ldr_train):
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                fx = net(features)  # net is the FraudResNetClient model
                client_fx = fx.clone().detach().requires_grad_(True)
                dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                fx.backward(dfx)
                optimizer_client.step()
        return net.state_dict()
    
    def evaluate(self, net, ell):
        net.eval()
        global test_all_preds, test_all_labels
        test_all_preds = []
        test_all_labels = []
        len_batch = len(self.ldr_test)
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(self.ldr_test):
                features, labels = features.to(self.device), labels.to(self.device)
                fx = net(features)
                evaluate_server(fx, labels, self.idx, len_batch, ell)
        return

# =============================================================================
# 12. FraudDataset Class for Tabular Data (unchanged)
# =============================================================================
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# =============================================================================
# 13. Data Loading and Preprocessing
# =============================================================================
df_full = pd.read_parquet('../BIS_data/Final_BIS_Data.parquet')
df_full = df_full.reset_index(drop=True)
df_full.info()

# Separate features and target (assumes column "label" holds the binary target)
X = df_full.drop('laundering_schema_type', axis=1)
y = df_full['laundering_schema_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

Xy_train_ctgan = pd.read_csv("BIS_CTGAN_Full.csv")
print(Xy_train_ctgan.info())

# Separate features and labels for the undersampled training data.
X_train_final = Xy_train_ctgan.drop('laundering_schema_type', axis=1).values
y_train_final = Xy_train_ctgan['laundering_schema_type'].values

# For the test set, keep the original (imbalanced) distribution.
X_test = X_test.values
y_test = y_test.values

print("After CTGAN, training set shape:", X_train_final.shape)

input_dim = X_train_final.shape[1]

dataset_train = FraudDataset(X_train_final, y_train_final)
dataset_test = FraudDataset(X_test, y_test)

# =============================================================================
# 14. Partition Data Among Clients (IID split)
# =============================================================================
def dataset_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

dict_users = dataset_iid(dataset_train, num_users)
dict_users_test = dataset_iid(dataset_test, num_users)

# =============================================================================
# 15. Main Training and Evaluation Loop
# =============================================================================
# Set the client model to use our FraudResNetClient design
net_glob_client.train()
w_glob_client = net_glob_client.state_dict()

# Initialize best test loss for saving the best model
best_test_loss = float('inf')

for iter in range(epochs):
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)
    w_locals_client = []
    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device,
                       dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx])
        w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))
        w_locals_client.append(copy.deepcopy(w_client))
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)
    print("-----------------------------------------------------------")
    print("Federation process at Client-Side")
    print("-----------------------------------------------------------")
    w_glob_client = FedAvg(w_locals_client)
    net_glob_client.load_state_dict(w_glob_client)

    # Check if the current round's global test loss is the best so far and save models
    if len(loss_test_collect) > 0:
        current_loss = loss_test_collect[-1]
        if current_loss < best_test_loss:
            best_test_loss = current_loss
            # Save both the global client-side and server-side models
            torch.save(net_glob_client.state_dict(), "best_client_model_CTGAN_ResNet_V2.pth")
            torch.save(net_glob_server.state_dict(), "best_server_model_CTGAN_ResNet_V2.pth")
            print(f"New best model saved with test loss: {best_test_loss:.4f}")

print("Training and Evaluation completed!")

# =============================================================================
# 16. Save Metrics to an Excel File
# =============================================================================
round_process = list(range(1, len(acc_train_collect) + 1))
metrics_df = pd.DataFrame({
    'round': round_process,
    'acc_train': acc_train_collect,
    'loss_train': loss_train_collect,
    'precision_train': precision_train_collect,
    'recall_train': recall_train_collect,
    'f1_train': f1_train_collect,
    'acc_test': acc_test_collect,
    'loss_test': loss_test_collect,
    'precision_test': precision_test_collect,
    'recall_test': recall_test_collect,
    'f1_test': f1_test_collect
})

metrics_df.to_excel(program + ".xlsx", sheet_name="v2_test", index=False)

print("--- %s seconds ---" % (time.time() - start_time))
