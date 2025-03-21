from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from SplitFed_ResNet_CTGAN_BIS_Demo import FraudResNetClient, FraudResNetServer, NewDataDataset
from sklearn.metrics import accuracy_score, classification_report
import os

app = FastAPI()

# Model configuration
input_dim = 5
hidden_dim = 128
output_dim = 2
num_blocks = 5

# Load models
net_client = FraudResNetClient(input_dim, hidden_dim, num_blocks=num_blocks)
net_server = FraudResNetServer(hidden_dim, output_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_client.to(device)
net_server.to(device)

# Load saved weights
net_client.load_state_dict(
    torch.load("best_client_model_CTGAN_ResNet_V2_wo_drop_128_005.pth", map_location=torch.device('cpu'))
)
net_server.load_state_dict(
    torch.load("best_server_model_CTGAN_ResNet_V2_wo_drop_128_005.pth", map_location=torch.device('cpu'))
)

net_client.eval()
net_server.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV
        data = pd.read_csv(file.file)
        if 'laundering_schema_type' not in data.columns:
            raise HTTPException(status_code=400, detail="Missing 'laundering_schema_type' column")

        X_new = data.drop('laundering_schema_type', axis=1).values
        y_new = data['laundering_schema_type'].values

        # Create dataset and dataloader
        new_dataset = NewDataDataset(X_new, y_new)
        new_loader = DataLoader(new_dataset, batch_size=128, shuffle=False)

        all_preds = []
        all_labels = []

        # Inference
        with torch.no_grad():
            for features, labels in new_loader:
                features = features.to(device)
                labels = labels.to(device)
                client_out = net_client(features)
                logits = net_server(client_out)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Accuracy & report
        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, output_dict=True)
        return {"accuracy": f"{acc * 100:.2f}%", "report": report}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "FedShield Backend Running!"}
