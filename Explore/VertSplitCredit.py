import pandas as pd
import random

# Change with the actual CSV file path
df = pd.read_csv('CreditFraudData/creditcard.csv')

# Print the shape of the DataFrame
print("DataFrame Shape:", df.shape)

# Define available PCs from V11 to V28
available_pcs = [f"V{i}" for i in range(11, 29)]

# The 10 common PCs + target column
base_columns = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "Class"]

# Creating separate DataFrames for each client
for i in range(1, 6):
    extra_pcs = random.sample(available_pcs, 5)  # Randomly select 5 PCs
    selected_columns = base_columns[:-1] + extra_pcs + ["Class"]  # Ensure Class is last
    client_df = df[selected_columns]
    
    # Print the shape and selected PCs
    print(f"Client_{i}: Shape = {client_df.shape}")
    print(f"Columns: {selected_columns[:-1]} + ['Class']\n")
