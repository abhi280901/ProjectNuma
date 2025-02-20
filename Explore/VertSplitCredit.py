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

# Compute variance for each PC
pc_variance = df.var()

# Compute total variance across all PCs
total_variance = sum([pc_variance[f"V{i}"] for i in range(1, 29)])

# Creating separate DataFrames for each client
for i in range(1, 6):
    extra_pcs = random.sample(available_pcs, 5)  # Randomly select 5 PCs
    selected_columns = base_columns[:-1] + extra_pcs + ["Class"]  # Ensure Class is last
    client_df = df[selected_columns]

    # Calculate total variance captured by the client
    variance_captured = sum(pc_variance[selected_columns[:-1]])
    variance_percentage = (variance_captured / total_variance) * 100
    
    # Print the shape and selected PCs
    print(f"Client_{i}: Shape = {client_df.shape}")
    print(f"Columns: {selected_columns[:-1]} + ['Class']")
    print(f"Total Variance Captured:{variance_percentage:.2f}%\n")
