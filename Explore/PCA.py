import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Path to your Parquet files (update this as needed)
parquet_file_1 = "JoinedData/part.2.parquet"
#parquet_file_2 = "JoinedData/part.3.parquet"
#parquet_file_3 = "JoinedData/part.4.parquet"
#parquet_file_4 = "JoinedData/part.5.parquet"
#parquet_file_5 = "JoinedData/part.6.parquet"
#parquet_file_6 = "JoinedData/part.7.parquet"

# Get a list of all Parquet files
parquet_files = glob.glob(parquet_file_1)
#parquet_files += glob.glob(parquet_file_2)
#parquet_files += glob.glob(parquet_file_3)
#parquet_files += glob.glob(parquet_file_4)
#parquet_files += glob.glob(parquet_file_5)
#parquet_files += glob.glob(parquet_file_6)

# Load and concatenate all Parquet files
df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)

# Display basic info
print(df.info())

# Step 1: Drop unnecessary columns (but keep target attributes)
df = df.drop(columns=[
    'transaction_id', 'account_id', 'counterpart_id', 'assigned_bank', 
    'min', 'sec'  # Drop granular time features but keep targets
])
targets = df[['laundering_schema_type', 'laundering_schema_id']]
df = df.drop(columns=['laundering_schema_type', 'laundering_schema_id'])


# Step 2: Convert age to numerical (midpoint of the range)
df['age'] = df['age'].apply(lambda x: int(float(x)))

# Step 3: Cyclical encoding for time features
def cyclical_encode(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df.drop(columns=[col])

df = cyclical_encode(df, 'month', 12)
df = cyclical_encode(df, 'day', 31)
df = cyclical_encode(df, 'hour', 24)

# Step 4: Define numerical and categorical features
numerical_features = ['amount', 'initial_balance', 'age']
cyclical_features = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos']
categorical_features = [
    'weekday', 'transaction_direction', 'channel', 'payment_system', 
    'category_0', 'category_1', 'category_2', 'assigned_bank_type'
]

# Step 5: Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Standardize numerical features
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Step 6: Combine preprocessing steps
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features),
    ('cycl', 'passthrough', cyclical_features)  # Pass through cyclical features as-is
])

# Step 7: Apply preprocessing
X_preprocessed = preprocessor.fit_transform(df)

# Step 8: Perform PCA
svd = TruncatedSVD(n_components=10)  # Force 10 components
X_pca = svd.fit_transform(X_preprocessed)  # Works directly with sparse matrix

# Rejoin target attributes with PCA-transformed data
X_pca_with_targets = np.hstack((X_pca, targets))

# Step 9: Define regions and their associated lower PCs
regions = ["Asia", "Europe", "North America", "Africa", "South America"]
region_pc_mappings = {
    "Asia": [4, 5],      # PCs correlated with regional payment systems
    "Europe": [6, 7],    # PCs for timezone-adjusted hour patterns
    "North America": [8, 9],  # PCs for local spending habits
    "Africa": [4, 6],    # PCs for regional transaction patterns
    "South America": [5, 7]  # PCs for local currency effects
}

# Step 10: Assign PCs to clients with regional stratification
def assign_pcs_with_regions(client_regions, total_pcs=10, fixed_pcs=3):
    client_indices = []
    for region in client_regions:
        fixed = list(range(fixed_pcs))  # Top PCs (global)
        regional = region_pc_mappings[region]  # Region-specific PCs
        client_indices.append(fixed + regional)
    return client_indices

# Assign PCs to 5 clients with predefined regions
client_regions = np.random.choice(list(region_pc_mappings.keys()), size=5)
client_pc_indices = assign_pcs_with_regions(client_regions)

# Step 11: Add noise to simulate local variations
def add_noise(data, noise_scale=0.1):
    noise = np.random.randn(*data.shape) * noise_scale
    return data + noise

# Step 12: Assign subsets of PCs to each client and add noise
client_data = []
for indices in client_pc_indices:
    # Separate PCA features and targets
    client_pcs = X_pca_with_targets[:, indices] # PCA features
    client_targets = X_pca_with_targets[:, [10, 11]]  # Target columns
    # Add noise only to PCA features
    client_pcs_noisy = add_noise(client_pcs)
    client_data.append(np.hstack((client_pcs_noisy, client_targets)))

# Step 13: Simulate federated training
for client_id, (region, data) in enumerate(zip(client_regions, client_data)):
    print(f"Client {client_id + 1} (Region: {region}) PCs: {client_pc_indices[client_id]}")
    print(f"Client {client_id + 1} data shape: {data.shape}")
    # Code to train local model on data (simulated)

'''
# Step 14: Vary PCA variance thresholds (optional)
# Simulate clients retaining different numbers of PCs
variance_thresholds = [0.90, 0.92, 0.95, 0.97, 0.99]  # Different thresholds for each client
client_pca_results = []
for threshold in variance_thresholds:
    pca_client = PCA(n_components=threshold)
    X_pca_client = pca_client.fit_transform(X_preprocessed)
    client_pca_results.append(X_pca_client)

# Print variance explained for each client
for client_id, X_pca_client in enumerate(client_pca_results):
    print(f"Client {client_id + 1} retained {X_pca_client.shape[1]} PCs for {variance_thresholds[client_id] * 100}% variance")

'''
