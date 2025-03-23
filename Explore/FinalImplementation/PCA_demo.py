import pandas as pd
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Path to your Parquet files (update this as needed)
parquet_file_1 = "FinalImplementation/BIS_Raw_sample_5.parquet"

# Get a list of all Parquet files
parquet_files = glob.glob(parquet_file_1)

# Load and concatenate all Parquet files
df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)

# Display basic info
print("Original Data")
print(df)

# Step 1: Drop unnecessary columns (but keep target attributes)
df = df.drop(columns=[
    'transaction_id', 'account_id', 'counterpart_id', 'assigned_bank', 
    'min', 'sec','category_1', 'category_2','weekday', 'laundering_schema_id'  # Drop granular time features but keep targets
])
targets = df[['laundering_schema_type']]
df = df.drop(columns=['laundering_schema_type'])

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
    'transaction_direction', 'channel', 'payment_system', 
    'category_0', 'assigned_bank_type'
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

# Step 8: Perform SVD truncation (Top 5 PCs)
svd = TruncatedSVD(n_components=5)  # Keep only top 5 principal components
X_svd = svd.fit_transform(X_preprocessed)

# Step 9: Add noise to simulate local variations
def add_noise(data, mean=0.0, std_dev=0.1):
    noise = np.random.normal(loc=mean, scale=std_dev, size=data.shape)
    return data + noise

# Step 10: Apply noise to SVD-truncated data
X_svd_noisy = add_noise(X_svd)

# Step 11: Rejoin target attributes with noisy SVD-transformed data
X_svd_noisy_with_targets = np.hstack((X_svd_noisy, targets.values))


# Save the noisy data with targets to a CSV file
noisy_data_with_targets_df = pd.DataFrame(
    X_svd_noisy_with_targets, 
    columns=[f"pca_{i}" for i in range(5)] + ['laundering_schema_type']
)
noisy_data_df = pd.DataFrame(
    X_svd_noisy, 
    columns=[f"PC{i+1}" for i in range(5)]
)

# Assuming this is a pandas DataFrame
noisy_data_with_targets_df["laundering_schema_type"] = noisy_data_with_targets_df["laundering_schema_type"].apply(
    lambda x: 0.0 if pd.isna(x) else 1.0
)

noisy_data_with_targets_df.to_csv("FinalImplementation/noisy_data_with_targets.csv", index=False)

print("\nNoisy Data with Targets:")
print(noisy_data_df)
