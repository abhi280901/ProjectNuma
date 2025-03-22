import pandas as pd

# Load the original Parquet file into a DataFrame
df = pd.read_parquet('part.1.parquet')

# Select the first 5 rows
first_five_rows = df.head(5)

# Save those rows to a new Parquet file
first_five_rows.to_parquet('first_five_rows.parquet', index=False)

print("Saved the first 5 rows to 'first_five_rows.parquet'")