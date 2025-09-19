
import pandas as pd
from chembl_webresource_client.new_client import new_client

# --- Configuration ---
target_chembl_id = "CHEMBL203"  # EGFR, a well-studied kinase
output_filename = "egfr_ic50_data.csv"

# --- Fetching Data ---
print(f"Connecting to ChEMBL and fetching activities for target: {target_chembl_id}")
activity = new_client.activity
res = activity.filter(
    target_chembl_id=target_chembl_id,
    standard_type="IC50"
).only(
    'molecule_chembl_id',
    'canonical_smiles',
    'standard_value',
    'standard_units'
)

# --- Processing Data ---
print("Converting to DataFrame...")
df = pd.DataFrame(res)

print(f"Initial records fetched: {len(df)}")

# Drop records with missing values in critical columns
df.dropna(subset=['canonical_smiles', 'standard_value', 'standard_units'], inplace=True)
print(f"Records after dropping NA: {len(df)}")

# Filter for standard units of 'nM'
df = df[df['standard_units'] == 'nM']
print(f"Records after filtering for 'nM' units: {len(df)}")

# Convert standard_value to numeric, coercing errors
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
df.dropna(subset=['standard_value'], inplace=True)

# Remove duplicates based on SMILES string, keeping the first entry
df.drop_duplicates('canonical_smiles', inplace=True)

print(f"Final clean records: {len(df)}")

# --- Saving Data ---
if not df.empty:
    df.to_csv(output_filename, index=False)
    print(f"Successfully saved {len(df)} compounds to {output_filename}")
else:
    print("No data to save.")

