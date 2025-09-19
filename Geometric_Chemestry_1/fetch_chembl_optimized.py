#!/usr/bin/env python3
"""
Optimized ChEMBL Data Fetcher for Chemical Geometry Study
Fetches compounds with IC50 data for multiple kinase targets to reach 10,000+ compounds
"""

import pandas as pd
from chembl_webresource_client.new_client import new_client
import time
import sys

# Configuration
KINASE_TARGETS = [
    "CHEMBL203",   # EGFR
    "CHEMBL2971",  # CDK2
    "CHEMBL301",   # ALK
    "CHEMBL4005",  # ABL1
    "CHEMBL3778",  # SRC
    "CHEMBL2034",  # VEGFR2
    "CHEMBL279",   # PDGFRB
    "CHEMBL2695",  # PI3K
    "CHEMBL4822",  # mTOR
    "CHEMBL3880"   # AKT1
]

OUTPUT_FILE = "kinase_compounds_dataset.csv"
MAX_COMPOUNDS_PER_TARGET = 2000

def fetch_target_data(target_id, max_compounds=MAX_COMPOUNDS_PER_TARGET):
    """Fetch compound data for a single target"""
    print(f"Fetching data for target: {target_id}")
    
    try:
        activity = new_client.activity
        
        # Fetch with pagination to avoid timeouts
        compounds = []
        offset = 0
        limit = 500
        
        while len(compounds) < max_compounds:
            print(f"  Fetching batch {offset//limit + 1} (offset: {offset})")
            
            batch = activity.filter(
                target_chembl_id=target_id,
                standard_type="IC50",
                standard_units="nM"
            ).only(
                'molecule_chembl_id',
                'canonical_smiles', 
                'standard_value'
            )[offset:offset+limit]
            
            batch_list = list(batch)
            if not batch_list:
                break
                
            compounds.extend(batch_list)
            offset += limit
            
            if len(batch_list) < limit:
                break
                
        print(f"  Retrieved {len(compounds)} compounds for {target_id}")
        return compounds
        
    except Exception as e:
        print(f"  Error fetching {target_id}: {str(e)}")
        return []

def main():
    print("Starting optimized ChEMBL data collection...")
    print(f"Target: {len(KINASE_TARGETS)} kinase targets")
    
    all_compounds = []
    
    for target_id in KINASE_TARGETS:
        compounds = fetch_target_data(target_id)
        
        # Add target information
        for compound in compounds:
            compound['target_chembl_id'] = target_id
            
        all_compounds.extend(compounds)
        
        print(f"Total compounds collected so far: {len(all_compounds)}")
        
        # Stop if we have enough compounds
        if len(all_compounds) >= 10000:
            print("Reached target of 10,000+ compounds")
            break
            
        # Brief pause to avoid overwhelming the API
        time.sleep(1)
    
    if not all_compounds:
        print("No compounds retrieved. Exiting.")
        sys.exit(1)
    
    # Convert to DataFrame and clean
    print("Processing and cleaning data...")
    df = pd.DataFrame(all_compounds)
    
    print(f"Initial compounds: {len(df)}")
    
    # Remove compounds with missing critical data
    df = df.dropna(subset=['canonical_smiles', 'standard_value'])
    print(f"After removing missing data: {len(df)}")
    
    # Convert IC50 values to numeric
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna(subset=['standard_value'])
    print(f"After numeric conversion: {len(df)}")
    
    # Remove duplicates based on SMILES
    df = df.drop_duplicates(subset=['canonical_smiles'])
    print(f"After removing duplicates: {len(df)}")
    
    # Filter for reasonable IC50 values (0.1 nM to 100,000 nM)
    df = df[(df['standard_value'] >= 0.1) & (df['standard_value'] <= 100000)]
    print(f"After filtering IC50 range: {len(df)}")
    
    # Add pIC50 (negative log of IC50 in M)
    df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)
    
    # Save the dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved {len(df)} compounds to {OUTPUT_FILE}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Unique targets: {df['target_chembl_id'].nunique()}")
    print(f"IC50 range: {df['standard_value'].min():.2f} - {df['standard_value'].max():.2f} nM")
    print(f"pIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
    print(f"Mean pIC50: {df['pIC50'].mean():.2f}")

if __name__ == "__main__":
    import numpy as np
    main()
