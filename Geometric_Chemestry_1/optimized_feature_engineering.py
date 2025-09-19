#!/usr/bin/env python3
"""
Optimized Feature Engineering for Chemical Geometry Study
Focused on key molecular descriptors for geometric mapping analysis
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_key_descriptors(mol):
    """Calculate the most important molecular descriptors for biological activity"""
    descriptors = {}
    
    # Basic physicochemical properties (Lipinski's Rule of Five)
    descriptors['MolWt'] = Descriptors.MolWt(mol)
    descriptors['LogP'] = Descriptors.MolLogP(mol)
    descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
    descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    descriptors['TPSA'] = Descriptors.TPSA(mol)
    
    # Structural complexity
    descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    descriptors['RingCount'] = Descriptors.RingCount(mol)
    descriptors['BertzCT'] = Descriptors.BertzCT(mol)
    
    # Connectivity indices (important for SAR)
    descriptors['Chi0v'] = Descriptors.Chi0v(mol)
    descriptors['Chi1v'] = Descriptors.Chi1v(mol)
    descriptors['Chi2v'] = Descriptors.Chi2v(mol)
    descriptors['Chi3v'] = Descriptors.Chi3v(mol)
    
    # Shape and flexibility
    descriptors['Kappa1'] = Descriptors.Kappa1(mol)
    descriptors['Kappa2'] = Descriptors.Kappa2(mol)
    descriptors['Kappa3'] = Descriptors.Kappa3(mol)
    
    # Electronic properties
    descriptors['MaxEStateIndex'] = Descriptors.MaxEStateIndex(mol)
    descriptors['MinEStateIndex'] = Descriptors.MinEStateIndex(mol)
    descriptors['qed'] = Descriptors.qed(mol)
    
    # Surface area
    descriptors['LabuteASA'] = Descriptors.LabuteASA(mol)
    
    # Key pharmacophore features
    descriptors['fr_NH0'] = Descriptors.fr_NH0(mol)
    descriptors['fr_NH1'] = Descriptors.fr_NH1(mol)
    descriptors['fr_NH2'] = Descriptors.fr_NH2(mol)
    descriptors['fr_C_O'] = Descriptors.fr_C_O(mol)
    descriptors['fr_COO'] = Descriptors.fr_COO(mol)
    descriptors['fr_N_O'] = Descriptors.fr_N_O(mol)
    
    return descriptors

def calculate_morgan_fingerprint(mol, radius=2, nBits=1024):
    """Calculate Morgan (ECFP) fingerprint - most important for similarity"""
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

def process_compounds_optimized(df):
    """Process compounds with optimized feature calculation"""
    print("Starting optimized feature engineering...")
    
    descriptors_list = []
    morgan_fps = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = row['canonical_smiles']
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Calculate key descriptors
            desc = calculate_key_descriptors(mol)
            descriptors_list.append(desc)
            
            # Calculate Morgan fingerprint
            morgan_fp = calculate_morgan_fingerprint(mol)
            morgan_fps.append(morgan_fp)
            
            valid_indices.append(idx)
            
            if len(valid_indices) % 500 == 0:
                print(f"Processed {len(valid_indices)} compounds...")
                
        except Exception as e:
            print(f"Error processing compound at index {idx}: {str(e)}")
            continue
    
    print(f"Successfully processed {len(valid_indices)} compounds")
    
    # Convert to DataFrames
    descriptors_df = pd.DataFrame(descriptors_list)
    morgan_array = np.array(morgan_fps)
    
    # Filter original dataframe
    filtered_df = df.iloc[valid_indices].reset_index(drop=True)
    
    # Combine with descriptors
    result_df = pd.concat([filtered_df, descriptors_df], axis=1)
    
    return result_df, morgan_array

def main():
    """Main execution function"""
    print("Loading kinase compounds dataset...")
    df = pd.read_csv('kinase_compounds_dataset.csv')
    print(f"Loaded {len(df)} compounds")
    
    # Process with optimized features
    processed_df, morgan_fps = process_compounds_optimized(df)
    
    # Save processed dataset
    processed_df.to_csv('kinase_compounds_features.csv', index=False)
    print(f"Saved processed dataset with {len(processed_df)} compounds")
    
    # Save Morgan fingerprints
    np.save('morgan_fingerprints.npy', morgan_fps)
    print("Saved Morgan fingerprints")
    
    # Print summary
    print(f"\nFeature Engineering Summary:")
    print(f"Original compounds: {len(df)}")
    print(f"Successfully processed: {len(processed_df)}")
    print(f"Success rate: {len(processed_df)/len(df)*100:.1f}%")
    
    # Show feature statistics
    feature_cols = [col for col in processed_df.columns if col not in ['canonical_smiles', 'molecule_chembl_id', 'standard_value', 'target_chembl_id', 'pIC50']]
    print(f"Molecular descriptors: {len(feature_cols)}")
    print(f"Morgan fingerprint bits: {morgan_fps.shape[1]}")
    
    # Display basic statistics
    print(f"\nDataset statistics:")
    print(f"pIC50 range: {processed_df['pIC50'].min():.2f} - {processed_df['pIC50'].max():.2f}")
    print(f"Mean pIC50: {processed_df['pIC50'].mean():.2f}")
    print(f"Molecular weight range: {processed_df['MolWt'].min():.1f} - {processed_df['MolWt'].max():.1f}")
    print(f"LogP range: {processed_df['LogP'].min():.2f} - {processed_df['LogP'].max():.2f}")

if __name__ == "__main__":
    main()
