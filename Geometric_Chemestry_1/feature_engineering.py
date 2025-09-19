#!/usr/bin/env python3
"""
Advanced Feature Engineering for Chemical Geometry Study
Implements multiple molecular fingerprinting methods and descriptors
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MolecularFeatureEngineer:
    """Advanced molecular feature engineering class"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def calculate_rdkit_descriptors(self, mol):
        """Calculate comprehensive RDKit molecular descriptors"""
        descriptors = {}
        
        # Basic molecular properties
        descriptors['MolWt'] = Descriptors.MolWt(mol)
        descriptors['LogP'] = Descriptors.MolLogP(mol)
        descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
        descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
        descriptors['TPSA'] = Descriptors.TPSA(mol)
        descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
        descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
        descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
        descriptors['RingCount'] = Descriptors.RingCount(mol)
        
        # Connectivity and complexity
        descriptors['BertzCT'] = Descriptors.BertzCT(mol)
        descriptors['Chi0v'] = Descriptors.Chi0v(mol)
        descriptors['Chi1v'] = Descriptors.Chi1v(mol)
        descriptors['Chi2v'] = Descriptors.Chi2v(mol)
        descriptors['Chi3v'] = Descriptors.Chi3v(mol)
        descriptors['Chi4v'] = Descriptors.Chi4v(mol)
        descriptors['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
        descriptors['Kappa1'] = Descriptors.Kappa1(mol)
        descriptors['Kappa2'] = Descriptors.Kappa2(mol)
        descriptors['Kappa3'] = Descriptors.Kappa3(mol)
        
        # Electronic properties
        descriptors['MaxEStateIndex'] = Descriptors.MaxEStateIndex(mol)
        descriptors['MinEStateIndex'] = Descriptors.MinEStateIndex(mol)
        descriptors['MaxAbsEStateIndex'] = Descriptors.MaxAbsEStateIndex(mol)
        descriptors['MinAbsEStateIndex'] = Descriptors.MinAbsEStateIndex(mol)
        descriptors['qed'] = Descriptors.qed(mol)
        
        # Surface area and volume
        descriptors['LabuteASA'] = Descriptors.LabuteASA(mol)
        descriptors['PEOE_VSA1'] = Descriptors.PEOE_VSA1(mol)
        descriptors['PEOE_VSA2'] = Descriptors.PEOE_VSA2(mol)
        descriptors['PEOE_VSA3'] = Descriptors.PEOE_VSA3(mol)
        descriptors['PEOE_VSA4'] = Descriptors.PEOE_VSA4(mol)
        descriptors['PEOE_VSA5'] = Descriptors.PEOE_VSA5(mol)
        descriptors['PEOE_VSA6'] = Descriptors.PEOE_VSA6(mol)
        
        # Pharmacophore features
        descriptors['fr_NH0'] = Descriptors.fr_NH0(mol)
        descriptors['fr_NH1'] = Descriptors.fr_NH1(mol)
        descriptors['fr_NH2'] = Descriptors.fr_NH2(mol)
        descriptors['fr_N_O'] = Descriptors.fr_N_O(mol)
        descriptors['fr_C_O'] = Descriptors.fr_C_O(mol)
        descriptors['fr_C_O_noCOO'] = Descriptors.fr_C_O_noCOO(mol)
        descriptors['fr_Al_COO'] = Descriptors.fr_Al_COO(mol)
        descriptors['fr_Ar_COO'] = Descriptors.fr_Ar_COO(mol)
        descriptors['fr_COO'] = Descriptors.fr_COO(mol)
        descriptors['fr_COO2'] = Descriptors.fr_COO2(mol)
        
        return descriptors
    
    def calculate_morgan_fingerprint(self, mol, radius=2, nBits=2048):
        """Calculate Morgan (ECFP) fingerprint"""
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(fp)
    
    def calculate_topological_fingerprint(self, mol, nBits=2048):
        """Calculate RDKit topological fingerprint"""
        fp = Chem.RDKFingerprint(mol, fpSize=nBits)
        return np.array(fp)
    
    def calculate_maccs_keys(self, mol):
        """Calculate MACCS keys fingerprint"""
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        return np.array(fp)
    
    def calculate_atom_pair_fingerprint(self, mol, nBits=2048):
        """Calculate atom pair fingerprint"""
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)
        return np.array(fp)
    
    def calculate_topological_torsion_fingerprint(self, mol, nBits=2048):
        """Calculate topological torsion fingerprint"""
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits)
        return np.array(fp)
    
    def calculate_geometric_features(self, mol):
        """Calculate 3D geometric features if available"""
        features = {}
        
        try:
            # Try to generate 3D coordinates
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            AllChem.OptimizeMolecule(mol_3d)
            
            # Calculate 3D descriptors
            features['Asphericity'] = rdMolDescriptors.CalcAsphericity(mol_3d)
            features['Eccentricity'] = rdMolDescriptors.CalcEccentricity(mol_3d)
            features['InertialShapeFactor'] = rdMolDescriptors.CalcInertialShapeFactor(mol_3d)
            features['NPR1'] = rdMolDescriptors.CalcNPR1(mol_3d)
            features['NPR2'] = rdMolDescriptors.CalcNPR2(mol_3d)
            features['PMI1'] = rdMolDescriptors.CalcPMI1(mol_3d)
            features['PMI2'] = rdMolDescriptors.CalcPMI2(mol_3d)
            features['PMI3'] = rdMolDescriptors.CalcPMI3(mol_3d)
            features['RadiusOfGyration'] = rdMolDescriptors.CalcRadiusOfGyration(mol_3d)
            features['SpherocityIndex'] = rdMolDescriptors.CalcSpherocityIndex(mol_3d)
            
        except:
            # If 3D generation fails, use default values
            features = {
                'Asphericity': 0.0,
                'Eccentricity': 0.0,
                'InertialShapeFactor': 0.0,
                'NPR1': 0.0,
                'NPR2': 0.0,
                'PMI1': 0.0,
                'PMI2': 0.0,
                'PMI3': 0.0,
                'RadiusOfGyration': 0.0,
                'SpherocityIndex': 0.0
            }
        
        return features
    
    def process_dataset(self, df, include_fingerprints=True, include_3d=True):
        """Process the entire dataset with comprehensive feature engineering"""
        print("Starting comprehensive feature engineering...")
        
        # Initialize feature matrices
        rdkit_features = []
        morgan_fps = []
        topo_fps = []
        maccs_fps = []
        atom_pair_fps = []
        torsion_fps = []
        geometric_features = []
        
        valid_indices = []
        
        for idx, row in df.iterrows():
            smiles = row['canonical_smiles']
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Calculate RDKit descriptors
                rdkit_desc = self.calculate_rdkit_descriptors(mol)
                rdkit_features.append(rdkit_desc)
                
                if include_fingerprints:
                    # Calculate fingerprints
                    morgan_fps.append(self.calculate_morgan_fingerprint(mol))
                    topo_fps.append(self.calculate_topological_fingerprint(mol))
                    maccs_fps.append(self.calculate_maccs_keys(mol))
                    atom_pair_fps.append(self.calculate_atom_pair_fingerprint(mol))
                    torsion_fps.append(self.calculate_topological_torsion_fingerprint(mol))
                
                if include_3d:
                    # Calculate 3D geometric features
                    geom_features = self.calculate_geometric_features(mol)
                    geometric_features.append(geom_features)
                
                valid_indices.append(idx)
                
                if len(valid_indices) % 500 == 0:
                    print(f"Processed {len(valid_indices)} compounds...")
                    
            except Exception as e:
                print(f"Error processing compound at index {idx}: {str(e)}")
                continue
        
        print(f"Successfully processed {len(valid_indices)} compounds")
        
        # Convert to DataFrames
        rdkit_df = pd.DataFrame(rdkit_features)
        
        # Combine all features
        feature_df = rdkit_df.copy()
        
        if include_3d and geometric_features:
            geom_df = pd.DataFrame(geometric_features)
            feature_df = pd.concat([feature_df, geom_df], axis=1)
        
        # Add fingerprints as separate arrays (for later use)
        fingerprint_data = {}
        if include_fingerprints:
            fingerprint_data['morgan'] = np.array(morgan_fps)
            fingerprint_data['topological'] = np.array(topo_fps)
            fingerprint_data['maccs'] = np.array(maccs_fps)
            fingerprint_data['atom_pair'] = np.array(atom_pair_fps)
            fingerprint_data['torsion'] = np.array(torsion_fps)
        
        # Filter original dataframe to valid compounds
        filtered_df = df.iloc[valid_indices].reset_index(drop=True)
        
        # Combine with original data
        result_df = pd.concat([filtered_df, feature_df], axis=1)
        
        # Store feature names
        self.feature_names = list(feature_df.columns)
        
        print(f"Feature engineering complete. Generated {len(self.feature_names)} molecular descriptors")
        
        return result_df, fingerprint_data

def main():
    """Main execution function"""
    print("Loading kinase compounds dataset...")
    df = pd.read_csv('kinase_compounds_dataset.csv')
    print(f"Loaded {len(df)} compounds")
    
    # Initialize feature engineer
    engineer = MolecularFeatureEngineer()
    
    # Process dataset with comprehensive features
    processed_df, fingerprints = engineer.process_dataset(
        df, 
        include_fingerprints=True, 
        include_3d=True
    )
    
    # Save processed dataset
    processed_df.to_csv('kinase_compounds_features.csv', index=False)
    print(f"Saved processed dataset with {len(processed_df)} compounds and {len(engineer.feature_names)} features")
    
    # Save fingerprints separately
    np.savez('molecular_fingerprints.npz', **fingerprints)
    print("Saved molecular fingerprints to molecular_fingerprints.npz")
    
    # Print summary statistics
    print("\nFeature Engineering Summary:")
    print(f"Original compounds: {len(df)}")
    print(f"Successfully processed: {len(processed_df)}")
    print(f"Success rate: {len(processed_df)/len(df)*100:.1f}%")
    print(f"Total molecular descriptors: {len(engineer.feature_names)}")
    print(f"Fingerprint types: {len(fingerprints)}")
    
    # Display feature categories
    print("\nFeature categories:")
    basic_features = [f for f in engineer.feature_names if any(x in f for x in ['MolWt', 'LogP', 'TPSA', 'Num'])]
    connectivity_features = [f for f in engineer.feature_names if any(x in f for x in ['Chi', 'Kappa', 'Bertz'])]
    electronic_features = [f for f in engineer.feature_names if any(x in f for x in ['EState', 'qed', 'PEOE'])]
    pharmacophore_features = [f for f in engineer.feature_names if f.startswith('fr_')]
    geometric_features = [f for f in engineer.feature_names if any(x in f for x in ['Asphericity', 'PMI', 'NPR'])]
    
    print(f"  Basic properties: {len(basic_features)}")
    print(f"  Connectivity: {len(connectivity_features)}")
    print(f"  Electronic: {len(electronic_features)}")
    print(f"  Pharmacophore: {len(pharmacophore_features)}")
    print(f"  3D Geometric: {len(geometric_features)}")

if __name__ == "__main__":
    main()
