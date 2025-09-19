#!/usr/bin/env python3
"""
Version 2 Comprehensive Molecular Feature Engineering Pipeline
Advanced feature generation for geometric mapping study

Features generated:
1. ECFP4 fingerprints (2048 bits) - for ML prediction
2. Mordred descriptors - for interpretability
3. RDKit descriptors - for chemical space characterization
4. Custom geometric features - for resonance analysis
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from mordred import Calculator, descriptors
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveFeatureEngineering:
    """Advanced molecular feature engineering for Version 2 study"""
    
    def __init__(self):
        self.mordred_calc = Calculator(descriptors, ignore_3D=True)
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.pi = np.pi
        self.sqrt2 = np.sqrt(2)
        self.e = np.e
        
    def load_dataset(self, filename="chembl_v2_drd2_dataset.csv"):
        """Load the ChEMBL dataset"""
        print(f"Loading dataset: {filename}")
        df = pd.read_csv(filename)
        print(f"Dataset loaded: {len(df)} compounds")
        return df
    
    def validate_smiles(self, df):
        """Validate and clean SMILES strings"""
        print("Validating SMILES strings...")
        
        valid_mols = []
        valid_indices = []
        
        for idx, smiles in enumerate(df['canonical_smiles']):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Additional validation
                    if mol.GetNumAtoms() > 5:  # Minimum atom count
                        valid_mols.append(mol)
                        valid_indices.append(idx)
            except:
                continue
        
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        print(f"Valid SMILES: {len(valid_mols)}/{len(df)} compounds")
        
        return df_valid, valid_mols
    
    def generate_ecfp4_fingerprints(self, mols):
        """Generate ECFP4 fingerprints for ML prediction"""
        print("Generating ECFP4 fingerprints (2048 bits)...")
        
        fingerprints = []
        for mol in mols:
            try:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=2048
                )
                fp_array = np.array(fp)
                fingerprints.append(fp_array)
            except:
                # Fallback to zero vector
                fingerprints.append(np.zeros(2048))
        
        fingerprints_array = np.array(fingerprints)
        print(f"ECFP4 fingerprints generated: {fingerprints_array.shape}")
        
        return fingerprints_array
    
    def generate_rdkit_descriptors(self, mols):
        """Generate comprehensive RDKit descriptors"""
        print("Generating RDKit molecular descriptors...")
        
        descriptor_data = []
        
        for mol in mols:
            try:
                desc_dict = {
                    # Basic molecular properties
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Crippen.MolLogP(mol),
                    'hbd': Lipinski.NumHDonors(mol),
                    'hba': Lipinski.NumHAcceptors(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    
                    # Ring and aromatic properties
                    'num_rings': Descriptors.RingCount(mol),
                    'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                    'num_saturated_rings': Descriptors.NumSaturatedRings(mol),
                    'num_aliphatic_rings': Descriptors.NumAliphaticRings(mol),
                    
                    # Atom counts
                    'num_atoms': mol.GetNumAtoms(),
                    'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                    'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
                    
                    # Connectivity and complexity
                    'bertz_ct': Descriptors.BertzCT(mol),
                    'balaban_j': Descriptors.BalabanJ(mol),
                    'kappa1': Descriptors.Kappa1(mol),
                    'kappa2': Descriptors.Kappa2(mol),
                    'kappa3': Descriptors.Kappa3(mol),
                    
                    # Electronic properties
                    'max_partial_charge': Descriptors.MaxPartialCharge(mol),
                    'min_partial_charge': Descriptors.MinPartialCharge(mol),
                    
                    # Lipophilicity and solubility
                    'slogp': Descriptors.SlogP_VSA1(mol),
                    'smr': Descriptors.SMR_VSA1(mol),
                    
                    # Pharmacophore features
                    'num_pharmacophore_features': len(Chem.rdMolChemicalFeatures.GetFeaturesForMol(mol, 'Pharmacophore')),
                    
                    # Molecular formula features
                    'formula': CalcMolFormula(mol),
                }
                
                descriptor_data.append(desc_dict)
                
            except Exception as e:
                # Fallback with NaN values
                descriptor_data.append({key: np.nan for key in [
                    'molecular_weight', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds',
                    'num_rings', 'num_aromatic_rings', 'num_saturated_rings', 'num_aliphatic_rings',
                    'num_atoms', 'num_heavy_atoms', 'num_heteroatoms',
                    'bertz_ct', 'balaban_j', 'kappa1', 'kappa2', 'kappa3',
                    'max_partial_charge', 'min_partial_charge',
                    'slogp', 'smr', 'num_pharmacophore_features', 'formula'
                ]})
        
        descriptors_df = pd.DataFrame(descriptor_data)
        print(f"RDKit descriptors generated: {descriptors_df.shape}")
        
        return descriptors_df
    
    def generate_mordred_descriptors(self, mols, max_descriptors=100):
        """Generate Mordred descriptors (subset for efficiency)"""
        print(f"Generating Mordred descriptors (top {max_descriptors})...")
        
        try:
            # Calculate all descriptors
            mordred_data = self.mordred_calc.pandas(mols)
            
            # Remove descriptors with too many missing values
            missing_threshold = 0.1  # Allow up to 10% missing
            valid_cols = []
            
            for col in mordred_data.columns:
                missing_ratio = mordred_data[col].isna().sum() / len(mordred_data)
                if missing_ratio <= missing_threshold:
                    valid_cols.append(col)
            
            mordred_filtered = mordred_data[valid_cols]
            
            # Select most informative descriptors (highest variance)
            numeric_cols = mordred_filtered.select_dtypes(include=[np.number]).columns
            variances = mordred_filtered[numeric_cols].var().sort_values(ascending=False)
            
            top_descriptors = variances.head(max_descriptors).index.tolist()
            mordred_final = mordred_filtered[top_descriptors]
            
            print(f"Mordred descriptors generated: {mordred_final.shape}")
            return mordred_final
            
        except Exception as e:
            print(f"Error generating Mordred descriptors: {e}")
            # Return empty DataFrame with correct index
            return pd.DataFrame(index=range(len(mols)))
    
    def generate_geometric_features(self, descriptors_df):
        """Generate custom geometric features for resonance analysis"""
        print("Generating custom geometric features...")
        
        geometric_features = []
        
        for idx, row in descriptors_df.iterrows():
            try:
                # Extract key molecular properties
                mw = row.get('molecular_weight', 300)
                logp = row.get('logp', 2)
                tpsa = row.get('tpsa', 60)
                num_atoms = row.get('num_atoms', 20)
                
                # Normalize properties to [0, 1] range
                mw_norm = min(mw / 1000, 1.0)  # Normalize by 1000 Da
                logp_norm = (logp + 5) / 10  # Shift and normalize LogP
                tpsa_norm = min(tpsa / 200, 1.0)  # Normalize by 200 Ų
                atoms_norm = min(num_atoms / 100, 1.0)  # Normalize by 100 atoms
                
                # Calculate geometric resonance features
                features = {
                    # Sacred geometry resonances
                    'phi_resonance': abs(mw_norm - self.phi/3),  # Golden ratio resonance
                    'pi_resonance': abs(logp_norm - self.pi/4),  # Pi resonance
                    'sqrt2_resonance': abs(tpsa_norm - self.sqrt2/2),  # Square root 2 resonance
                    'e_resonance': abs(atoms_norm - self.e/3),  # Euler's number resonance
                    
                    # Composite geometric features
                    'geometric_complexity': mw_norm * logp_norm * tpsa_norm,
                    'molecular_symmetry': abs(mw_norm - atoms_norm),
                    'lipophilic_balance': abs(logp_norm - 0.5),
                    'polar_surface_ratio': tpsa_norm / (mw_norm + 0.001),
                    
                    # Harmonic features
                    'harmonic_mean': 4 / (1/mw_norm + 1/logp_norm + 1/tpsa_norm + 1/atoms_norm + 0.001),
                    'geometric_mean': (mw_norm * logp_norm * tpsa_norm * atoms_norm) ** 0.25,
                    
                    # Resonance combinations
                    'phi_pi_interaction': abs(mw_norm * self.phi - logp_norm * self.pi),
                    'sqrt2_e_interaction': abs(tpsa_norm * self.sqrt2 - atoms_norm * self.e),
                }
                
                geometric_features.append(features)
                
            except Exception as e:
                # Fallback with default values
                geometric_features.append({
                    'phi_resonance': 0.5, 'pi_resonance': 0.5, 'sqrt2_resonance': 0.5, 'e_resonance': 0.5,
                    'geometric_complexity': 0.1, 'molecular_symmetry': 0.5, 'lipophilic_balance': 0.5,
                    'polar_surface_ratio': 0.3, 'harmonic_mean': 0.2, 'geometric_mean': 0.2,
                    'phi_pi_interaction': 0.1, 'sqrt2_e_interaction': 0.1
                })
        
        geometric_df = pd.DataFrame(geometric_features)
        print(f"Geometric features generated: {geometric_df.shape}")
        
        return geometric_df
    
    def combine_features(self, df_original, rdkit_desc, mordred_desc, geometric_feat, ecfp4_fps):
        """Combine all features into comprehensive dataset"""
        print("Combining all features...")
        
        # Start with original data
        df_combined = df_original.copy()
        
        # Add RDKit descriptors
        for col in rdkit_desc.columns:
            if col != 'formula':  # Skip non-numeric formula
                df_combined[f'rdkit_{col}'] = rdkit_desc[col]
        
        # Add Mordred descriptors
        for col in mordred_desc.columns:
            df_combined[f'mordred_{col}'] = mordred_desc[col]
        
        # Add geometric features
        for col in geometric_feat.columns:
            df_combined[f'geom_{col}'] = geometric_feat[col]
        
        # Save ECFP4 fingerprints separately (too many columns)
        np.save('v2_ecfp4_fingerprints.npy', ecfp4_fps)
        
        print(f"Combined dataset: {df_combined.shape}")
        print(f"ECFP4 fingerprints saved separately: {ecfp4_fps.shape}")
        
        return df_combined
    
    def save_features(self, df_combined, filename="v2_comprehensive_features.csv"):
        """Save the comprehensive feature dataset"""
        print(f"Saving comprehensive features to {filename}")
        
        # Add feature engineering metadata
        df_combined['feature_version'] = 'v2.0_comprehensive'
        df_combined['feature_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Save to CSV
        df_combined.to_csv(filename, index=False)
        
        print(f"Features saved: {len(df_combined)} compounds, {len(df_combined.columns)} features")
        
        # Generate feature summary
        numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
        print(f"Numeric features: {len(numeric_cols)}")
        print(f"Missing values: {df_combined[numeric_cols].isna().sum().sum()}")
        
        return df_combined

def main():
    """Main execution function"""
    print("="*80)
    print("VERSION 2 COMPREHENSIVE FEATURE ENGINEERING")
    print("="*80)
    
    # Initialize feature engineering
    feature_eng = ComprehensiveFeatureEngineering()
    
    # Load dataset
    df = feature_eng.load_dataset()
    
    # Validate SMILES
    df_valid, mols = feature_eng.validate_smiles(df)
    
    if len(mols) < 100:
        print(f"Insufficient valid molecules: {len(mols)}")
        return None
    
    # Generate ECFP4 fingerprints
    ecfp4_fingerprints = feature_eng.generate_ecfp4_fingerprints(mols)
    
    # Generate RDKit descriptors
    rdkit_descriptors = feature_eng.generate_rdkit_descriptors(mols)
    
    # Generate Mordred descriptors
    mordred_descriptors = feature_eng.generate_mordred_descriptors(mols, max_descriptors=50)
    
    # Generate geometric features
    geometric_features = feature_eng.generate_geometric_features(rdkit_descriptors)
    
    # Combine all features
    df_comprehensive = feature_eng.combine_features(
        df_valid, rdkit_descriptors, mordred_descriptors, 
        geometric_features, ecfp4_fingerprints
    )
    
    # Save comprehensive dataset
    final_dataset = feature_eng.save_features(df_comprehensive)
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print(f"Final dataset: {len(final_dataset)} compounds")
    print(f"Total features: {len(final_dataset.columns)}")
    print(f"RDKit descriptors: ~23")
    print(f"Mordred descriptors: {len([c for c in final_dataset.columns if 'mordred_' in c])}")
    print(f"Geometric features: {len([c for c in final_dataset.columns if 'geom_' in c])}")
    print(f"ECFP4 fingerprints: 2048 (saved separately)")
    print("Ready for geometric mapping and resonance analysis!")
    
    return final_dataset

if __name__ == "__main__":
    dataset = main()
