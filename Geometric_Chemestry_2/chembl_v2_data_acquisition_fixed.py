#!/usr/bin/env python3
"""
ChEMBL Version 2 Data Acquisition Script - Fixed Version
Comprehensive real biological data collection for geometric mapping study

Target: Dopamine D2 Receptor (DRD2) - well-studied GPCR with extensive ChEMBL data
Goal: N=500-1000 compounds with real IC50/Ki values
"""

import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
import time
import warnings
warnings.filterwarnings('ignore')

class ChEMBLDataAcquisition:
    """Comprehensive ChEMBL data acquisition for Version 2 study"""
    
    def __init__(self):
        self.target_client = new_client.target
        self.activity_client = new_client.activity
        self.molecule_client = new_client.molecule
        self.assay_client = new_client.assay
        
    def search_target(self, target_name="dopamine D2"):
        """Search for target and return ChEMBL ID"""
        print(f"Searching for target: {target_name}")
        
        try:
            targets = self.target_client.search(target_name)
            
            if not targets:
                print(f"No targets found for '{target_name}'")
                return None
                
            # Filter for human targets and select the most relevant
            human_targets = [t for t in targets if 'Homo sapiens' in str(t.get('organism', ''))]
            
            if human_targets:
                target = human_targets[0]
            else:
                target = targets[0]
                
            print(f"Selected target: {target['pref_name']} ({target['target_chembl_id']})")
            print(f"Organism: {target.get('organism', 'Unknown')}")
            print(f"Target type: {target.get('target_type', 'Unknown')}")
            
            return target['target_chembl_id']
            
        except Exception as e:
            print(f"Error searching for target: {e}")
            return None
    
    def fetch_activities(self, target_chembl_id, min_compounds=500):
        """Fetch activities for the target with comprehensive filtering"""
        print(f"Fetching activities for target: {target_chembl_id}")
        
        # Define activity types of interest (binding affinity measures)
        activity_types = ['IC50', 'Ki', 'Kd', 'EC50', 'AC50']
        
        all_activities = []
        
        for activity_type in activity_types:
            print(f"  Fetching {activity_type} data...")
            
            try:
                # Fetch activities with comprehensive filters
                activities = self.activity_client.filter(
                    target_chembl_id=target_chembl_id,
                    standard_type=activity_type,
                    standard_relation='=',  # Only exact values
                    assay_type='B'  # Binding assays
                ).only([
                    'molecule_chembl_id',
                    'canonical_smiles', 
                    'standard_value',
                    'standard_units',
                    'standard_type',
                    'standard_relation',
                    'assay_chembl_id',
                    'assay_description',
                    'data_validity_comment'
                ])
                
                activity_list = list(activities)
                print(f"    Found {len(activity_list)} {activity_type} records")
                all_activities.extend(activity_list)
                
                # Add small delay to be respectful to ChEMBL servers
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    Error fetching {activity_type} data: {e}")
                continue
        
        print(f"Total activities fetched: {len(all_activities)}")
        
        if len(all_activities) < min_compounds:
            print(f"Warning: Only {len(all_activities)} activities found, less than target {min_compounds}")
        
        return all_activities
    
    def clean_and_process_data(self, activities):
        """Clean and process the activity data"""
        print("Cleaning and processing activity data...")
        
        # Convert to DataFrame
        df = pd.DataFrame(activities)
        
        if df.empty:
            print("No data to process")
            return None
        
        print(f"Initial dataset size: {len(df)} records")
        
        # Remove records without SMILES or activity values
        df = df.dropna(subset=['canonical_smiles', 'standard_value'])
        print(f"After removing missing SMILES/values: {len(df)} records")
        
        # Remove records with invalid SMILES (too short or obviously invalid)
        df = df[df['canonical_smiles'].str.len() > 5]
        print(f"After SMILES length filter: {len(df)} records")
        
        # Convert standard_value to numeric
        df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
        df = df.dropna(subset=['standard_value'])
        
        # Filter for reasonable activity ranges (1 nM to 100 μM)
        df = df[(df['standard_value'] >= 1) & (df['standard_value'] <= 100000)]
        print(f"After activity range filter (1 nM - 100 μM): {len(df)} records")
        
        # Convert to pIC50/pKi values (negative log10 of IC50/Ki in M)
        df['pActivity'] = -np.log10(df['standard_value'] * 1e-9)  # Convert nM to M, then -log10
        
        # Handle confidence score if available, otherwise use data validity
        if 'confidence_score' in df.columns:
            df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce')
            df = df.sort_values('confidence_score', ascending=False, na_position='last')
        else:
            # Use data validity as a proxy for quality
            df['quality_score'] = df['data_validity_comment'].apply(
                lambda x: 1 if pd.isna(x) or x == '' else 0
            )
            df = df.sort_values('quality_score', ascending=False)
        
        # Remove duplicates based on SMILES (keep the highest quality one)
        df = df.drop_duplicates(subset=['canonical_smiles'], keep='first')
        print(f"After removing SMILES duplicates: {len(df)} records")
        
        # Filter for high-quality data
        if 'confidence_score' in df.columns:
            df = df[df['confidence_score'] >= 7]  # ChEMBL confidence score >= 7
            print(f"After confidence filter (≥7): {len(df)} records")
        else:
            # Filter based on data validity
            df = df[df['quality_score'] == 1]  # No validity issues
            print(f"After data validity filter: {len(df)} records")
        
        # Add activity classification
        df['activity_class'] = pd.cut(df['pActivity'], 
                                    bins=[0, 5, 6, 7, 8, 15], 
                                    labels=['Inactive', 'Low', 'Moderate', 'High', 'Very High'])
        
        return df
    
    def enrich_molecular_data(self, df):
        """Enrich with additional molecular information"""
        print("Enriching with molecular information...")
        
        # Get unique molecule ChEMBL IDs
        molecule_ids = df['molecule_chembl_id'].unique()
        
        enriched_data = []
        
        for i, mol_id in enumerate(molecule_ids):
            if i % 50 == 0:
                print(f"  Processing molecule {i+1}/{len(molecule_ids)}")
            
            try:
                # Get molecular properties
                molecule = self.molecule_client.get(mol_id)
                
                if molecule:
                    mol_props = molecule.get('molecule_properties', {})
                    mol_data = {
                        'molecule_chembl_id': mol_id,
                        'molecular_weight': mol_props.get('mw_freebase'),
                        'logp': mol_props.get('alogp'),
                        'hbd': mol_props.get('hbd'),
                        'hba': mol_props.get('hba'),
                        'psa': mol_props.get('psa'),
                        'rtb': mol_props.get('rtb'),
                        'ro5_violations': mol_props.get('num_ro5_violations'),
                        'molecule_type': molecule.get('molecule_type'),
                        'max_phase': molecule.get('max_phase')
                    }
                    enriched_data.append(mol_data)
                
                # Be respectful to ChEMBL servers
                if i % 10 == 0:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"    Error processing molecule {mol_id}: {e}")
                continue
        
        # Merge with main dataframe
        enriched_df = pd.DataFrame(enriched_data)
        df_final = df.merge(enriched_df, on='molecule_chembl_id', how='left')
        
        return df_final
    
    def save_dataset(self, df, filename="chembl_v2_dataset.csv"):
        """Save the final dataset"""
        print(f"Saving dataset to {filename}")
        
        # Add metadata
        df['dataset_version'] = 'v2.0'
        df['acquisition_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Reorder columns for clarity
        column_order = [
            'molecule_chembl_id', 'canonical_smiles', 'standard_type', 'standard_value', 
            'pActivity', 'activity_class', 'molecular_weight', 'logp', 'hbd', 'hba', 
            'psa', 'rtb', 'ro5_violations', 'molecule_type', 'max_phase',
            'assay_chembl_id', 'dataset_version', 'acquisition_date'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        df_final = df[available_columns]
        
        df_final.to_csv(filename, index=False)
        
        print(f"Dataset saved: {len(df_final)} compounds")
        print(f"Activity range: {df_final['pActivity'].min():.2f} - {df_final['pActivity'].max():.2f}")
        print(f"Activity distribution:")
        print(df_final['activity_class'].value_counts())
        
        return df_final

def main():
    """Main execution function"""
    print("="*80)
    print("ChEMBL Version 2 Data Acquisition - Fixed Version")
    print("Target: Dopamine D2 Receptor (DRD2)")
    print("Goal: N=500-1000 compounds with real biological activity")
    print("="*80)
    
    # Initialize acquisition system
    acquisition = ChEMBLDataAcquisition()
    
    # Search for target
    target_id = acquisition.search_target("dopamine D2")
    
    if not target_id:
        print("Failed to find target. Trying alternative search...")
        target_id = acquisition.search_target("DRD2")
    
    if not target_id:
        print("Could not find Dopamine D2 receptor. Exiting.")
        return None
    
    # Fetch activities
    activities = acquisition.fetch_activities(target_id, min_compounds=500)
    
    if not activities:
        print("No activities found. Exiting.")
        return None
    
    # Clean and process data
    df_clean = acquisition.clean_and_process_data(activities)
    
    if df_clean is None or len(df_clean) < 100:
        print(f"Insufficient data after cleaning: {len(df_clean) if df_clean is not None else 0} compounds")
        return None
    
    # Enrich with molecular data
    df_final = acquisition.enrich_molecular_data(df_clean)
    
    # Save dataset
    final_dataset = acquisition.save_dataset(df_final)
    
    print("\n" + "="*80)
    print("DATA ACQUISITION SUMMARY")
    print("="*80)
    print(f"Final dataset size: {len(final_dataset)} compounds")
    print(f"Target: {target_id}")
    print(f"Activity types: {final_dataset['standard_type'].value_counts().to_dict()}")
    if 'molecular_weight' in final_dataset.columns:
        print(f"Molecular weight range: {final_dataset['molecular_weight'].min():.1f} - {final_dataset['molecular_weight'].max():.1f}")
    if 'logp' in final_dataset.columns:
        print(f"LogP range: {final_dataset['logp'].min():.2f} - {final_dataset['logp'].max():.2f}")
    print("Data quality: High-confidence ChEMBL records only")
    print("Ready for geometric mapping analysis!")
    
    return final_dataset

if __name__ == "__main__":
    dataset = main()
