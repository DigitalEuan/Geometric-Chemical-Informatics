#!/usr/bin/env python3
"""
Simplified Materials Project Data Acquisition
Focus on getting working data quickly for UBP analysis
"""

import pandas as pd
import numpy as np
from mp_api.client import MPRester
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json
import warnings
warnings.filterwarnings('ignore')

def fetch_materials_data():
    """Simplified approach to fetch materials data"""
    
    api_key = "xxx"
    
    print("="*80)
    print("SIMPLIFIED MATERIALS PROJECT ACQUISITION")
    print("="*80)
    print("Target: Pure inorganic compounds (Binary/Ternary)")
    print("Focus: Transition metals + main group elements")
    print("Crystal systems: Cubic and Hexagonal")
    print()
    
    compounds_data = []
    
    with MPRester(api_key) as mpr:
        
        # Step 1: Simple search for transition metal compounds
        print("Step 1: Searching for transition metal compounds...")
        
        try:
            # Start with a simple search - just Fe compounds first
            materials_docs = mpr.materials.search(
                elements=["Fe"],  # Start simple
                exclude_elements=["C", "H", "N", "O"],  # No organics
                num_elements=(2, 3),  # Binary and ternary
                crystal_system=["Cubic", "Hexagonal"],  # Proper capitalization
                fields=["material_id", "formula_pretty", "structure", "symmetry", 
                       "volume", "density", "nsites"]
            )
            
            print(f"✅ Found {len(materials_docs)} Fe-containing compounds")
            
            if len(materials_docs) == 0:
                print("❌ No compounds found!")
                return pd.DataFrame()
            
            # Limit to reasonable number for testing
            if len(materials_docs) > 100:
                materials_docs = materials_docs[:100]
                print(f"📊 Limited to {len(materials_docs)} compounds for testing")
            
            # Get material IDs for property queries
            material_ids = [doc.material_id for doc in materials_docs]
            
        except Exception as e:
            print(f"❌ Error in basic search: {e}")
            return pd.DataFrame()
        
        # Step 2: Get additional properties
        print("Step 2: Fetching additional properties...")
        
        # Get thermodynamic data
        thermo_data = {}
        try:
            thermo_docs = mpr.materials.thermo.search(material_ids=material_ids)
            for doc in thermo_docs:
                thermo_data[doc.material_id] = {
                    'formation_energy_per_atom': getattr(doc, 'formation_energy_per_atom', np.nan),
                    'energy_above_hull': getattr(doc, 'energy_above_hull', np.nan)
                }
            print(f"  ✅ Thermodynamic data: {len(thermo_data)} compounds")
        except Exception as e:
            print(f"  ⚠️  Thermo error: {e}")
        
        # Get electronic structure data
        electronic_data = {}
        try:
            es_docs = mpr.materials.electronic_structure.search(material_ids=material_ids)
            for doc in es_docs:
                electronic_data[doc.material_id] = {
                    'band_gap': getattr(doc, 'band_gap', np.nan)
                }
            print(f"  ✅ Electronic data: {len(electronic_data)} compounds")
        except Exception as e:
            print(f"  ⚠️  Electronic error: {e}")
        
        # Get magnetism data
        magnetism_data = {}
        try:
            mag_docs = mpr.materials.magnetism.search(material_ids=material_ids)
            for doc in mag_docs:
                magnetism_data[doc.material_id] = {
                    'total_magnetization': getattr(doc, 'total_magnetization', np.nan)
                }
            print(f"  ✅ Magnetism data: {len(magnetism_data)} compounds")
        except Exception as e:
            print(f"  ⚠️  Magnetism error: {e}")
        
        # Step 3: Process compounds
        print("Step 3: Processing compound data...")
        
        for i, doc in enumerate(materials_docs):
            try:
                # Basic data
                compound_data = {
                    'material_id': doc.material_id,
                    'formula': doc.formula_pretty,
                    'nsites': doc.nsites,
                    'volume': doc.volume,
                    'density': doc.density
                }
                
                # Add property data
                if doc.material_id in thermo_data:
                    compound_data.update(thermo_data[doc.material_id])
                else:
                    compound_data['formation_energy_per_atom'] = np.nan
                    compound_data['energy_above_hull'] = np.nan
                
                if doc.material_id in electronic_data:
                    compound_data.update(electronic_data[doc.material_id])
                else:
                    compound_data['band_gap'] = np.nan
                
                if doc.material_id in magnetism_data:
                    compound_data.update(magnetism_data[doc.material_id])
                else:
                    compound_data['total_magnetization'] = np.nan
                
                # Structural analysis
                structure = doc.structure
                sga = SpacegroupAnalyzer(structure)
                
                compound_data['spacegroup_symbol'] = sga.get_space_group_symbol()
                compound_data['spacegroup_number'] = sga.get_space_group_number()
                compound_data['crystal_system'] = sga.get_crystal_system()
                
                # Lattice parameters
                lattice = structure.lattice
                compound_data['a'] = lattice.a
                compound_data['b'] = lattice.b
                compound_data['c'] = lattice.c
                compound_data['volume_per_atom'] = structure.volume / len(structure)
                
                # Simple coordination analysis
                tm_elements = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
                tm_sites = [i for i, site in enumerate(structure.sites) 
                           if str(site.specie) in tm_elements]
                
                if tm_sites:
                    compound_data['tm_element'] = str(structure[tm_sites[0]].specie)
                    # Simple coordination number estimation
                    compound_data['coordination_number'] = 6  # Default for now
                else:
                    compound_data['tm_element'] = None
                    compound_data['coordination_number'] = None
                
                compounds_data.append(compound_data)
                
            except Exception as e:
                print(f"  ⚠️  Error processing {doc.material_id}: {e}")
                continue
        
        print(f"✅ Successfully processed {len(compounds_data)} compounds")
        
        # Create DataFrame
        df = pd.DataFrame(compounds_data)
        
        # Print summary
        print("\\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Total compounds: {len(df)}")
        print(f"Unique formulas: {df['formula'].nunique()}")
        
        if 'crystal_system' in df.columns:
            print(f"Crystal systems: {df['crystal_system'].value_counts().to_dict()}")
        
        # Property coverage
        properties = ['formation_energy_per_atom', 'band_gap', 'total_magnetization']
        print("\\nProperty coverage:")
        for prop in properties:
            if prop in df.columns:
                non_null = df[prop].notna().sum()
                print(f"  {prop}: {non_null}/{len(df)} ({100*non_null/len(df):.1f}%)")
        
        # Save dataset
        filename = "mp_simple_inorganic_dataset.csv"
        df.to_csv(filename, index=False)
        print(f"\\n💾 Saved dataset to {filename}")
        
        return df

def expand_dataset(initial_df):
    """Expand the dataset with more elements"""
    
    if len(initial_df) == 0:
        print("❌ Cannot expand empty dataset")
        return initial_df
    
    print("\\n" + "="*60)
    print("EXPANDING DATASET")
    print("="*60)
    
    api_key = "xxx"
    
    # Additional elements to search
    additional_elements = ["Ti", "V", "Cr", "Mn", "Co", "Ni", "Cu", "Zn"]
    
    all_compounds = []
    
    with MPRester(api_key) as mpr:
        
        for element in additional_elements:
            print(f"Searching for {element} compounds...")
            
            try:
                materials_docs = mpr.materials.search(
                    elements=[element],
                    exclude_elements=["C", "H", "N", "O"],
                    num_elements=(2, 3),
                    crystal_system=["Cubic", "Hexagonal"],
                    fields=["material_id", "formula_pretty", "structure", "symmetry", 
                           "volume", "density", "nsites"]
                )
                
                print(f"  Found {len(materials_docs)} {element} compounds")
                
                # Limit per element
                if len(materials_docs) > 50:
                    materials_docs = materials_docs[:50]
                
                # Process quickly (simplified)
                for doc in materials_docs:
                    try:
                        structure = doc.structure
                        sga = SpacegroupAnalyzer(structure)
                        
                        compound_data = {
                            'material_id': doc.material_id,
                            'formula': doc.formula_pretty,
                            'nsites': doc.nsites,
                            'volume': doc.volume,
                            'density': doc.density,
                            'spacegroup_number': sga.get_space_group_number(),
                            'crystal_system': sga.get_crystal_system(),
                            'a': structure.lattice.a,
                            'b': structure.lattice.b,
                            'c': structure.lattice.c,
                            'volume_per_atom': structure.volume / len(structure),
                            'tm_element': element,
                            'coordination_number': 6  # Default
                        }
                        
                        all_compounds.append(compound_data)
                        
                    except Exception as e:
                        continue
                
            except Exception as e:
                print(f"  ⚠️  Error with {element}: {e}")
                continue
    
    # Combine with initial dataset
    if all_compounds:
        expanded_df = pd.concat([initial_df, pd.DataFrame(all_compounds)], ignore_index=True)
        expanded_df = expanded_df.drop_duplicates(subset=['material_id'])
        
        print(f"\\n✅ Expanded dataset: {len(initial_df)} → {len(expanded_df)} compounds")
        
        # Save expanded dataset
        filename = "mp_expanded_inorganic_dataset.csv"
        expanded_df.to_csv(filename, index=False)
        print(f"💾 Saved expanded dataset to {filename}")
        
        return expanded_df
    else:
        print("❌ No additional compounds found")
        return initial_df

def main():
    """Main execution"""
    
    print("Starting simplified Materials Project acquisition...")
    
    # Get initial dataset
    df = fetch_materials_data()
    
    if len(df) > 0:
        print("\\n✅ Initial dataset acquired successfully!")
        
        # Expand if successful
        df_expanded = expand_dataset(df)
        
        print("\\n" + "="*80)
        print("ACQUISITION COMPLETE")
        print("="*80)
        print(f"✅ Final dataset: {len(df_expanded)} compounds")
        print("✅ Ready for UBP geometric analysis")
        
        return df_expanded
    else:
        print("❌ Failed to acquire initial dataset")
        return None

if __name__ == "__main__":
    dataset = main()
