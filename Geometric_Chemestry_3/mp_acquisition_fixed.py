#!/usr/bin/env python3
"""
Materials Project Data Acquisition - Fixed Version
Using proper MP API structure for pure inorganic mapping
"""

import pandas as pd
import numpy as np
from mp_api.client import MPRester
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json
import time
import warnings
warnings.filterwarnings('ignore')

class MaterialsProjectAcquisitionFixed:
    """Fixed version using proper MP API structure"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Target elements: First-row transition metals + some main group
        self.target_elements = [
            "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",  # First-row TM
            "Al", "Si", "P", "S", "Cl",  # Key main group (no C/H/N/O)
            "Li", "Na", "K", "Mg", "Ca"  # Alkali/alkaline earth
        ]
        
        # Excluded elements (organic + some problematic ones)
        self.excluded_elements = ["C", "H", "N", "O", "F", "He", "Ne", "Ar", "Kr", "Xe", "Rn"]
        
        self.compounds_data = []
        
    def fetch_compounds(self, max_compounds: int = 1000):
        """Fetch compounds using the correct MP API structure"""
        
        print("="*80)
        print("MATERIALS PROJECT DATA ACQUISITION - FIXED")
        print("="*80)
        print(f"Target: {max_compounds} compounds")
        print(f"Elements: {', '.join(self.target_elements)}")
        print(f"Excluded: {', '.join(self.excluded_elements)}")
        print(f"Composition: Binary/Ternary only")
        print()
        
        with MPRester(self.api_key) as mpr:
            
            # Step 1: Get basic materials data
            print("Step 1: Fetching basic materials data...")
            try:
                materials_docs = mpr.materials.search(
                    elements=self.target_elements,
                    exclude_elements=self.excluded_elements,
                    num_elements=(2, 3),  # Binary and ternary
                    crystal_system=["cubic", "hexagonal"],
                    fields=["material_id", "formula_pretty", "structure", "symmetry", 
                           "volume", "density", "nsites", "elements", "composition"]
                )
                
                print(f"✅ Found {len(materials_docs)} materials")
                
                if len(materials_docs) == 0:
                    print("❌ No materials found!")
                    return pd.DataFrame()
                
                # Limit to target number
                if len(materials_docs) > max_compounds:
                    materials_docs = materials_docs[:max_compounds]
                    print(f"📊 Limited to {max_compounds} materials")
                
                # Extract material IDs for property queries
                material_ids = [doc.material_id for doc in materials_docs]
                
            except Exception as e:
                print(f"❌ Error fetching materials: {e}")
                return pd.DataFrame()
            
            # Step 2: Get thermodynamic data
            print("Step 2: Fetching thermodynamic data...")
            thermo_data = {}
            try:
                thermo_docs = mpr.materials.thermo.search(
                    material_ids=material_ids,
                    fields=["material_id", "formation_energy_per_atom", "energy_above_hull"]
                )
                
                for doc in thermo_docs:
                    thermo_data[doc.material_id] = {
                        'formation_energy_per_atom': doc.formation_energy_per_atom,
                        'energy_above_hull': doc.energy_above_hull
                    }
                
                print(f"✅ Got thermodynamic data for {len(thermo_data)} materials")
                
            except Exception as e:
                print(f"⚠️  Error fetching thermo data: {e}")
                thermo_data = {}
            
            # Step 3: Get electronic structure data
            print("Step 3: Fetching electronic structure data...")
            electronic_data = {}
            try:
                es_docs = mpr.materials.electronic_structure.search(
                    material_ids=material_ids,
                    fields=["material_id", "band_gap"]
                )
                
                for doc in es_docs:
                    electronic_data[doc.material_id] = {
                        'band_gap': doc.band_gap
                    }
                
                print(f"✅ Got electronic data for {len(electronic_data)} materials")
                
            except Exception as e:
                print(f"⚠️  Error fetching electronic data: {e}")
                electronic_data = {}
            
            # Step 4: Get magnetism data
            print("Step 4: Fetching magnetism data...")
            magnetism_data = {}
            try:
                mag_docs = mpr.materials.magnetism.search(
                    material_ids=material_ids,
                    fields=["material_id", "total_magnetization"]
                )
                
                for doc in mag_docs:
                    magnetism_data[doc.material_id] = {
                        'total_magnetization': doc.total_magnetization
                    }
                
                print(f"✅ Got magnetism data for {len(magnetism_data)} materials")
                
            except Exception as e:
                print(f"⚠️  Error fetching magnetism data: {e}")
                magnetism_data = {}
            
            # Step 5: Process and combine all data
            print("Step 5: Processing and combining data...")
            processed_count = 0
            
            for i, doc in enumerate(materials_docs):
                try:
                    compound_data = self._process_compound(
                        doc, 
                        thermo_data.get(doc.material_id, {}),
                        electronic_data.get(doc.material_id, {}),
                        magnetism_data.get(doc.material_id, {})
                    )
                    
                    if compound_data:
                        self.compounds_data.append(compound_data)
                        processed_count += 1
                    
                    # Progress update
                    if (i + 1) % 50 == 0:
                        print(f"  Processed {i + 1}/{len(materials_docs)} materials...")
                        
                except Exception as e:
                    print(f"  ⚠️  Error processing {doc.material_id}: {str(e)}")
                    continue
            
            print(f"✅ Successfully processed {processed_count} materials")
            
            # Convert to DataFrame
            df = pd.DataFrame(self.compounds_data)
            
            # Basic statistics
            self._print_dataset_summary(df)
            
            return df
    
    def _process_compound(self, materials_doc, thermo_doc, electronic_doc, magnetism_doc):
        """Process a single compound with all its property data"""
        
        try:
            structure = materials_doc.structure
            
            # Basic properties from materials doc
            compound_data = {
                'material_id': materials_doc.material_id,
                'formula': materials_doc.formula_pretty,
                'nsites': materials_doc.nsites,
                'volume': materials_doc.volume,
                'density': materials_doc.density,
                'elements': [str(el) for el in materials_doc.elements],
                'num_elements': len(materials_doc.elements)
            }
            
            # Add thermodynamic properties
            compound_data['formation_energy_per_atom'] = thermo_doc.get('formation_energy_per_atom', np.nan)
            compound_data['energy_above_hull'] = thermo_doc.get('energy_above_hull', np.nan)
            
            # Add electronic properties
            compound_data['band_gap'] = electronic_doc.get('band_gap', np.nan)
            
            # Add magnetic properties
            compound_data['total_magnetization'] = magnetism_doc.get('total_magnetization', np.nan)
            
            # Structural analysis
            sga = SpacegroupAnalyzer(structure)
            
            # Space group and crystal system
            compound_data['spacegroup_symbol'] = sga.get_space_group_symbol()
            compound_data['spacegroup_number'] = sga.get_space_group_number()
            compound_data['crystal_system'] = sga.get_crystal_system()
            compound_data['point_group'] = sga.get_point_group_symbol()
            
            # Lattice parameters
            lattice = structure.lattice
            compound_data['a'] = lattice.a
            compound_data['b'] = lattice.b
            compound_data['c'] = lattice.c
            compound_data['alpha'] = lattice.alpha
            compound_data['beta'] = lattice.beta
            compound_data['gamma'] = lattice.gamma
            compound_data['volume_per_atom'] = structure.volume / len(structure)
            
            # Find transition metal sites and analyze coordination
            tm_elements = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
            
            tm_sites = []
            for i, site in enumerate(structure.sites):
                if str(site.specie) in tm_elements:
                    tm_sites.append(i)
            
            if tm_sites:
                # Analyze coordination environment of first TM site
                try:
                    cnn = CrystalNN()
                    cn_info = cnn.get_cn_dict(structure, tm_sites[0])
                    
                    # Get coordination number and geometry
                    total_cn = sum(cn_info.values())
                    compound_data['coordination_number'] = total_cn
                    compound_data['tm_element'] = str(structure[tm_sites[0]].specie)
                    
                except Exception as e:
                    # Fallback coordination analysis
                    compound_data['coordination_number'] = 6  # Default assumption
                    compound_data['tm_element'] = str(structure[tm_sites[0]].specie)
            else:
                compound_data['coordination_number'] = None
                compound_data['tm_element'] = None
            
            # Geometric features for UBP analysis
            compound_data['packing_fraction'] = self._calculate_packing_fraction(structure)
            compound_data['symmetry_index'] = self._calculate_symmetry_index(sga)
            
            return compound_data
            
        except Exception as e:
            print(f"    Error processing compound: {str(e)}")
            return None
    
    def _calculate_packing_fraction(self, structure):
        """Calculate approximate packing fraction"""
        try:
            # Simple approximation using ionic radii
            total_volume = 0
            for site in structure.sites:
                # Use a default radius if not available
                radius = getattr(site.specie, 'ionic_radius', 1.0) or 1.0
                total_volume += (4/3) * np.pi * (radius ** 3)
            
            packing_fraction = total_volume / structure.volume
            return min(packing_fraction, 1.0)  # Cap at 1.0
            
        except:
            return 0.5  # Default value
    
    def _calculate_symmetry_index(self, sga):
        """Calculate a symmetry index based on space group"""
        try:
            # Higher space group numbers generally mean higher symmetry
            spg_num = sga.get_space_group_number()
            
            # Normalize to 0-1 scale (space groups go from 1-230)
            symmetry_index = spg_num / 230.0
            
            return symmetry_index
            
        except:
            return 0.5  # Default value
    
    def _print_dataset_summary(self, df):
        """Print comprehensive dataset summary"""
        
        print("\\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total compounds: {len(df)}")
        print(f"Unique formulas: {df['formula'].nunique()}")
        
        if 'crystal_system' in df.columns:
            print(f"Crystal systems: {df['crystal_system'].value_counts().to_dict()}")
        
        if 'spacegroup_number' in df.columns:
            print(f"Space groups: {df['spacegroup_number'].nunique()} unique")
        
        print()
        print("Property coverage:")
        
        properties = ['formation_energy_per_atom', 'band_gap', 'total_magnetization']
        for prop in properties:
            if prop in df.columns:
                non_null = df[prop].notna().sum()
                print(f"  {prop}: {non_null}/{len(df)} ({100*non_null/len(df):.1f}%)")
        
        print()
        print("Property ranges (non-null values):")
        
        for prop in properties:
            if prop in df.columns and df[prop].notna().sum() > 0:
                values = df[prop].dropna()
                print(f"  {prop}: {values.min():.3f} to {values.max():.3f}")
    
    def save_dataset(self, df, filename="mp_pure_inorganic_fixed.csv"):
        """Save the processed dataset"""
        
        print(f"\\n💾 Saving dataset to {filename}...")
        
        # Create a clean version for CSV (remove complex objects)
        df_clean = df.copy()
        
        # Convert lists to strings for CSV compatibility
        if 'elements' in df_clean.columns:
            df_clean['elements'] = df_clean['elements'].apply(lambda x: ','.join(x) if isinstance(x, list) else str(x))
        
        # Save to CSV
        df_clean.to_csv(filename, index=False)
        
        # Also save raw data as JSON for full fidelity
        json_filename = filename.replace('.csv', '_raw.json')
        with open(json_filename, 'w') as f:
            json.dump(self.compounds_data, f, indent=2, default=str)
        
        print(f"✅ Saved {len(df)} compounds to {filename}")
        print(f"✅ Saved raw data to {json_filename}")
        
        return df_clean

def main():
    """Main execution function"""
    
    # Initialize with API key
    api_key = "xxx"
    
    # Create acquisition instance
    mp_acq = MaterialsProjectAcquisitionFixed(api_key)
    
    # Fetch compounds
    print("Starting Materials Project data acquisition (Fixed Version)...")
    print("Phase 1: Pure Inorganic Mapping - Binary/Ternary Compounds")
    print()
    
    # Start with moderate sample
    df = mp_acq.fetch_compounds(max_compounds=500)
    
    if len(df) > 0:
        # Save the dataset
        df_clean = mp_acq.save_dataset(df, "mp_pure_inorganic_phase1_fixed.csv")
        
        print("\\n" + "="*80)
        print("PHASE 1 COMPLETE - FIXED VERSION")
        print("="*80)
        print(f"✅ Successfully acquired {len(df)} pure inorganic compounds")
        print("✅ Ready for Phase 2: UBP geometric feature extraction")
        print("✅ Dataset saved and ready for analysis")
        
        return df_clean
    else:
        print("❌ No data acquired. Check API connection.")
        return None

if __name__ == "__main__":
    dataset = main()
