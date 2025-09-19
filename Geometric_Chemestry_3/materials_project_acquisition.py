#!/usr/bin/env python3
"""
Materials Project Data Acquisition for Pure Inorganic Mapping
Phase 1: Binary/Ternary + Cubic/Hexagonal + Pure Metals (No C/H/N/O)

Target: 1,000 compounds initially, scaling to 5,000-10,000
Focus: Formation energy, band gap, magnetic moment + geometric features
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

class MaterialsProjectAcquisition:
    """Acquire and process Materials Project data for UBP geometric mapping"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.mpr = MPRester(api_key)
        
        # Target elements: First-row transition metals + some main group
        self.target_elements = [
            "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",  # First-row TM
            "Sc", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",  # Extended TM
            "Al", "Si", "P", "S", "Cl", "Br", "I",  # Key main group (no C/H/N/O)
            "Li", "Na", "K", "Mg", "Ca", "Sr", "Ba"  # Alkali/alkaline earth
        ]
        
        # Excluded elements (organic + some problematic ones)
        self.excluded_elements = ["C", "H", "N", "O", "F", "He", "Ne", "Ar", "Kr", "Xe", "Rn"]
        
        # Target crystal systems (high symmetry first)
        self.target_crystal_systems = ["cubic", "hexagonal"]
        
        self.compounds_data = []
        
    def fetch_compounds(self, max_compounds: int = 1000, test_mode: bool = True):
        """Fetch compounds from Materials Project with specified filters"""
        
        print("="*80)
        print("MATERIALS PROJECT DATA ACQUISITION")
        print("="*80)
        print(f"Target: {max_compounds} compounds")
        print(f"Elements: {', '.join(self.target_elements[:9])}... (and others)")
        print(f"Excluded: {', '.join(self.excluded_elements)}")
        print(f"Crystal systems: {', '.join(self.target_crystal_systems)}")
        print(f"Composition: Binary/Ternary only")
        print()
        
        if test_mode:
            print("🧪 TEST MODE: Fetching small sample first...")
            max_compounds = min(max_compounds, 100)
        
        try:
            # Search with filters
            print("Searching Materials Project database...")
            
            docs = self.mpr.materials.search(
                elements=self.target_elements,
                exclude_elements=self.excluded_elements,
                num_elements=(2, 3),  # Binary and ternary only
                crystal_system=self.target_crystal_systems,
                fields=[
                    "material_id", 
                    "formula_pretty", 
                    "formation_energy_per_atom",
                    "band_gap", 
                    "total_magnetization",
                    "structure",
                    "energy_above_hull",
                    "theoretical",
                    "nsites",
                    "volume",
                    "density"
                ],
                chunk_size=1000,
                num_chunks=max_compounds // 1000 + 1
            )
            
            print(f"✅ Found {len(docs)} compounds from Materials Project")
            
            if len(docs) == 0:
                print("❌ No compounds found with current filters!")
                return pd.DataFrame()
            
            # Limit to target number
            if len(docs) > max_compounds:
                docs = docs[:max_compounds]
                print(f"📊 Limited to {max_compounds} compounds for analysis")
            
            # Process each compound
            print("Processing compound structures and properties...")
            processed_count = 0
            
            for i, doc in enumerate(docs):
                try:
                    compound_data = self._process_compound(doc)
                    if compound_data:
                        self.compounds_data.append(compound_data)
                        processed_count += 1
                    
                    # Progress update
                    if (i + 1) % 100 == 0:
                        print(f"  Processed {i + 1}/{len(docs)} compounds...")
                        
                except Exception as e:
                    print(f"  ⚠️  Error processing {doc.material_id}: {str(e)}")
                    continue
            
            print(f"✅ Successfully processed {processed_count} compounds")
            
            # Convert to DataFrame
            df = pd.DataFrame(self.compounds_data)
            
            # Basic statistics
            print("\\n" + "="*50)
            print("DATASET SUMMARY")
            print("="*50)
            print(f"Total compounds: {len(df)}")
            print(f"Unique formulas: {df['formula'].nunique()}")
            print(f"Crystal systems: {df['crystal_system'].value_counts().to_dict()}")
            print(f"Space groups: {df['spacegroup_number'].nunique()} unique")
            print()
            print("Property ranges:")
            print(f"  Formation energy: {df['formation_energy_per_atom'].min():.3f} to {df['formation_energy_per_atom'].max():.3f} eV/atom")
            print(f"  Band gap: {df['band_gap'].min():.3f} to {df['band_gap'].max():.3f} eV")
            print(f"  Magnetic moment: {df['total_magnetization'].min():.3f} to {df['total_magnetization'].max():.3f} μB")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def _process_compound(self, doc):
        """Process a single compound document from Materials Project"""
        
        try:
            structure = doc.structure
            
            # Basic properties
            compound_data = {
                'material_id': doc.material_id,
                'formula': doc.formula_pretty,
                'formation_energy_per_atom': doc.formation_energy_per_atom,
                'band_gap': doc.band_gap,
                'total_magnetization': doc.total_magnetization,
                'energy_above_hull': doc.energy_above_hull,
                'theoretical': doc.theoretical,
                'nsites': doc.nsites,
                'volume': doc.volume,
                'density': doc.density
            }
            
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
            
            # Composition analysis
            composition = structure.composition
            compound_data['num_elements'] = len(composition.elements)
            compound_data['elements'] = [str(el) for el in composition.elements]
            
            # Find transition metal sites and analyze coordination
            tm_elements = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
                          "Sc", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"]
            
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
                    
                    # Coordination environment details
                    compound_data['coordination_details'] = cn_info
                    
                except Exception as e:
                    # Fallback coordination analysis
                    compound_data['coordination_number'] = 6  # Default assumption
                    compound_data['tm_element'] = str(structure[tm_sites[0]].specie)
                    compound_data['coordination_details'] = {}
            else:
                compound_data['coordination_number'] = None
                compound_data['tm_element'] = None
                compound_data['coordination_details'] = {}
            
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
    
    def save_dataset(self, df, filename="materials_project_dataset.csv"):
        """Save the processed dataset"""
        
        print(f"\\n💾 Saving dataset to {filename}...")
        
        # Create a clean version for CSV (remove complex objects)
        df_clean = df.copy()
        
        # Convert lists to strings for CSV compatibility
        if 'elements' in df_clean.columns:
            df_clean['elements'] = df_clean['elements'].apply(lambda x: ','.join(x) if isinstance(x, list) else str(x))
        
        if 'coordination_details' in df_clean.columns:
            df_clean['coordination_details'] = df_clean['coordination_details'].apply(lambda x: str(x) if x else '')
        
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
    api_key = "QgtXNTeNADmxywaU4nAWo7oI5aT8J4g4"
    
    # Create acquisition instance
    mp_acq = MaterialsProjectAcquisition(api_key)
    
    # Fetch compounds (start with test mode)
    print("Starting Materials Project data acquisition...")
    print("Phase 1: Pure Inorganic Mapping - Binary/Ternary Compounds")
    print()
    
    # Test with small sample first
    df = mp_acq.fetch_compounds(max_compounds=1000, test_mode=True)
    
    if len(df) > 0:
        # Save the dataset
        df_clean = mp_acq.save_dataset(df, "mp_pure_inorganic_phase1.csv")
        
        print("\\n" + "="*80)
        print("PHASE 1 COMPLETE")
        print("="*80)
        print(f"✅ Successfully acquired {len(df)} pure inorganic compounds")
        print("✅ Ready for Phase 2: Geometric feature extraction")
        print("✅ Dataset saved and ready for UBP analysis")
        
        return df_clean
    else:
        print("❌ No data acquired. Check filters and API key.")
        return None

if __name__ == "__main__":
    dataset = main()
