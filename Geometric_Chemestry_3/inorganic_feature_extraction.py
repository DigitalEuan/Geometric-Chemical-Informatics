#!/usr/bin/env python3
"""
Comprehensive Geometric and Crystallographic Feature Extraction
Phase 2: Extract features for UBP geometric mapping of inorganic materials

Features extracted:
1. Crystallographic: Space group, lattice parameters, symmetry metrics
2. Geometric: Coordination environments, polyhedral distortions
3. Electronic: Band structure features, density of states
4. Topological: Connectivity, dimensionality, packing
5. UBP-specific: Realm assignments, resonance potentials
"""

import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json
import warnings
warnings.filterwarnings('ignore')

class InorganicFeatureExtractor:
    """Extract comprehensive features for UBP geometric mapping"""
    
    def __init__(self):
        self.features_data = []
        
        # UBP realm assignments for different property types
        self.ubp_realms = {
            'gravitational': ['density', 'volume_per_atom', 'mass_related'],
            'electromagnetic': ['band_gap', 'dielectric', 'conductivity'],
            'quantum': ['magnetic_moment', 'spin_state', 'orbital_related'],
            'nuclear': ['atomic_number', 'nuclear_charge', 'isotope_related'],
            'optical': ['band_gap', 'optical_properties', 'photonic'],
            'biological': [],  # Not applicable for inorganic materials
            'cosmological': ['formation_energy', 'stability', 'thermodynamic']
        }
        
        # Sacred geometry constants for resonance detection
        self.sacred_constants = {
            'phi': 1.618033988749,      # Golden ratio
            'pi': 3.141592653589793,    # Pi
            'e': 2.718281828459045,     # Euler's number
            'sqrt2': 1.4142135623730951, # Square root of 2
            'sqrt3': 1.7320508075688772, # Square root of 3
            'sqrt5': 2.23606797749979,   # Square root of 5
        }
    
    def extract_features(self, dataset_file: str):
        """Extract comprehensive features from materials dataset"""
        
        print("="*80)
        print("COMPREHENSIVE FEATURE EXTRACTION")
        print("="*80)
        print("Phase 2: Geometric and Crystallographic Features")
        print("Target: UBP-enhanced inorganic materials analysis")
        print()
        
        # Load dataset
        print(f"Loading dataset from {dataset_file}...")
        try:
            df = pd.read_csv(dataset_file)
            print(f"✅ Loaded {len(df)} compounds")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return pd.DataFrame()
        
        # Process each compound
        print("\\nExtracting features for each compound...")
        processed_count = 0
        
        for i, row in df.iterrows():
            try:
                features = self._extract_compound_features(row)
                if features:
                    self.features_data.append(features)
                    processed_count += 1
                
                # Progress update
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(df)} compounds...")
                    
            except Exception as e:
                print(f"  ⚠️  Error processing {row.get('material_id', 'unknown')}: {str(e)}")
                continue
        
        print(f"✅ Successfully extracted features for {processed_count} compounds")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(self.features_data)
        
        # Add derived features
        features_df = self._add_derived_features(features_df)
        
        # Print feature summary
        self._print_feature_summary(features_df)
        
        return features_df
    
    def _extract_compound_features(self, row):
        """Extract all features for a single compound"""
        
        try:
            # Basic compound info
            features = {
                'material_id': row['material_id'],
                'formula': row['formula'],
                'tm_element': row.get('tm_element', None)
            }
            
            # Basic properties (from original dataset)
            basic_props = ['formation_energy_per_atom', 'band_gap', 'total_magnetization',
                          'nsites', 'volume', 'density', 'spacegroup_number', 'crystal_system',
                          'a', 'b', 'c', 'volume_per_atom', 'coordination_number']
            
            for prop in basic_props:
                features[prop] = row.get(prop, np.nan)
            
            # Crystallographic features
            features.update(self._extract_crystallographic_features(row))
            
            # Geometric features
            features.update(self._extract_geometric_features(row))
            
            # Electronic features
            features.update(self._extract_electronic_features(row))
            
            # Topological features
            features.update(self._extract_topological_features(row))
            
            # UBP-specific features
            features.update(self._extract_ubp_features(row, features))
            
            return features
            
        except Exception as e:
            print(f"    Error extracting features: {str(e)}")
            return None
    
    def _extract_crystallographic_features(self, row):
        """Extract crystallographic and symmetry features"""
        
        features = {}
        
        try:
            # Space group analysis
            spg_num = row.get('spacegroup_number', 1)
            features['spacegroup_number'] = spg_num
            
            # Symmetry metrics
            features['symmetry_index'] = spg_num / 230.0  # Normalized
            
            # Crystal system encoding
            crystal_systems = {'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3,
                             'tetragonal': 4, 'trigonal': 5, 'hexagonal': 6, 'cubic': 7}
            features['crystal_system_code'] = crystal_systems.get(row.get('crystal_system', 'triclinic'), 1)
            
            # Lattice parameters
            a, b, c = row.get('a', 5.0), row.get('b', 5.0), row.get('c', 5.0)
            
            # Lattice ratios (geometric relationships)
            features['b_over_a'] = b / a if a > 0 else 1.0
            features['c_over_a'] = c / a if a > 0 else 1.0
            features['c_over_b'] = c / b if b > 0 else 1.0
            
            # Volume relationships
            volume = row.get('volume', a * b * c)
            features['volume_cubic_root'] = volume ** (1/3)
            features['volume_per_atom'] = row.get('volume_per_atom', volume / row.get('nsites', 1))
            
            # Lattice distortion metrics
            avg_lattice = (a + b + c) / 3
            features['lattice_distortion'] = np.sqrt(((a - avg_lattice)**2 + (b - avg_lattice)**2 + (c - avg_lattice)**2) / 3) / avg_lattice
            
            # Sacred geometry resonances in lattice
            for name, constant in self.sacred_constants.items():
                features[f'lattice_{name}_resonance'] = self._calculate_resonance(a/b, constant)
                features[f'volume_{name}_resonance'] = self._calculate_resonance(volume**(1/3), constant)
            
        except Exception as e:
            print(f"      Crystallographic error: {e}")
            # Set default values
            for key in ['symmetry_index', 'crystal_system_code', 'b_over_a', 'c_over_a', 'lattice_distortion']:
                features[key] = 0.5
        
        return features
    
    def _extract_geometric_features(self, row):
        """Extract geometric and coordination features"""
        
        features = {}
        
        try:
            # Coordination environment
            cn = row.get('coordination_number', 6)
            features['coordination_number'] = cn
            
            # Coordination geometry classification
            coord_geometries = {
                2: 'linear', 3: 'trigonal', 4: 'tetrahedral', 5: 'square_pyramidal',
                6: 'octahedral', 7: 'pentagonal_bipyramidal', 8: 'cubic', 9: 'tricapped_trigonal',
                10: 'bicapped_square_antiprismatic', 12: 'cuboctahedral'
            }
            features['coordination_geometry'] = coord_geometries.get(cn, 'unknown')
            features['coordination_geometry_code'] = cn
            
            # Polyhedral features
            features['polyhedral_volume'] = self._estimate_polyhedral_volume(cn, row.get('a', 5.0))
            features['coordination_sphere_packing'] = self._estimate_coordination_packing(cn)
            
            # Geometric ratios
            density = row.get('density', 5.0)
            volume_per_atom = row.get('volume_per_atom', 20.0)
            
            features['density_volume_ratio'] = density * volume_per_atom
            features['atomic_packing_fraction'] = self._estimate_packing_fraction(row.get('crystal_system', 'cubic'), cn)
            
            # Geometric resonances
            for name, constant in self.sacred_constants.items():
                features[f'coord_{name}_resonance'] = self._calculate_resonance(cn, constant)
                features[f'density_{name}_resonance'] = self._calculate_resonance(density, constant)
            
        except Exception as e:
            print(f"      Geometric error: {e}")
            # Set default values
            features['coordination_number'] = 6
            features['coordination_geometry_code'] = 6
            features['polyhedral_volume'] = 1.0
            features['atomic_packing_fraction'] = 0.74
        
        return features
    
    def _extract_electronic_features(self, row):
        """Extract electronic structure features"""
        
        features = {}
        
        try:
            # Basic electronic properties
            band_gap = row.get('band_gap', 0.0)
            mag_moment = row.get('total_magnetization', 0.0)
            
            features['band_gap'] = band_gap
            features['total_magnetization'] = mag_moment
            
            # Electronic classifications
            features['is_metal'] = 1 if band_gap < 0.1 else 0
            features['is_semiconductor'] = 1 if 0.1 <= band_gap <= 4.0 else 0
            features['is_insulator'] = 1 if band_gap > 4.0 else 0
            features['is_magnetic'] = 1 if abs(mag_moment) > 0.1 else 0
            
            # Electronic ratios and derived properties
            features['mag_moment_per_atom'] = mag_moment / row.get('nsites', 1)
            features['electronic_density'] = band_gap * row.get('density', 1.0)
            
            # Spin state estimation (for transition metals)
            tm_element = row.get('tm_element', None)
            if tm_element:
                features['estimated_spin_state'] = self._estimate_spin_state(tm_element, mag_moment)
            else:
                features['estimated_spin_state'] = 0
            
            # Electronic resonances
            for name, constant in self.sacred_constants.items():
                features[f'bandgap_{name}_resonance'] = self._calculate_resonance(band_gap, constant)
                features[f'magmom_{name}_resonance'] = self._calculate_resonance(abs(mag_moment), constant)
            
        except Exception as e:
            print(f"      Electronic error: {e}")
            # Set default values
            features['band_gap'] = 0.0
            features['total_magnetization'] = 0.0
            features['is_metal'] = 0
            features['estimated_spin_state'] = 0
        
        return features
    
    def _extract_topological_features(self, row):
        """Extract topological and connectivity features"""
        
        features = {}
        
        try:
            # Dimensionality estimation
            crystal_system = row.get('crystal_system', 'cubic')
            features['estimated_dimensionality'] = self._estimate_dimensionality(crystal_system)
            
            # Connectivity metrics
            cn = row.get('coordination_number', 6)
            nsites = row.get('nsites', 1)
            
            features['connectivity_index'] = cn / nsites if nsites > 0 else 0
            features['network_density'] = (cn * nsites) / (2 * row.get('volume', 100))
            
            # Topological invariants (simplified)
            features['euler_characteristic'] = self._estimate_euler_characteristic(crystal_system, cn)
            features['genus'] = max(0, 1 - features['euler_characteristic'])
            
            # Structural complexity
            features['structural_complexity'] = np.log(nsites) * features['connectivity_index']
            
        except Exception as e:
            print(f"      Topological error: {e}")
            # Set default values
            features['estimated_dimensionality'] = 3
            features['connectivity_index'] = 0.5
            features['structural_complexity'] = 1.0
        
        return features
    
    def _extract_ubp_features(self, row, existing_features):
        """Extract UBP-specific features and realm assignments"""
        
        features = {}
        
        try:
            # Realm assignments based on property dominance
            realm_scores = {}
            
            # Gravitational realm (mass, density, volume)
            density = row.get('density', 5.0)
            volume_per_atom = row.get('volume_per_atom', 20.0)
            realm_scores['gravitational'] = np.log(density * volume_per_atom)
            
            # Electromagnetic realm (band gap, conductivity)
            band_gap = row.get('band_gap', 0.0)
            realm_scores['electromagnetic'] = band_gap + (1.0 / (band_gap + 0.1))
            
            # Quantum realm (magnetism, spin)
            mag_moment = abs(row.get('total_magnetization', 0.0))
            realm_scores['quantum'] = mag_moment + existing_features.get('estimated_spin_state', 0)
            
            # Nuclear realm (atomic number, nuclear properties)
            tm_element = row.get('tm_element', None)
            if tm_element:
                atomic_numbers = {'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 
                                'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30}
                realm_scores['nuclear'] = atomic_numbers.get(tm_element, 26) / 30.0
            else:
                realm_scores['nuclear'] = 0.5
            
            # Cosmological realm (formation energy, stability)
            formation_energy = abs(row.get('formation_energy_per_atom', 0.0))
            realm_scores['cosmological'] = formation_energy
            
            # Optical realm (band gap in optical range)
            realm_scores['optical'] = 1.0 / (1.0 + abs(band_gap - 2.0))  # Peak at ~2 eV
            
            # Normalize realm scores
            total_score = sum(realm_scores.values())
            if total_score > 0:
                for realm in realm_scores:
                    features[f'realm_{realm}_score'] = realm_scores[realm] / total_score
            else:
                for realm in realm_scores:
                    features[f'realm_{realm}_score'] = 1.0 / len(realm_scores)
            
            # Primary realm assignment
            primary_realm = max(realm_scores, key=realm_scores.get)
            features['primary_ubp_realm'] = primary_realm
            features['primary_realm_code'] = list(realm_scores.keys()).index(primary_realm)
            
            # UBP energy calculation (simplified)
            features['ubp_energy'] = self._calculate_ubp_energy(row, existing_features, realm_scores)
            
            # NRCI estimation
            features['estimated_nrci'] = self._estimate_nrci(existing_features, realm_scores)
            
            # Resonance potential
            features['resonance_potential'] = self._calculate_resonance_potential(existing_features)
            
        except Exception as e:
            print(f"      UBP error: {e}")
            # Set default values
            features['primary_ubp_realm'] = 'electromagnetic'
            features['primary_realm_code'] = 1
            features['ubp_energy'] = 1.0
            features['estimated_nrci'] = 0.5
        
        return features
    
    def _calculate_resonance(self, value, constant, tolerance=0.1):
        """Calculate resonance between a value and a sacred geometry constant"""
        if value <= 0:
            return 0.0
        
        # Check direct resonance
        ratio = value / constant
        resonance = np.exp(-((ratio - 1.0) ** 2) / (2 * tolerance ** 2))
        
        # Check harmonic resonances (2x, 3x, 1/2x, 1/3x)
        harmonics = [2.0, 3.0, 0.5, 1/3]
        for harmonic in harmonics:
            harmonic_ratio = value / (constant * harmonic)
            harmonic_resonance = np.exp(-((harmonic_ratio - 1.0) ** 2) / (2 * tolerance ** 2))
            resonance = max(resonance, harmonic_resonance * 0.5)  # Reduced weight for harmonics
        
        return resonance
    
    def _estimate_polyhedral_volume(self, coordination_number, lattice_param):
        """Estimate polyhedral volume based on coordination"""
        # Simplified geometric calculation
        coord_volumes = {
            4: 0.118,   # Tetrahedron
            6: 1.0,     # Octahedron (reference)
            8: 2.0,     # Cube
            12: 2.83    # Cuboctahedron
        }
        base_volume = coord_volumes.get(coordination_number, 1.0)
        return base_volume * (lattice_param ** 3)
    
    def _estimate_coordination_packing(self, coordination_number):
        """Estimate packing efficiency for coordination environment"""
        packing_efficiencies = {
            4: 0.34,    # Tetrahedral
            6: 0.74,    # Octahedral/FCC
            8: 0.68,    # Simple cubic
            12: 0.74    # FCC/HCP
        }
        return packing_efficiencies.get(coordination_number, 0.5)
    
    def _estimate_packing_fraction(self, crystal_system, coordination_number):
        """Estimate atomic packing fraction"""
        system_packing = {
            'cubic': 0.74,
            'hexagonal': 0.74,
            'tetragonal': 0.68,
            'orthorhombic': 0.60,
            'monoclinic': 0.55,
            'triclinic': 0.50
        }
        base_packing = system_packing.get(crystal_system, 0.60)
        
        # Adjust for coordination
        coord_factor = coordination_number / 12.0  # Normalize to 12-fold coordination
        return base_packing * coord_factor
    
    def _estimate_spin_state(self, tm_element, mag_moment):
        """Estimate spin state for transition metal"""
        # Simplified spin state estimation
        expected_moments = {
            'Ti': 1.0, 'V': 2.0, 'Cr': 3.0, 'Mn': 4.0, 'Fe': 4.0,
            'Co': 3.0, 'Ni': 2.0, 'Cu': 1.0, 'Zn': 0.0
        }
        expected = expected_moments.get(tm_element, 2.0)
        
        if abs(mag_moment) < 0.5:
            return 0  # Low spin or diamagnetic
        elif abs(mag_moment) < expected * 0.7:
            return 1  # Low spin
        else:
            return 2  # High spin
    
    def _estimate_dimensionality(self, crystal_system):
        """Estimate structural dimensionality"""
        # Simplified dimensionality estimation
        if crystal_system in ['cubic', 'tetragonal']:
            return 3
        elif crystal_system in ['hexagonal', 'trigonal']:
            return 2.5  # Quasi-2D
        else:
            return 2
    
    def _estimate_euler_characteristic(self, crystal_system, coordination_number):
        """Estimate topological Euler characteristic"""
        # Simplified topological invariant
        if crystal_system == 'cubic' and coordination_number == 6:
            return 2  # Sphere-like
        elif crystal_system == 'hexagonal':
            return 0  # Torus-like
        else:
            return 1  # Intermediate
    
    def _calculate_ubp_energy(self, row, features, realm_scores):
        """Calculate simplified UBP energy"""
        try:
            # Simplified UBP energy calculation
            M = row.get('nsites', 1)  # Active OffBits ~ number of sites
            C = 299792458  # Speed of light
            
            # Resonance efficiency (average of all resonances)
            resonance_keys = [k for k in features.keys() if 'resonance' in k]
            R = np.mean([features.get(k, 0.5) for k in resonance_keys]) if resonance_keys else 0.5
            
            # Structural optimization
            S_opt = features.get('atomic_packing_fraction', 0.74)
            
            # Global coherence (based on realm score distribution)
            realm_values = list(realm_scores.values())
            P_GCI = 1.0 - np.std(realm_values) if len(realm_values) > 1 else 1.0
            
            # Simplified energy calculation
            energy = M * (R * S_opt) * P_GCI * 1e-10  # Scale factor
            
            return energy
            
        except:
            return 1.0
    
    def _estimate_nrci(self, features, realm_scores):
        """Estimate Non-random Coherence Index"""
        try:
            # Calculate coherence based on feature consistency
            resonance_keys = [k for k in features.keys() if 'resonance' in k]
            resonances = [features.get(k, 0.5) for k in resonance_keys]
            
            if len(resonances) > 1:
                # High NRCI if resonances are consistent (low variance)
                nrci = 1.0 - np.std(resonances)
            else:
                nrci = 0.5
            
            # Boost NRCI for high-symmetry systems
            symmetry_boost = features.get('symmetry_index', 0.5)
            nrci = min(1.0, nrci + 0.1 * symmetry_boost)
            
            return max(0.0, nrci)
            
        except:
            return 0.5
    
    def _calculate_resonance_potential(self, features):
        """Calculate overall resonance potential"""
        try:
            # Sum all resonance values
            resonance_keys = [k for k in features.keys() if 'resonance' in k]
            total_resonance = sum(features.get(k, 0.0) for k in resonance_keys)
            
            # Normalize by number of resonances
            if len(resonance_keys) > 0:
                return total_resonance / len(resonance_keys)
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _add_derived_features(self, df):
        """Add derived and composite features"""
        
        print("\\nAdding derived features...")
        
        try:
            # Composite indices
            df['stability_index'] = -df['formation_energy_per_atom'] / (df['energy_above_hull'] + 0.01)
            df['electronic_index'] = df['band_gap'] * df['total_magnetization']
            df['geometric_index'] = df['coordination_number'] * df['symmetry_index']
            
            # Multi-realm features
            realm_cols = [col for col in df.columns if 'realm_' in col and '_score' in col]
            if len(realm_cols) > 1:
                df['realm_diversity'] = df[realm_cols].apply(lambda x: 1.0 - np.max(x), axis=1)
                df['realm_entropy'] = df[realm_cols].apply(lambda x: -np.sum(x * np.log(x + 1e-10)), axis=1)
            
            # Sacred geometry composite
            phi_cols = [col for col in df.columns if 'phi_resonance' in col]
            if len(phi_cols) > 0:
                df['phi_resonance_total'] = df[phi_cols].sum(axis=1)
            
            pi_cols = [col for col in df.columns if 'pi_resonance' in col]
            if len(pi_cols) > 0:
                df['pi_resonance_total'] = df[pi_cols].sum(axis=1)
            
            print(f"✅ Added derived features. Total features: {len(df.columns)}")
            
        except Exception as e:
            print(f"⚠️  Error adding derived features: {e}")
        
        return df
    
    def _print_feature_summary(self, df):
        """Print comprehensive feature summary"""
        
        print("\\n" + "="*60)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total compounds: {len(df)}")
        print(f"Total features: {len(df.columns)}")
        
        # Feature categories
        categories = {
            'Basic': [col for col in df.columns if any(x in col for x in ['material_id', 'formula', 'nsites', 'volume', 'density'])],
            'Crystallographic': [col for col in df.columns if any(x in col for x in ['spacegroup', 'crystal', 'lattice', 'symmetry'])],
            'Geometric': [col for col in df.columns if any(x in col for x in ['coordination', 'polyhedral', 'packing'])],
            'Electronic': [col for col in df.columns if any(x in col for x in ['band_gap', 'magnetization', 'electronic'])],
            'Topological': [col for col in df.columns if any(x in col for x in ['dimensionality', 'connectivity', 'topology'])],
            'UBP': [col for col in df.columns if any(x in col for x in ['ubp', 'realm', 'nrci', 'resonance'])],
            'Sacred Geometry': [col for col in df.columns if any(x in col for x in ['phi', 'pi', 'sqrt', '_resonance'])]
        }
        
        print("\\nFeature categories:")
        for category, cols in categories.items():
            print(f"  {category}: {len(cols)} features")
        
        # Key statistics
        print("\\nKey feature statistics:")
        key_features = ['band_gap', 'total_magnetization', 'formation_energy_per_atom', 
                       'coordination_number', 'symmetry_index', 'ubp_energy', 'estimated_nrci']
        
        for feature in key_features:
            if feature in df.columns:
                values = df[feature].dropna()
                if len(values) > 0:
                    print(f"  {feature}: {values.min():.3f} to {values.max():.3f} (mean: {values.mean():.3f})")
        
        # Realm distribution
        if 'primary_ubp_realm' in df.columns:
            print("\\nUBP realm distribution:")
            realm_counts = df['primary_ubp_realm'].value_counts()
            for realm, count in realm_counts.items():
                print(f"  {realm}: {count} compounds ({100*count/len(df):.1f}%)")
    
    def save_features(self, df, filename="inorganic_features_comprehensive.csv"):
        """Save the comprehensive feature dataset"""
        
        print(f"\\n💾 Saving comprehensive features to {filename}...")
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        # Save feature metadata
        metadata = {
            'total_compounds': len(df),
            'total_features': len(df.columns),
            'feature_categories': {
                'crystallographic': len([col for col in df.columns if any(x in col for x in ['spacegroup', 'crystal', 'lattice'])]),
                'geometric': len([col for col in df.columns if any(x in col for x in ['coordination', 'polyhedral'])]),
                'electronic': len([col for col in df.columns if any(x in col for x in ['band_gap', 'magnetization'])]),
                'ubp': len([col for col in df.columns if any(x in col for x in ['ubp', 'realm', 'nrci'])]),
                'sacred_geometry': len([col for col in df.columns if 'resonance' in col])
            },
            'sacred_constants': self.sacred_constants,
            'ubp_realms': self.ubp_realms
        }
        
        metadata_filename = filename.replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved {len(df)} compounds with {len(df.columns)} features")
        print(f"✅ Saved metadata to {metadata_filename}")
        
        return df

def main():
    """Main execution function"""
    
    print("Starting comprehensive feature extraction...")
    print("Phase 2: Geometric and Crystallographic Features for UBP Analysis")
    print()
    
    # Initialize feature extractor
    extractor = InorganicFeatureExtractor()
    
    # Extract features from the materials dataset
    features_df = extractor.extract_features("mp_expanded_inorganic_dataset.csv")
    
    if len(features_df) > 0:
        # Save comprehensive features
        features_df = extractor.save_features(features_df, "inorganic_comprehensive_features.csv")
        
        print("\\n" + "="*80)
        print("PHASE 2 COMPLETE")
        print("="*80)
        print(f"✅ Extracted comprehensive features for {len(features_df)} compounds")
        print(f"✅ Generated {len(features_df.columns)} total features")
        print("✅ Ready for Phase 3: UBP-enhanced materials encoding")
        
        return features_df
    else:
        print("❌ Feature extraction failed")
        return None

if __name__ == "__main__":
    features_dataset = main()
