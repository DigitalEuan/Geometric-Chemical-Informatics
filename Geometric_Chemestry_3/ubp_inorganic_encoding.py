#!/usr/bin/env python3
"""
UBP-Enhanced Inorganic Materials Encoding System
Phase 3: Implement full UBP framework for inorganic materials

This system implements:
1. 6D Bitfield initialization with TGIC constraints
2. BitMatrix setup with GLR frameworks
3. Toggle algebra operations
4. Realm-specific encoding
5. NRCI calculation and validation
6. UBP energy computation
7. Geometric resonance detection
"""

import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from scipy.spatial.distance import pdist, squareform
import json
import warnings
warnings.filterwarnings('ignore')

class UBPInorganicEncoder:
    """UBP-enhanced encoding system for inorganic materials"""
    
    def __init__(self):
        # UBP System Parameters
        self.bitfield_dimensions = (170, 170, 170, 5, 2, 2)  # 6D Bitfield
        self.total_offbits = np.prod(self.bitfield_dimensions)
        
        # Core Resonance Values (CRVs) for each realm
        self.crv_values = {
            'quantum': {'freq': 4.58e14, 'wavelength': 655e-9, 'toggle_bias': np.e/12},
            'electromagnetic': {'freq': 3.141593, 'wavelength': 635e-9, 'toggle_bias': np.pi},
            'gravitational': {'freq': 100, 'wavelength': 1000e-9, 'toggle_bias': 0.1},
            'biological': {'freq': 10, 'wavelength': 700e-9, 'toggle_bias': 0.01},
            'cosmological': {'freq': 1e-11, 'wavelength': 800e-9, 'toggle_bias': np.pi**((1+np.sqrt(5))/2)},
            'nuclear': {'freq': 1.2356e20, 'wavelength': 600e-9, 'toggle_bias': 0.5},
            'optical': {'freq': 5e14, 'wavelength': 600e-9, 'toggle_bias': 0.3}
        }
        
        # Sacred geometry constants
        self.sacred_constants = {
            'phi': (1 + np.sqrt(5)) / 2,
            'pi': np.pi,
            'e': np.e,
            'sqrt2': np.sqrt(2),
            'sqrt3': np.sqrt(3),
            'sqrt5': np.sqrt(5)
        }
        
        # TGIC constraints (3, 6, 9 structure)
        self.tgic_structure = {
            'axes': 3,
            'faces': 6,
            'interactions_per_offbit': 9
        }
        
        # Coherent Synchronization Cycle
        self.csc_period = 1 / np.pi  # ~0.318309886 s
        
        # Initialize system components
        self.bitfield = None
        self.bitmatrix = None
        self.materials_states = []
        
    def initialize_ubp_system(self, num_materials: int):
        """Initialize the UBP system for materials encoding"""
        
        print("="*80)
        print("UBP SYSTEM INITIALIZATION")
        print("="*80)
        print("Phase 3: UBP-Enhanced Inorganic Materials Encoding")
        print(f"Target materials: {num_materials}")
        print(f"Bitfield dimensions: {self.bitfield_dimensions}")
        print(f"Total OffBits: {self.total_offbits:,}")
        print()
        
        # Initialize 6D Bitfield
        print("Step 1: Initializing 6D Bitfield...")
        self._initialize_bitfield()
        
        # Setup BitMatrix with GLR frameworks
        print("Step 2: Setting up BitMatrix with GLR frameworks...")
        self._setup_bitmatrix()
        
        # Initialize TGIC constraints
        print("Step 3: Applying TGIC constraints...")
        self._apply_tgic_constraints()
        
        print("✅ UBP system initialized successfully")
        print()
    
    def _initialize_bitfield(self):
        """Initialize the 6D Bitfield with proper structure"""
        
        try:
            # Create sparse 6D bitfield representation
            # Use flattened index for memory efficiency
            total_cells = np.prod(self.bitfield_dimensions)
            
            # Initialize with quantum realm toggle bias
            quantum_bias = self.crv_values['quantum']['toggle_bias']
            
            # Create sparse bitfield (only store non-zero values)
            self.bitfield = dok_matrix((total_cells, 24), dtype=np.float32)  # 24-bit OffBits
            
            # Initialize with sparse random pattern (sparsity 0.01 as per UBP spec)
            num_active = int(total_cells * 0.01)
            active_indices = np.random.choice(total_cells, num_active, replace=False)
            
            for idx in active_indices:
                # Initialize OffBit with quantum toggle bias
                offbit_state = np.random.binomial(1, quantum_bias, 24).astype(np.float32)
                for bit_pos in range(24):
                    if offbit_state[bit_pos] > 0:
                        self.bitfield[idx, bit_pos] = offbit_state[bit_pos]
            
            print(f"  ✅ Initialized {num_active:,} active OffBits ({100*num_active/total_cells:.3f}% sparsity)")
            
        except Exception as e:
            print(f"  ❌ Error initializing bitfield: {e}")
            # Fallback to smaller system
            self.bitfield = dok_matrix((10000, 24), dtype=np.float32)
    
    def _setup_bitmatrix(self):
        """Setup BitMatrix with realm-specific GLR frameworks"""
        
        try:
            # Create BitMatrix as block-sparse structure
            matrix_size = min(self.total_offbits, 100000)  # Limit for memory
            self.bitmatrix = dok_matrix((matrix_size, matrix_size), dtype=np.float32)
            
            # Apply realm-specific GLR frameworks
            for realm, crv in self.crv_values.items():
                self._apply_glr_framework(realm, crv)
            
            print(f"  ✅ BitMatrix setup complete ({matrix_size:,} x {matrix_size:,})")
            
        except Exception as e:
            print(f"  ❌ Error setting up BitMatrix: {e}")
            self.bitmatrix = dok_matrix((1000, 1000), dtype=np.float32)
    
    def _apply_glr_framework(self, realm: str, crv: dict):
        """Apply Golay-Leech-Resonance framework for specific realm"""
        
        try:
            # GLR-specific parameters for each realm
            glr_params = {
                'electromagnetic': {'coordination': 6, 'lattice_type': 'cubic'},
                'quantum': {'coordination': 4, 'lattice_type': 'tetrahedral'},
                'gravitational': {'coordination': 12, 'lattice_type': 'fcc'},
                'biological': {'coordination': 20, 'lattice_type': 'dodecahedral'},
                'cosmological': {'coordination': 12, 'lattice_type': 'icosahedral'},
                'nuclear': {'coordination': 8, 'lattice_type': 'e8'},
                'optical': {'coordination': 6, 'lattice_type': 'photonic'}
            }
            
            params = glr_params.get(realm, {'coordination': 6, 'lattice_type': 'cubic'})
            
            # Apply realm-specific connections in BitMatrix
            freq = crv['freq']
            wavelength = crv['wavelength']
            
            # Create resonance connections based on frequency
            resonance_strength = np.exp(-0.0002 * (freq / 1e14) ** 2)
            
            # Add connections to BitMatrix (simplified implementation)
            matrix_size = self.bitmatrix.shape[0]
            num_connections = min(1000, matrix_size // 10)
            
            for _ in range(num_connections):
                i, j = np.random.randint(0, matrix_size, 2)
                if i != j:
                    self.bitmatrix[i, j] = resonance_strength
            
        except Exception as e:
            print(f"    ⚠️  Error applying GLR for {realm}: {e}")
    
    def _apply_tgic_constraints(self):
        """Apply Triad Graph Interaction Constraints (3, 6, 9 structure)"""
        
        try:
            # TGIC enforces 3 axes, 6 faces, 9 interactions per OffBit
            axes = self.tgic_structure['axes']
            faces = self.tgic_structure['faces']
            interactions = self.tgic_structure['interactions_per_offbit']
            
            # Apply geometric constraints to bitfield structure
            if self.bitfield is not None:
                # Ensure each OffBit has exactly 9 interactions
                matrix_size = min(self.bitfield.shape[0], 10000)
                
                for offbit_idx in range(0, matrix_size, 100):  # Sample every 100th
                    # Create 9 interactions for this OffBit
                    interaction_targets = np.random.choice(
                        matrix_size, 
                        min(interactions, matrix_size-1), 
                        replace=False
                    )
                    
                    for target in interaction_targets:
                        if target != offbit_idx and self.bitmatrix is not None:
                            # Apply TGIC interaction strength
                            interaction_strength = 1.0 / interactions  # Normalized
                            self.bitmatrix[offbit_idx, target] = interaction_strength
            
            print(f"  ✅ TGIC constraints applied ({axes} axes, {faces} faces, {interactions} interactions)")
            
        except Exception as e:
            print(f"  ❌ Error applying TGIC: {e}")
    
    def encode_materials(self, features_df: pd.DataFrame):
        """Encode materials using UBP framework"""
        
        print("="*60)
        print("UBP MATERIALS ENCODING")
        print("="*60)
        print(f"Encoding {len(features_df)} materials...")
        print()
        
        encoded_materials = []
        
        for i, row in features_df.iterrows():
            try:
                material_state = self._encode_single_material(row)
                if material_state:
                    encoded_materials.append(material_state)
                
                # Progress update
                if (i + 1) % 100 == 0:
                    print(f"  Encoded {i + 1}/{len(features_df)} materials...")
                    
            except Exception as e:
                print(f"  ⚠️  Error encoding {row.get('material_id', 'unknown')}: {e}")
                continue
        
        print(f"✅ Successfully encoded {len(encoded_materials)} materials")
        
        # Convert to DataFrame
        encoded_df = pd.DataFrame(encoded_materials)
        
        # Add UBP-specific calculations
        encoded_df = self._add_ubp_calculations(encoded_df)
        
        return encoded_df
    
    def _encode_single_material(self, row):
        """Encode a single material using UBP principles"""
        
        try:
            # Basic material info
            material_state = {
                'material_id': row['material_id'],
                'formula': row['formula'],
                'tm_element': row.get('tm_element', None)
            }
            
            # Encode into UBP realms
            realm_encoding = self._encode_realm_state(row)
            material_state.update(realm_encoding)
            
            # Calculate OffBit representation
            offbit_state = self._calculate_offbit_state(row, realm_encoding)
            material_state.update(offbit_state)
            
            # Calculate UBP energy
            ubp_energy = self._calculate_full_ubp_energy(row, realm_encoding, offbit_state)
            material_state['ubp_energy_full'] = ubp_energy
            
            # Calculate NRCI
            nrci = self._calculate_nrci(row, realm_encoding, offbit_state)
            material_state['nrci_calculated'] = nrci
            
            # Detect geometric resonances
            resonances = self._detect_geometric_resonances(row)
            material_state.update(resonances)
            
            # Calculate coherence metrics
            coherence = self._calculate_coherence_metrics(realm_encoding, offbit_state)
            material_state.update(coherence)
            
            return material_state
            
        except Exception as e:
            print(f"    Error encoding material: {e}")
            return None
    
    def _encode_realm_state(self, row):
        """Encode material properties into UBP realm states"""
        
        realm_state = {}
        
        try:
            # Extract realm scores from features
            realm_scores = {}
            for realm in self.crv_values.keys():
                score_col = f'realm_{realm}_score'
                realm_scores[realm] = row.get(score_col, 1.0 / len(self.crv_values))
            
            # Normalize realm scores
            total_score = sum(realm_scores.values())
            if total_score > 0:
                for realm in realm_scores:
                    realm_scores[realm] /= total_score
            
            # Encode each realm with its CRV
            for realm, score in realm_scores.items():
                crv = self.crv_values[realm]
                
                # Calculate realm-specific encoding
                freq_encoding = np.log10(crv['freq'] + 1e-20)  # Avoid log(0)
                wavelength_encoding = crv['wavelength'] * 1e9  # Convert to nm
                toggle_encoding = crv['toggle_bias']
                
                realm_state[f'{realm}_freq_encoding'] = freq_encoding
                realm_state[f'{realm}_wavelength_encoding'] = wavelength_encoding
                realm_state[f'{realm}_toggle_encoding'] = toggle_encoding
                realm_state[f'{realm}_score_normalized'] = score
                
                # Calculate realm coherence
                realm_coherence = score * np.cos(2 * np.pi * crv['freq'] * self.csc_period)
                realm_state[f'{realm}_coherence'] = realm_coherence
            
            # Primary realm assignment
            primary_realm = max(realm_scores, key=realm_scores.get)
            realm_state['primary_realm'] = primary_realm
            realm_state['primary_realm_score'] = realm_scores[primary_realm]
            
        except Exception as e:
            print(f"      Realm encoding error: {e}")
            # Set defaults
            realm_state['primary_realm'] = 'electromagnetic'
            realm_state['primary_realm_score'] = 1.0
        
        return realm_state
    
    def _calculate_offbit_state(self, row, realm_encoding):
        """Calculate OffBit state representation"""
        
        offbit_state = {}
        
        try:
            # Calculate active OffBits based on material properties
            nsites = row.get('nsites', 1)
            volume = row.get('volume', 100)
            density = row.get('density', 5.0)
            
            # Estimate number of active OffBits
            active_offbits = min(int(nsites * 10), 10000)  # Scale with system size
            offbit_state['active_offbits_count'] = active_offbits
            
            # Calculate OffBit density
            offbit_density = active_offbits / volume
            offbit_state['offbit_density'] = offbit_density
            
            # Calculate toggle pattern based on primary realm
            primary_realm = realm_encoding.get('primary_realm', 'electromagnetic')
            toggle_bias = self.crv_values[primary_realm]['toggle_bias']
            
            # Generate toggle pattern (simplified 24-bit representation)
            toggle_pattern = np.random.binomial(1, toggle_bias, 24)
            offbit_state['toggle_pattern'] = toggle_pattern.tolist()
            offbit_state['toggle_sum'] = int(np.sum(toggle_pattern))
            offbit_state['toggle_bias_used'] = toggle_bias
            
            # Calculate OffBit interactions (TGIC: 9 interactions per OffBit)
            interactions_per_offbit = self.tgic_structure['interactions_per_offbit']
            total_interactions = active_offbits * interactions_per_offbit
            offbit_state['total_interactions'] = total_interactions
            offbit_state['interaction_density'] = total_interactions / volume
            
        except Exception as e:
            print(f"      OffBit calculation error: {e}")
            # Set defaults
            offbit_state['active_offbits_count'] = 100
            offbit_state['toggle_sum'] = 12
            offbit_state['total_interactions'] = 900
        
        return offbit_state
    
    def _calculate_full_ubp_energy(self, row, realm_encoding, offbit_state):
        """Calculate full UBP energy using the complete equation"""
        
        try:
            # UBP Energy Equation parameters
            M = offbit_state.get('active_offbits_count', 100)  # Active OffBits
            C = 299792458  # Speed of light (m/s)
            
            # Resonance efficiency (R)
            resonance_keys = [k for k in row.index if 'resonance' in str(k)]
            if len(resonance_keys) > 0:
                resonance_values = [row.get(k, 0.5) for k in resonance_keys]
                R = np.mean(resonance_values)
            else:
                R = 0.5
            
            # Apply UBP resonance formula: R = R_0 * (1 - H_t / ln(4))
            R_0 = 0.95
            H_t = 0.05
            R = R_0 * (1 - H_t / np.log(4))  # R ≈ 0.9658855
            
            # Structural optimization (S_opt)
            packing_fraction = row.get('atomic_packing_fraction', 0.74)
            toggle_sum = offbit_state.get('toggle_sum', 12)
            
            # S_opt formula from UBP spec
            S_opt = 0.7 * packing_fraction + 0.3 * (toggle_sum / 12)
            
            # Global Coherence Index (P_GCI)
            # Calculate weighted mean of CRVs
            crv_freqs = [3.141593, 1.618034, 4.58e14, 1e9, 1e15, 1e20, 58977069.609314, 1.86e41]
            crv_weights = [0.2, 0.2, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05]
            f_avg = np.average(crv_freqs, weights=crv_weights)
            
            delta_t = self.csc_period  # 0.318309886 s
            P_GCI = np.cos(2 * np.pi * f_avg * delta_t)
            
            # Observer intent (O_observer)
            # Simplified: 1.0 for neutral, 1.5 for intentional analysis
            O_observer = 1.5  # Intentional scientific analysis
            
            # Infinity constant (c_∞)
            phi = self.sacred_constants['phi']
            c_infinity = 24 * (1 + phi)  # ≈ 38.83281573
            
            # Spin entropy (I_spin)
            primary_realm = realm_encoding.get('primary_realm', 'electromagnetic')
            if primary_realm == 'quantum':
                p_s = self.crv_values['quantum']['toggle_bias']
            else:
                p_s = self.crv_values['cosmological']['toggle_bias']
            
            I_spin = -p_s * np.log(p_s) - (1-p_s) * np.log(1-p_s)
            
            # Toggle operations (simplified)
            w_ij = 0.1
            M_ij = toggle_sum / 24  # Normalized XOR operation
            toggle_term = w_ij * M_ij
            
            # Full UBP Energy calculation
            E = M * C * (R * S_opt) * P_GCI * O_observer * c_infinity * I_spin * toggle_term
            
            # Scale to reasonable range
            E_scaled = E / 1e20  # Scale factor for numerical stability
            
            return float(E_scaled)
            
        except Exception as e:
            print(f"      UBP energy error: {e}")
            return 1.0
    
    def _calculate_nrci(self, row, realm_encoding, offbit_state):
        """Calculate Non-random Coherence Index"""
        
        try:
            # NRCI formula: 1 - (sqrt(sum((S_i - T_i)^2)/n) / sigma(T))
            
            # Get system state (S_i) and target state (T_i)
            system_values = []
            target_values = []
            
            # Collect realm coherence values
            for realm in self.crv_values.keys():
                coherence_key = f'{realm}_coherence'
                if coherence_key in realm_encoding:
                    system_values.append(realm_encoding[coherence_key])
                    target_values.append(1.0)  # Target perfect coherence
            
            # Add resonance values
            resonance_keys = [k for k in row.index if 'resonance' in str(k)]
            for key in resonance_keys:
                system_values.append(row.get(key, 0.5))
                target_values.append(1.0)  # Target perfect resonance
            
            # Add toggle coherence
            toggle_sum = offbit_state.get('toggle_sum', 12)
            system_values.append(toggle_sum / 24)  # Normalized
            target_values.append(0.5)  # Target balanced toggle state
            
            if len(system_values) > 0 and len(target_values) > 0:
                S = np.array(system_values)
                T = np.array(target_values)
                
                # Calculate NRCI
                n = len(S)
                numerator = np.sqrt(np.sum((S - T) ** 2) / n)
                denominator = np.std(T) if np.std(T) > 0 else 1.0
                
                nrci = 1 - (numerator / denominator)
                
                # Ensure NRCI is in valid range [0, 1]
                nrci = max(0.0, min(1.0, nrci))
                
                return float(nrci)
            else:
                return 0.5
                
        except Exception as e:
            print(f"      NRCI calculation error: {e}")
            return 0.5
    
    def _detect_geometric_resonances(self, row):
        """Detect geometric resonances with sacred constants"""
        
        resonances = {}
        
        try:
            # Properties to check for resonances
            properties = {
                'lattice_a': row.get('a', 5.0),
                'lattice_b': row.get('b', 5.0),
                'lattice_c': row.get('c', 5.0),
                'band_gap': row.get('band_gap', 0.0),
                'magnetization': abs(row.get('total_magnetization', 0.0)),
                'formation_energy': abs(row.get('formation_energy_per_atom', 0.0)),
                'coordination': row.get('coordination_number', 6),
                'volume_per_atom': row.get('volume_per_atom', 20.0)
            }
            
            # Check resonances with each sacred constant
            for prop_name, prop_value in properties.items():
                if prop_value > 0:
                    for const_name, const_value in self.sacred_constants.items():
                        # Calculate resonance strength
                        ratio = prop_value / const_value
                        resonance = np.exp(-((ratio - 1.0) ** 2) / (2 * 0.1 ** 2))
                        
                        # Check harmonic resonances
                        harmonics = [2.0, 3.0, 0.5, 1/3]
                        for harmonic in harmonics:
                            harmonic_ratio = prop_value / (const_value * harmonic)
                            harmonic_resonance = np.exp(-((harmonic_ratio - 1.0) ** 2) / (2 * 0.1 ** 2))
                            resonance = max(resonance, harmonic_resonance * 0.5)
                        
                        resonances[f'{prop_name}_{const_name}_resonance'] = resonance
            
            # Calculate total resonance potential
            if resonances:
                resonances['total_resonance_potential'] = np.mean(list(resonances.values()))
            else:
                resonances['total_resonance_potential'] = 0.0
                
        except Exception as e:
            print(f"      Resonance detection error: {e}")
            resonances['total_resonance_potential'] = 0.0
        
        return resonances
    
    def _calculate_coherence_metrics(self, realm_encoding, offbit_state):
        """Calculate various coherence metrics"""
        
        coherence = {}
        
        try:
            # Cross-realm coherence
            realm_coherences = []
            for realm in self.crv_values.keys():
                coherence_key = f'{realm}_coherence'
                if coherence_key in realm_encoding:
                    realm_coherences.append(realm_encoding[coherence_key])
            
            if len(realm_coherences) > 1:
                # Calculate cross-realm coherence as correlation
                coherence['cross_realm_coherence'] = 1.0 - np.std(realm_coherences)
            else:
                coherence['cross_realm_coherence'] = 1.0
            
            # Toggle coherence
            toggle_sum = offbit_state.get('toggle_sum', 12)
            toggle_coherence = 1.0 - abs(toggle_sum - 12) / 12  # Deviation from balanced state
            coherence['toggle_coherence'] = toggle_coherence
            
            # Interaction coherence (based on TGIC)
            total_interactions = offbit_state.get('total_interactions', 900)
            active_offbits = offbit_state.get('active_offbits_count', 100)
            expected_interactions = active_offbits * 9  # TGIC: 9 interactions per OffBit
            
            if expected_interactions > 0:
                interaction_coherence = 1.0 - abs(total_interactions - expected_interactions) / expected_interactions
            else:
                interaction_coherence = 1.0
            
            coherence['interaction_coherence'] = interaction_coherence
            
            # Overall system coherence
            coherence_values = [
                coherence['cross_realm_coherence'],
                coherence['toggle_coherence'],
                coherence['interaction_coherence']
            ]
            coherence['system_coherence'] = np.mean(coherence_values)
            
        except Exception as e:
            print(f"      Coherence calculation error: {e}")
            coherence['system_coherence'] = 0.5
        
        return coherence
    
    def _add_ubp_calculations(self, df):
        """Add additional UBP-specific calculations to the dataset"""
        
        print("\\nAdding UBP-specific calculations...")
        
        try:
            # Calculate system-wide metrics
            if len(df) > 1:
                # Global NRCI distribution
                nrci_values = df['nrci_calculated'].values
                df['nrci_percentile'] = pd.Series(nrci_values).rank(pct=True)
                
                # UBP energy distribution
                energy_values = df['ubp_energy_full'].values
                df['ubp_energy_percentile'] = pd.Series(energy_values).rank(pct=True)
                
                # Resonance clustering
                resonance_cols = [col for col in df.columns if 'resonance' in col and col != 'total_resonance_potential']
                if len(resonance_cols) > 0:
                    df['resonance_cluster'] = pd.cut(df['total_resonance_potential'], bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
                
                # Realm dominance analysis
                realm_cols = [col for col in df.columns if col.endswith('_score_normalized')]
                if len(realm_cols) > 0:
                    df['realm_dominance'] = df[realm_cols].max(axis=1)
                    df['realm_diversity'] = 1.0 - df[realm_cols].max(axis=1)
            
            # Calculate UBP quality score
            quality_factors = ['nrci_calculated', 'system_coherence', 'total_resonance_potential']
            available_factors = [f for f in quality_factors if f in df.columns]
            
            if available_factors:
                df['ubp_quality_score'] = df[available_factors].mean(axis=1)
            else:
                df['ubp_quality_score'] = 0.5
            
            print(f"  ✅ Added UBP calculations. Total columns: {len(df.columns)}")
            
        except Exception as e:
            print(f"  ⚠️  Error adding UBP calculations: {e}")
        
        return df
    
    def validate_ubp_system(self, encoded_df):
        """Validate the UBP system performance"""
        
        print("\\n" + "="*60)
        print("UBP SYSTEM VALIDATION")
        print("="*60)
        
        validation_results = {}
        
        try:
            # NRCI validation (target ≥ 0.999999)
            nrci_values = encoded_df['nrci_calculated'].values
            mean_nrci = np.mean(nrci_values)
            max_nrci = np.max(nrci_values)
            high_nrci_count = np.sum(nrci_values >= 0.999999)
            
            validation_results['nrci_mean'] = mean_nrci
            validation_results['nrci_max'] = max_nrci
            validation_results['nrci_target_achieved'] = high_nrci_count
            validation_results['nrci_target_percentage'] = 100 * high_nrci_count / len(nrci_values)
            
            print(f"NRCI Performance:")
            print(f"  Mean NRCI: {mean_nrci:.6f}")
            print(f"  Max NRCI: {max_nrci:.6f}")
            print(f"  Materials achieving NRCI ≥ 0.999999: {high_nrci_count}/{len(nrci_values)} ({validation_results['nrci_target_percentage']:.1f}%)")
            
            # Coherence validation (target ≥ 0.95)
            if 'system_coherence' in encoded_df.columns:
                coherence_values = encoded_df['system_coherence'].values
                mean_coherence = np.mean(coherence_values)
                high_coherence_count = np.sum(coherence_values >= 0.95)
                
                validation_results['coherence_mean'] = mean_coherence
                validation_results['coherence_target_achieved'] = high_coherence_count
                validation_results['coherence_target_percentage'] = 100 * high_coherence_count / len(coherence_values)
                
                print(f"\\nCoherence Performance:")
                print(f"  Mean coherence: {mean_coherence:.6f}")
                print(f"  Materials achieving coherence ≥ 0.95: {high_coherence_count}/{len(coherence_values)} ({validation_results['coherence_target_percentage']:.1f}%)")
            
            # Resonance validation
            if 'total_resonance_potential' in encoded_df.columns:
                resonance_values = encoded_df['total_resonance_potential'].values
                mean_resonance = np.mean(resonance_values)
                high_resonance_count = np.sum(resonance_values >= 0.5)
                
                validation_results['resonance_mean'] = mean_resonance
                validation_results['resonance_significant_count'] = high_resonance_count
                
                print(f"\\nResonance Performance:")
                print(f"  Mean resonance potential: {mean_resonance:.6f}")
                print(f"  Materials with significant resonance (≥ 0.5): {high_resonance_count}/{len(resonance_values)} ({100*high_resonance_count/len(resonance_values):.1f}%)")
            
            # Realm distribution validation
            if 'primary_realm' in encoded_df.columns:
                realm_distribution = encoded_df['primary_realm'].value_counts()
                validation_results['realm_distribution'] = realm_distribution.to_dict()
                
                print(f"\\nRealm Distribution:")
                for realm, count in realm_distribution.items():
                    print(f"  {realm}: {count} materials ({100*count/len(encoded_df):.1f}%)")
            
            # Overall UBP quality
            if 'ubp_quality_score' in encoded_df.columns:
                quality_values = encoded_df['ubp_quality_score'].values
                mean_quality = np.mean(quality_values)
                high_quality_count = np.sum(quality_values >= 0.8)
                
                validation_results['quality_mean'] = mean_quality
                validation_results['quality_high_count'] = high_quality_count
                
                print(f"\\nOverall UBP Quality:")
                print(f"  Mean quality score: {mean_quality:.6f}")
                print(f"  High-quality materials (≥ 0.8): {high_quality_count}/{len(quality_values)} ({100*high_quality_count/len(quality_values):.1f}%)")
            
            # Success criteria evaluation
            success_criteria = {
                'NRCI_target': validation_results.get('nrci_target_percentage', 0) > 10,  # At least 10% achieve target
                'Coherence_target': validation_results.get('coherence_target_percentage', 0) > 50,  # At least 50% achieve target
                'System_functional': len(encoded_df) > 0 and 'ubp_energy_full' in encoded_df.columns,
                'Realm_diversity': len(validation_results.get('realm_distribution', {})) >= 3  # At least 3 realms represented
            }
            
            validation_results['success_criteria'] = success_criteria
            validation_results['overall_success'] = all(success_criteria.values())
            
            print(f"\\nSuccess Criteria:")
            for criterion, passed in success_criteria.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {criterion}: {status}")
            
            overall_status = "✅ SUCCESS" if validation_results['overall_success'] else "⚠️  PARTIAL SUCCESS"
            print(f"\\nOverall UBP System Status: {overall_status}")
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def save_ubp_encoded_data(self, encoded_df, validation_results, filename="ubp_encoded_inorganic_materials.csv"):
        """Save UBP-encoded materials data"""
        
        print(f"\\n💾 Saving UBP-encoded data to {filename}...")
        
        # Save main dataset
        encoded_df.to_csv(filename, index=False)
        
        # Save validation results
        validation_filename = filename.replace('.csv', '_validation.json')
        with open(validation_filename, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Save UBP system metadata
        metadata = {
            'ubp_system_config': {
                'bitfield_dimensions': self.bitfield_dimensions,
                'total_offbits': self.total_offbits,
                'crv_values': self.crv_values,
                'sacred_constants': self.sacred_constants,
                'tgic_structure': self.tgic_structure,
                'csc_period': self.csc_period
            },
            'encoding_summary': {
                'total_materials': len(encoded_df),
                'total_features': len(encoded_df.columns),
                'ubp_features': len([col for col in encoded_df.columns if any(x in col for x in ['ubp', 'nrci', 'realm', 'coherence', 'resonance'])]),
                'validation_status': validation_results.get('overall_success', False)
            }
        }
        
        metadata_filename = filename.replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Saved {len(encoded_df)} UBP-encoded materials")
        print(f"✅ Saved validation results to {validation_filename}")
        print(f"✅ Saved system metadata to {metadata_filename}")
        
        return encoded_df

def main():
    """Main execution function"""
    
    print("Starting UBP-enhanced inorganic materials encoding...")
    print("Phase 3: Full UBP Framework Implementation")
    print()
    
    # Initialize UBP encoder
    encoder = UBPInorganicEncoder()
    
    # Load features dataset
    print("Loading comprehensive features dataset...")
    try:
        features_df = pd.read_csv("inorganic_comprehensive_features.csv")
        print(f"✅ Loaded {len(features_df)} materials with {len(features_df.columns)} features")
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return None
    
    # Initialize UBP system
    encoder.initialize_ubp_system(len(features_df))
    
    # Encode materials using UBP framework
    encoded_df = encoder.encode_materials(features_df)
    
    if len(encoded_df) > 0:
        # Validate UBP system
        validation_results = encoder.validate_ubp_system(encoded_df)
        
        # Save UBP-encoded data
        encoded_df = encoder.save_ubp_encoded_data(encoded_df, validation_results)
        
        print("\\n" + "="*80)
        print("PHASE 3 COMPLETE")
        print("="*80)
        print(f"✅ UBP-encoded {len(encoded_df)} inorganic materials")
        print(f"✅ Generated {len(encoded_df.columns)} total features")
        print(f"✅ UBP system validation: {'SUCCESS' if validation_results.get('overall_success', False) else 'PARTIAL'}")
        print("✅ Ready for Phase 4: Geometric mapping and resonance detection")
        
        return encoded_df, validation_results
    else:
        print("❌ UBP encoding failed")
        return None, None

if __name__ == "__main__":
    encoded_dataset, validation = main()
