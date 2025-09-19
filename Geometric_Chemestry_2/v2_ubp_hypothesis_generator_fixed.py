#!/usr/bin/env python3
"""
Version 2 UBP-Enhanced Geometric Hypothesis Generator - Fixed
Integrating Universal Binary Principle system for advanced chemical discovery
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import time
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# UBP System Components
@dataclass
class OffBit:
    """UBP OffBit representation for molecular encoding"""
    value: int = 0
    realm: str = "biological"
    resonance_frequency: float = 10.0  # Hz
    coherence_state: float = 0.0
    toggle_count: int = 0
    metadata: Optional[Dict] = None

@dataclass
class UBPMolecularState:
    """UBP state representation for molecules"""
    offbits: List[OffBit]
    nrci_score: float
    coherence_matrix: np.ndarray
    temporal_phase: float
    energy_state: float
    realm_distribution: Dict[str, float]

@dataclass
class GeometricHypothesis:
    """Generated geometric hypothesis for drug discovery"""
    hypothesis_id: str
    molecular_pattern: str
    geometric_signature: Dict[str, float]
    ubp_encoding: UBPMolecularState
    predicted_activity: float
    confidence_score: float
    nrci_validation: float
    supporting_evidence: List[str]
    testable_predictions: List[str]

class UBPConstants:
    """UBP Core Resonance Values and Constants"""
    
    # Core Resonance Values (CRVs)
    CRV_QUANTUM = 0.2265234857  # e/12
    CRV_ELECTROMAGNETIC = 3.141593  # π
    CRV_GRAVITATIONAL = 100.0
    CRV_BIOLOGICAL = 10.0
    CRV_COSMOLOGICAL = 0.83203682  # π^φ
    CRV_GOLDEN_RATIO = 1.618034  # φ
    CRV_SQRT2 = 1.414214  # √2
    CRV_EULER = 2.718282  # e
    
    # Planck-scale constants
    PLANCK_TIME = 5.391247e-44  # seconds
    PLANCK_LENGTH = 1.616255e-35  # meters
    LIGHT_SPEED = 299792458.0  # m/s
    HBAR = 1.054571817e-34  # J⋅s
    
    # UBP-specific parameters
    NRCI_TARGET = 0.999999
    COHERENCE_THRESHOLD = 0.95
    TGIC_STRUCTURE = (3, 6, 9)  # 3 axes, 6 faces, 9 interactions

class UBPGeometricHypothesisGenerator:
    """UBP-enhanced geometric hypothesis generator for chemical discovery"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.constants = UBPConstants()
        
        # Initialize UBP components
        self.realm_frequencies = {
            'quantum': self.constants.CRV_QUANTUM,
            'electromagnetic': self.constants.CRV_ELECTROMAGNETIC,
            'gravitational': self.constants.CRV_GRAVITATIONAL,
            'biological': self.constants.CRV_BIOLOGICAL,
            'cosmological': self.constants.CRV_COSMOLOGICAL
        }
        
        # Hypothesis storage
        self.generated_hypotheses = []
        self.validation_results = {}
        
    def load_baseline_data(self):
        """Load baseline study results for UBP enhancement"""
        print("Loading baseline study data for UBP enhancement...")
        
        # Load original dataset
        self.dataset = pd.read_csv("v2_comprehensive_features.csv")
        self.target = self.dataset['pKi'].values
        
        # Load fingerprints
        self.fingerprints = np.load("v2_ecfp4_fingerprints.npy")
        
        # Load geometric analysis results
        self.geometric_results = pd.read_csv("v2_geometric_analysis_results.csv")
        
        # Load predictive modeling results
        self.modeling_results = pd.read_csv("v2_predictive_modeling_results.csv")
        
        print(f"Loaded: {len(self.dataset)} compounds, {self.fingerprints.shape[1]} fingerprint features")
        print(f"Baseline best R²: {self.modeling_results['test_r2'].max():.4f}")
        
        return self.dataset
    
    def encode_molecules_as_ubp_states(self) -> List[UBPMolecularState]:
        """Encode molecules using UBP OffBit representation"""
        print("Encoding molecules as UBP states...")
        
        ubp_states = []
        
        for i, (_, compound) in enumerate(self.dataset.iterrows()):
            # Create OffBits for molecular representation
            offbits = self._create_molecular_offbits(compound, i)
            
            # Calculate NRCI score
            nrci_score = self._calculate_molecular_nrci(offbits)
            
            # Build coherence matrix
            coherence_matrix = self._build_coherence_matrix(offbits)
            
            # Calculate temporal phase
            temporal_phase = self._calculate_temporal_phase(offbits)
            
            # Calculate energy state using UBP energy equation
            energy_state = self._calculate_ubp_energy(offbits)
            
            # Determine realm distribution
            realm_distribution = self._calculate_realm_distribution(offbits)
            
            ubp_state = UBPMolecularState(
                offbits=offbits,
                nrci_score=nrci_score,
                coherence_matrix=coherence_matrix,
                temporal_phase=temporal_phase,
                energy_state=energy_state,
                realm_distribution=realm_distribution
            )
            
            ubp_states.append(ubp_state)
        
        self.ubp_molecular_states = ubp_states
        print(f"Encoded {len(ubp_states)} molecules as UBP states")
        
        return ubp_states
    
    def _create_molecular_offbits(self, compound: pd.Series, compound_idx: int) -> List[OffBit]:
        """Create OffBits for molecular representation"""
        offbits = []
        
        # Extract molecular features
        mordred_features = [col for col in compound.index if col.startswith('mordred_')]
        fingerprint_bits = self.fingerprints[compound_idx]
        
        # Create OffBits for key molecular properties
        # Molecular weight -> Gravitational realm
        if 'mordred_MW' in compound.index:
            mw_offbit = OffBit(
                value=int(compound['mordred_MW'] % 1024) if not pd.isna(compound['mordred_MW']) else 0,
                realm="gravitational",
                resonance_frequency=self.constants.CRV_GRAVITATIONAL,
                coherence_state=0.8
            )
            offbits.append(mw_offbit)
        
        # LogP -> Electromagnetic realm
        if 'mordred_SlogP' in compound.index:
            logp_offbit = OffBit(
                value=int((compound['mordred_SlogP'] + 5) * 100) % 1024 if not pd.isna(compound['mordred_SlogP']) else 0,
                realm="electromagnetic",
                resonance_frequency=self.constants.CRV_ELECTROMAGNETIC,
                coherence_state=0.9
            )
            offbits.append(logp_offbit)
        
        # Activity -> Biological realm
        activity_offbit = OffBit(
            value=int(compound['pKi'] * 100) % 1024,
            realm="biological",
            resonance_frequency=self.constants.CRV_BIOLOGICAL,
            coherence_state=1.0
        )
        offbits.append(activity_offbit)
        
        # Fingerprint bits -> Quantum realm
        active_fingerprint_bits = np.where(fingerprint_bits > 0)[0]
        for bit_idx in active_fingerprint_bits[:10]:  # Limit to first 10 active bits
            fp_offbit = OffBit(
                value=bit_idx % 1024,
                realm="quantum",
                resonance_frequency=self.constants.CRV_QUANTUM,
                coherence_state=0.7
            )
            offbits.append(fp_offbit)
        
        # Sacred geometry features -> Cosmological realm
        if compound_idx < len(self.geometric_results):
            geom_data = self.geometric_results.iloc[compound_idx % len(self.geometric_results)]
            
            phi_resonance = geom_data.get('phi_resonance', 0)
            if not pd.isna(phi_resonance):
                phi_offbit = OffBit(
                    value=int(phi_resonance * 1000) % 1024,
                    realm="cosmological",
                    resonance_frequency=self.constants.CRV_COSMOLOGICAL,
                    coherence_state=phi_resonance
                )
                offbits.append(phi_offbit)
        
        return offbits
    
    def _calculate_molecular_nrci(self, offbits: List[OffBit]) -> float:
        """Calculate NRCI score for molecular OffBits"""
        if not offbits:
            return 0.0
        
        # Extract values and theoretical expectations
        observed_values = np.array([offbit.value for offbit in offbits])
        theoretical_values = np.array([offbit.resonance_frequency * 100 for offbit in offbits])
        
        # Calculate NRCI: 1 - (RMSE / σ(theoretical))
        rmse = np.sqrt(np.mean((observed_values - theoretical_values) ** 2))
        sigma_theoretical = np.std(theoretical_values)
        
        if sigma_theoretical == 0:
            return 1.0 if rmse == 0 else 0.0
        
        nrci = 1 - (rmse / sigma_theoretical)
        return max(0.0, min(1.0, nrci))
    
    def _build_coherence_matrix(self, offbits: List[OffBit]) -> np.ndarray:
        """Build coherence matrix for OffBits"""
        n_bits = len(offbits)
        if n_bits == 0:
            return np.array([[]])
        
        coherence_matrix = np.eye(n_bits)
        
        for i in range(n_bits):
            for j in range(i + 1, n_bits):
                # Calculate coherence between OffBits
                bit1, bit2 = offbits[i], offbits[j]
                
                # Frequency coherence
                freq_diff = abs(bit1.resonance_frequency - bit2.resonance_frequency)
                freq_coherence = np.exp(-freq_diff / 100.0)
                
                # Value coherence
                value_diff = abs(bit1.value - bit2.value)
                value_coherence = np.exp(-value_diff / 1000.0)
                
                # Realm coherence
                realm_coherence = 1.0 if bit1.realm == bit2.realm else 0.5
                
                # Combined coherence
                total_coherence = (freq_coherence + value_coherence + realm_coherence) / 3.0
                
                coherence_matrix[i, j] = total_coherence
                coherence_matrix[j, i] = total_coherence
        
        return coherence_matrix
    
    def _calculate_temporal_phase(self, offbits: List[OffBit]) -> float:
        """Calculate temporal phase for molecular state"""
        if not offbits:
            return 0.0
        
        # Phase based on resonance frequencies and values
        phase_sum = 0.0
        for offbit in offbits:
            # Convert to Planck time units
            planck_units = offbit.value * self.constants.PLANCK_TIME
            phase_contribution = np.sin(2 * np.pi * offbit.resonance_frequency * planck_units)
            phase_sum += phase_contribution
        
        # Normalize phase to [0, 2π]
        phase = (phase_sum / len(offbits)) % (2 * np.pi)
        return phase
    
    def _calculate_ubp_energy(self, offbits: List[OffBit]) -> float:
        """Calculate UBP energy using the energy equation"""
        if not offbits:
            return 0.0
        
        # UBP Energy Equation components
        M = len(offbits)  # Active OffBits count
        C = self.constants.LIGHT_SPEED
        
        # Resonance efficiency
        R = 0.9658855  # From UBP specification
        
        # Structural stability (based on coherence)
        coherence_matrix = self._build_coherence_matrix(offbits)
        S_opt = np.mean(coherence_matrix) if coherence_matrix.size > 0 else 0.98
        
        # Global coherence index
        f_avg = np.mean([offbit.resonance_frequency for offbit in offbits])
        delta_t = 1 / np.pi  # CSC period
        P_GCI = np.cos(2 * np.pi * f_avg * delta_t)
        
        # Observer intent (neutral)
        O_observer = 1.0
        
        # Infinity constant
        c_infinity = 24 * (1 + self.constants.CRV_GOLDEN_RATIO)
        
        # Spin entropy
        p_s = self.constants.CRV_QUANTUM
        I_spin = p_s * np.log(1 / p_s) if p_s > 0 else 0
        
        # Toggle operations (simplified)
        w_ij = 0.1
        M_ij_sum = sum(abs(offbits[i].value - offbits[j].value) 
                      for i in range(len(offbits)) 
                      for j in range(i + 1, len(offbits)))
        
        # Calculate energy
        energy = (M * C * (R * S_opt) * P_GCI * O_observer * 
                 c_infinity * I_spin * w_ij * M_ij_sum)
        
        return energy
    
    def _calculate_realm_distribution(self, offbits: List[OffBit]) -> Dict[str, float]:
        """Calculate distribution across UBP realms"""
        if not offbits:
            return {}
        
        realm_counts = {}
        for offbit in offbits:
            realm_counts[offbit.realm] = realm_counts.get(offbit.realm, 0) + 1
        
        total_bits = len(offbits)
        realm_distribution = {realm: count / total_bits 
                            for realm, count in realm_counts.items()}
        
        return realm_distribution
    
    def generate_geometric_hypotheses(self, n_hypotheses: int = 10) -> List[GeometricHypothesis]:
        """Generate geometric hypotheses using UBP principles"""
        print(f"Generating {n_hypotheses} UBP-enhanced geometric hypotheses...")
        
        hypotheses = []
        
        # Analyze UBP molecular states for patterns
        high_nrci_molecules = [state for state in self.ubp_molecular_states 
                              if state.nrci_score > 0.5]
        
        if not high_nrci_molecules:
            print("Warning: No high-NRCI molecules found for hypothesis generation")
            return []
        
        # Generate hypotheses based on UBP patterns
        for i in range(n_hypotheses):
            hypothesis = self._generate_single_hypothesis(high_nrci_molecules, i)
            hypotheses.append(hypothesis)
        
        self.generated_hypotheses = hypotheses
        print(f"Generated {len(hypotheses)} geometric hypotheses")
        
        return hypotheses
    
    def _generate_single_hypothesis(self, high_nrci_molecules: List[UBPMolecularState], 
                                   hypothesis_idx: int) -> GeometricHypothesis:
        """Generate a single geometric hypothesis"""
        
        # Select representative molecules
        if len(high_nrci_molecules) > 3:
            selected_indices = np.random.choice(len(high_nrci_molecules), 3, replace=False)
            selected_molecules = [high_nrci_molecules[i] for i in selected_indices]
        else:
            selected_molecules = high_nrci_molecules
        
        # Extract geometric signature
        geometric_signature = self._extract_geometric_signature(selected_molecules)
        
        # Create molecular pattern description
        molecular_pattern = self._describe_molecular_pattern(selected_molecules)
        
        # Predict activity using UBP energy
        predicted_activity = np.mean([mol.energy_state for mol in selected_molecules]) / 1e10
        
        # Calculate confidence based on NRCI
        confidence_score = np.mean([mol.nrci_score for mol in selected_molecules])
        
        # NRCI validation
        nrci_validation = self._validate_hypothesis_nrci(selected_molecules)
        
        # Generate supporting evidence
        supporting_evidence = self._generate_supporting_evidence(selected_molecules, geometric_signature)
        
        # Generate testable predictions
        testable_predictions = self._generate_testable_predictions(geometric_signature, predicted_activity)
        
        # Create representative UBP encoding
        representative_encoding = selected_molecules[0] if selected_molecules else None
        
        hypothesis = GeometricHypothesis(
            hypothesis_id=f"UBP_GH_{hypothesis_idx:03d}",
            molecular_pattern=molecular_pattern,
            geometric_signature=geometric_signature,
            ubp_encoding=representative_encoding,
            predicted_activity=predicted_activity,
            confidence_score=confidence_score,
            nrci_validation=nrci_validation,
            supporting_evidence=supporting_evidence,
            testable_predictions=testable_predictions
        )
        
        return hypothesis
    
    def _extract_geometric_signature(self, molecules: List[UBPMolecularState]) -> Dict[str, float]:
        """Extract geometric signature from UBP molecular states"""
        signature = {}
        
        # NRCI-based signature
        signature['mean_nrci'] = np.mean([mol.nrci_score for mol in molecules])
        signature['nrci_variance'] = np.var([mol.nrci_score for mol in molecules])
        
        # Energy-based signature
        energies = [mol.energy_state for mol in molecules]
        signature['mean_energy'] = np.mean(energies)
        signature['energy_range'] = np.max(energies) - np.min(energies)
        
        # Temporal phase signature
        phases = [mol.temporal_phase for mol in molecules]
        signature['phase_coherence'] = np.abs(np.mean(np.exp(1j * np.array(phases))))
        
        # Realm distribution signature
        all_realms = set()
        for mol in molecules:
            all_realms.update(mol.realm_distribution.keys())
        
        for realm in all_realms:
            realm_values = [mol.realm_distribution.get(realm, 0) for mol in molecules]
            signature[f'{realm}_distribution'] = np.mean(realm_values)
        
        # Sacred geometry resonance
        signature['phi_resonance'] = self._calculate_phi_resonance(molecules)
        signature['pi_resonance'] = self._calculate_pi_resonance(molecules)
        
        return signature
    
    def _calculate_phi_resonance(self, molecules: List[UBPMolecularState]) -> float:
        """Calculate phi (golden ratio) resonance in molecular ensemble"""
        phi = self.constants.CRV_GOLDEN_RATIO
        
        resonance_scores = []
        for mol in molecules:
            # Check for phi ratios in OffBit values
            values = [offbit.value for offbit in mol.offbits]
            if len(values) < 2:
                continue
            
            ratios = []
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    if values[j] != 0:
                        ratio = values[i] / values[j]
                        ratios.append(ratio)
            
            if ratios:
                # Find closest ratio to phi
                closest_to_phi = min(ratios, key=lambda x: abs(x - phi))
                resonance = np.exp(-abs(closest_to_phi - phi))
                resonance_scores.append(resonance)
        
        return np.mean(resonance_scores) if resonance_scores else 0.0
    
    def _calculate_pi_resonance(self, molecules: List[UBPMolecularState]) -> float:
        """Calculate pi resonance in molecular ensemble"""
        pi = self.constants.CRV_ELECTROMAGNETIC
        
        resonance_scores = []
        for mol in molecules:
            # Check for pi-related frequencies
            frequencies = [offbit.resonance_frequency for offbit in mol.offbits]
            
            for freq in frequencies:
                if freq > 0:
                    # Normalize frequency and check for pi resonance
                    normalized_freq = freq / 100.0  # Scale down
                    resonance = np.exp(-abs(normalized_freq - pi) / pi)
                    resonance_scores.append(resonance)
        
        return np.mean(resonance_scores) if resonance_scores else 0.0
    
    def _describe_molecular_pattern(self, molecules: List[UBPMolecularState]) -> str:
        """Generate description of molecular pattern"""
        if len(molecules) == 0:
            return "No molecular pattern identified"
        
        # Analyze realm distribution
        all_realms = set()
        for mol in molecules:
            all_realms.update(mol.realm_distribution.keys())
        
        dominant_realm = max(all_realms, 
                           key=lambda r: np.mean([mol.realm_distribution.get(r, 0) 
                                                 for mol in molecules]))
        
        # Analyze NRCI characteristics
        mean_nrci = np.mean([mol.nrci_score for mol in molecules])
        
        # Analyze energy characteristics
        mean_energy = np.mean([mol.energy_state for mol in molecules])
        
        pattern = f"UBP molecular pattern with dominant {dominant_realm} realm characteristics. "
        pattern += f"Average NRCI: {mean_nrci:.4f}, "
        pattern += f"Average UBP energy: {mean_energy:.2e}. "
        
        if mean_nrci > 0.8:
            pattern += "High coherence pattern suggesting strong geometric organization."
        elif mean_nrci > 0.5:
            pattern += "Moderate coherence pattern with emerging geometric structure."
        else:
            pattern += "Low coherence pattern requiring further optimization."
        
        return pattern
    
    def _validate_hypothesis_nrci(self, molecules: List[UBPMolecularState]) -> float:
        """Validate hypothesis using NRCI criteria"""
        if not molecules:
            return 0.0
        
        # Calculate ensemble NRCI
        individual_nrcis = [mol.nrci_score for mol in molecules]
        ensemble_nrci = np.mean(individual_nrcis)
        
        # Check coherence matrix consistency
        coherence_scores = []
        for mol in molecules:
            if mol.coherence_matrix.size > 0:
                mean_coherence = np.mean(mol.coherence_matrix)
                coherence_scores.append(mean_coherence)
        
        coherence_consistency = np.mean(coherence_scores) if coherence_scores else 0.0
        
        # Combined validation score
        validation_score = (ensemble_nrci + coherence_consistency) / 2.0
        
        return validation_score
    
    def _generate_supporting_evidence(self, molecules: List[UBPMolecularState], 
                                    signature: Dict[str, float]) -> List[str]:
        """Generate supporting evidence for hypothesis"""
        evidence = []
        
        # NRCI evidence
        if signature.get('mean_nrci', 0) > 0.7:
            evidence.append(f"High NRCI score ({signature['mean_nrci']:.4f}) indicates strong geometric coherence")
        
        # Energy evidence
        if signature.get('mean_energy', 0) > 1e6:
            evidence.append(f"Elevated UBP energy ({signature['mean_energy']:.2e}) suggests active molecular state")
        
        # Phase coherence evidence
        if signature.get('phase_coherence', 0) > 0.8:
            evidence.append(f"High temporal phase coherence ({signature['phase_coherence']:.4f}) indicates synchronized dynamics")
        
        # Sacred geometry evidence
        if signature.get('phi_resonance', 0) > 0.5:
            evidence.append(f"Golden ratio resonance ({signature['phi_resonance']:.4f}) suggests natural geometric optimization")
        
        if signature.get('pi_resonance', 0) > 0.5:
            evidence.append(f"Pi resonance ({signature['pi_resonance']:.4f}) indicates electromagnetic field organization")
        
        # Realm distribution evidence
        for realm in ['biological', 'quantum', 'electromagnetic']:
            realm_key = f'{realm}_distribution'
            if signature.get(realm_key, 0) > 0.3:
                evidence.append(f"Strong {realm} realm presence ({signature[realm_key]:.3f}) supports {realm} activity")
        
        if not evidence:
            evidence.append("Molecular pattern shows emerging UBP characteristics requiring further investigation")
        
        return evidence
    
    def _generate_testable_predictions(self, signature: Dict[str, float], 
                                     predicted_activity: float) -> List[str]:
        """Generate testable predictions from hypothesis"""
        predictions = []
        
        # Activity prediction
        predictions.append(f"Predicted biological activity: {predicted_activity:.4f} (pKi units)")
        
        # NRCI-based predictions
        if signature.get('mean_nrci', 0) > 0.8:
            predictions.append("Molecules with similar NRCI patterns should show comparable activity")
        
        # Energy-based predictions
        energy_range = signature.get('energy_range', 0)
        if energy_range > 1e5:
            predictions.append(f"Energy range ({energy_range:.2e}) suggests multiple conformational states")
        
        # Geometric predictions
        if signature.get('phi_resonance', 0) > 0.6:
            predictions.append("Structural modifications maintaining golden ratio proportions should preserve activity")
        
        if signature.get('pi_resonance', 0) > 0.6:
            predictions.append("Electromagnetic field interactions should follow pi-resonance patterns")
        
        # Realm-specific predictions
        if signature.get('biological_distribution', 0) > 0.4:
            predictions.append("Primary activity mechanism operates in biological realm (10 Hz frequency)")
        
        if signature.get('quantum_distribution', 0) > 0.3:
            predictions.append("Quantum effects contribute to molecular recognition (e/12 frequency)")
        
        # Temporal predictions
        if signature.get('phase_coherence', 0) > 0.7:
            predictions.append("Temporal dynamics should maintain phase coherence across measurement timescales")
        
        return predictions
    
    def validate_hypotheses_against_baseline(self) -> Dict:
        """Validate UBP hypotheses against baseline study results"""
        print("Validating UBP hypotheses against baseline results...")
        
        if not self.generated_hypotheses:
            return {'error': 'No hypotheses generated'}
        
        validation_results = {
            'n_hypotheses': len(self.generated_hypotheses),
            'mean_confidence': np.mean([h.confidence_score for h in self.generated_hypotheses]),
            'mean_nrci_validation': np.mean([h.nrci_validation for h in self.generated_hypotheses]),
            'baseline_comparison': {},
            'ubp_advantages': []
        }
        
        # Compare with baseline best performance
        baseline_best_r2 = self.modeling_results['test_r2'].max()
        ubp_mean_confidence = validation_results['mean_confidence']
        
        validation_results['baseline_comparison'] = {
            'baseline_best_r2': baseline_best_r2,
            'ubp_mean_confidence': ubp_mean_confidence,
            'improvement_ratio': ubp_mean_confidence / max(baseline_best_r2, 0.001)
        }
        
        # Identify UBP advantages
        if validation_results['mean_nrci_validation'] > 0.5:
            validation_results['ubp_advantages'].append("High NRCI validation scores indicate superior geometric coherence")
        
        if ubp_mean_confidence > baseline_best_r2:
            validation_results['ubp_advantages'].append("UBP confidence scores exceed baseline R² performance")
        
        # Analyze hypothesis diversity
        realm_diversity = set()
        for hypothesis in self.generated_hypotheses:
            if hypothesis.ubp_encoding:
                realm_diversity.update(hypothesis.ubp_encoding.realm_distribution.keys())
        
        validation_results['realm_diversity'] = len(realm_diversity)
        
        if len(realm_diversity) >= 4:
            validation_results['ubp_advantages'].append("Multi-realm analysis provides broader perspective than baseline")
        
        self.validation_results = validation_results
        
        return validation_results
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations of UBP hypotheses"""
        print("Generating UBP hypothesis visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('UBP-Enhanced Geometric Hypothesis Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: NRCI Distribution
        self.plot_nrci_distribution(axes[0, 0])
        
        # Plot 2: Energy vs Activity
        self.plot_energy_vs_activity(axes[0, 1])
        
        # Plot 3: Realm Distribution
        self.plot_realm_distribution(axes[0, 2])
        
        # Plot 4: Sacred Geometry Resonance
        self.plot_sacred_geometry_resonance(axes[1, 0])
        
        # Plot 5: Hypothesis Confidence
        self.plot_hypothesis_confidence(axes[1, 1])
        
        # Plot 6: UBP vs Baseline Comparison
        self.plot_ubp_baseline_comparison(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('v2_ubp_geometric_hypotheses.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("UBP hypothesis visualizations saved to v2_ubp_geometric_hypotheses.png")
    
    def plot_nrci_distribution(self, ax):
        """Plot NRCI score distribution"""
        if hasattr(self, 'ubp_molecular_states') and self.ubp_molecular_states:
            nrci_scores = [state.nrci_score for state in self.ubp_molecular_states]
            ax.hist(nrci_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(self.constants.NRCI_TARGET, color='red', linestyle='--', 
                      label=f'Target ({self.constants.NRCI_TARGET})')
            ax.set_xlabel('NRCI Score')
            ax.set_ylabel('Frequency')
            ax.set_title('UBP NRCI Distribution')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No NRCI data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('NRCI Distribution')
    
    def plot_energy_vs_activity(self, ax):
        """Plot UBP energy vs biological activity"""
        if hasattr(self, 'ubp_molecular_states') and self.ubp_molecular_states:
            energies = [state.energy_state for state in self.ubp_molecular_states]
            activities = self.target
            
            ax.scatter(energies, activities, alpha=0.6, s=20)
            ax.set_xlabel('UBP Energy')
            ax.set_ylabel('Biological Activity (pKi)')
            ax.set_title('UBP Energy vs Activity')
            
            # Add correlation
            if len(energies) > 1:
                corr, _ = pearsonr(energies, activities)
                ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No energy data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Energy vs Activity')
    
    def plot_realm_distribution(self, ax):
        """Plot realm distribution across molecules"""
        if hasattr(self, 'ubp_molecular_states') and self.ubp_molecular_states:
            all_realms = set()
            for state in self.ubp_molecular_states:
                all_realms.update(state.realm_distribution.keys())
            
            realm_means = {}
            for realm in all_realms:
                values = [state.realm_distribution.get(realm, 0) for state in self.ubp_molecular_states]
                realm_means[realm] = np.mean(values)
            
            if realm_means:
                realms = list(realm_means.keys())
                means = list(realm_means.values())
                
                bars = ax.bar(realms, means, alpha=0.7, color='lightgreen')
                ax.set_xlabel('UBP Realm')
                ax.set_ylabel('Average Distribution')
                ax.set_title('Realm Distribution')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.text(0.5, 0.5, 'No realm data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No realm data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Realm Distribution')
    
    def plot_sacred_geometry_resonance(self, ax):
        """Plot sacred geometry resonance patterns"""
        if self.generated_hypotheses:
            phi_resonances = [h.geometric_signature.get('phi_resonance', 0) for h in self.generated_hypotheses]
            pi_resonances = [h.geometric_signature.get('pi_resonance', 0) for h in self.generated_hypotheses]
            
            ax.scatter(phi_resonances, pi_resonances, alpha=0.7, s=50)
            ax.set_xlabel('Phi (φ) Resonance')
            ax.set_ylabel('Pi (π) Resonance')
            ax.set_title('Sacred Geometry Resonance')
            
            # Add quadrant lines
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'No hypothesis data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sacred Geometry Resonance')
    
    def plot_hypothesis_confidence(self, ax):
        """Plot hypothesis confidence scores"""
        if self.generated_hypotheses:
            confidences = [h.confidence_score for h in self.generated_hypotheses]
            nrci_validations = [h.nrci_validation for h in self.generated_hypotheses]
            
            ax.scatter(confidences, nrci_validations, alpha=0.7, s=50, color='orange')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('NRCI Validation')
            ax.set_title('Hypothesis Quality')
            
            # Add target lines
            ax.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='High NRCI')
            ax.axvline(0.8, color='blue', linestyle='--', alpha=0.5, label='High Confidence')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No hypothesis data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Hypothesis Quality')
    
    def plot_ubp_baseline_comparison(self, ax):
        """Plot UBP vs baseline comparison"""
        if self.validation_results:
            comparison = self.validation_results.get('baseline_comparison', {})
            
            categories = ['Baseline R²', 'UBP Confidence', 'UBP NRCI']
            values = [
                comparison.get('baseline_best_r2', 0),
                comparison.get('ubp_mean_confidence', 0),
                self.validation_results.get('mean_nrci_validation', 0)
            ]
            colors = ['lightcoral', 'lightblue', 'lightgreen']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_ylabel('Score')
            ax.set_title('UBP vs Baseline Performance')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No comparison data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('UBP vs Baseline')
    
    def save_hypotheses(self, filename="v2_ubp_geometric_hypotheses.json"):
        """Save generated hypotheses to file"""
        print(f"Saving UBP hypotheses to {filename}")
        
        # Convert hypotheses to serializable format
        hypotheses_data = []
        
        for hypothesis in self.generated_hypotheses:
            hypothesis_dict = {
                'hypothesis_id': hypothesis.hypothesis_id,
                'molecular_pattern': hypothesis.molecular_pattern,
                'geometric_signature': hypothesis.geometric_signature,
                'predicted_activity': hypothesis.predicted_activity,
                'confidence_score': hypothesis.confidence_score,
                'nrci_validation': hypothesis.nrci_validation,
                'supporting_evidence': hypothesis.supporting_evidence,
                'testable_predictions': hypothesis.testable_predictions
            }
            
            # Add UBP encoding summary
            if hypothesis.ubp_encoding:
                hypothesis_dict['ubp_encoding_summary'] = {
                    'n_offbits': len(hypothesis.ubp_encoding.offbits),
                    'nrci_score': hypothesis.ubp_encoding.nrci_score,
                    'energy_state': hypothesis.ubp_encoding.energy_state,
                    'temporal_phase': hypothesis.ubp_encoding.temporal_phase,
                    'realm_distribution': hypothesis.ubp_encoding.realm_distribution
                }
            
            hypotheses_data.append(hypothesis_dict)
        
        # Save with validation results
        output_data = {
            'hypotheses': hypotheses_data,
            'validation_results': self.validation_results,
            'generation_timestamp': time.time(),
            'ubp_version': '3.2+',
            'study_version': 'v2'
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"Saved {len(hypotheses_data)} UBP hypotheses")
        
        return output_data

def main():
    """Main execution function"""
    print("="*80)
    print("VERSION 2 UBP-ENHANCED GEOMETRIC HYPOTHESIS GENERATOR")
    print("="*80)
    
    # Initialize UBP hypothesis generator
    generator = UBPGeometricHypothesisGenerator()
    
    # Load baseline data
    dataset = generator.load_baseline_data()
    
    # Encode molecules as UBP states
    ubp_states = generator.encode_molecules_as_ubp_states()
    
    # Generate geometric hypotheses
    hypotheses = generator.generate_geometric_hypotheses(n_hypotheses=15)
    
    # Validate against baseline
    validation_results = generator.validate_hypotheses_against_baseline()
    
    # Generate visualizations
    generator.generate_visualizations()
    
    # Save hypotheses
    saved_data = generator.save_hypotheses()
    
    print("\n" + "="*80)
    print("UBP GEOMETRIC HYPOTHESIS GENERATION SUMMARY")
    print("="*80)
    print(f"Dataset: {len(dataset)} compounds")
    print(f"UBP molecular states: {len(ubp_states)}")
    print(f"Generated hypotheses: {len(hypotheses)}")
    print(f"Mean confidence: {validation_results.get('mean_confidence', 0):.4f}")
    print(f"Mean NRCI validation: {validation_results.get('mean_nrci_validation', 0):.4f}")
    print(f"Realm diversity: {validation_results.get('realm_diversity', 0)} realms")
    
    if validation_results.get('ubp_advantages'):
        print("\nUBP Advantages:")
        for advantage in validation_results['ubp_advantages']:
            print(f"  • {advantage}")
    
    print("\nFiles generated:")
    print("- v2_ubp_geometric_hypotheses.png")
    print("- v2_ubp_geometric_hypotheses.json")
    print("\nUBP-enhanced analysis complete!")
    
    return generator, hypotheses, validation_results

if __name__ == "__main__":
    generator, hypotheses, validation = main()
