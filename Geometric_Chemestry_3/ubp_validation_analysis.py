#!/usr/bin/env python3
"""
UBP Principles Validation Analysis
Phase 5: Validate UBP principles against materials properties

This system validates:
1. NRCI target achievement (≥ 0.999999)
2. Cross-realm coherence (C_ij ≥ 0.95)
3. Fractal dimension validation (D ≈ 2.3)
4. Sacred geometry resonance validation
5. UBP energy equation validation
6. Temporal coherence synchronization
7. GLR framework validation
8. TGIC constraint validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import r2_score, mean_squared_error
import json
import warnings
warnings.filterwarnings('ignore')

class UBPValidationAnalyzer:
    """Comprehensive UBP principles validation system"""
    
    def __init__(self):
        # UBP validation targets
        self.validation_targets = {
            'nrci_target': 0.999999,
            'coherence_target': 0.95,
            'fractal_dimension_target': 2.3,
            'fractal_tolerance': 0.3,
            'resonance_significance': 0.5,
            'energy_correlation_threshold': 0.7,
            'temporal_sync_threshold': 0.95
        }
        
        # UBP Core Resonance Values for validation
        self.crv_values = {
            'quantum': {'freq': 4.58e14, 'wavelength': 655e-9, 'toggle_bias': np.e/12},
            'electromagnetic': {'freq': 3.141593, 'wavelength': 635e-9, 'toggle_bias': np.pi},
            'gravitational': {'freq': 100, 'wavelength': 1000e-9, 'toggle_bias': 0.1},
            'biological': {'freq': 10, 'wavelength': 700e-9, 'toggle_bias': 0.01},
            'cosmological': {'freq': 1e-11, 'wavelength': 800e-9, 'toggle_bias': np.pi**((1+np.sqrt(5))/2)},
            'nuclear': {'freq': 1.2356e20, 'wavelength': 600e-9, 'toggle_bias': 0.5},
            'optical': {'freq': 5e14, 'wavelength': 600e-9, 'toggle_bias': 0.3}
        }
        
        # Sacred geometry constants for validation
        self.sacred_constants = {
            'phi': (1 + np.sqrt(5)) / 2,
            'pi': np.pi,
            'e': np.e,
            'sqrt2': np.sqrt(2),
            'sqrt3': np.sqrt(3),
            'sqrt5': np.sqrt(5)
        }
        
        # Validation results storage
        self.validation_results = {}
        
    def perform_comprehensive_validation(self, ubp_data_file: str, geometric_results_file: str):
        """Perform comprehensive UBP principles validation"""
        
        print("="*80)
        print("UBP PRINCIPLES VALIDATION ANALYSIS")
        print("="*80)
        print("Phase 5: Validating UBP Framework Against Materials Properties")
        print()
        
        # Load data
        print("Loading UBP-encoded data and geometric analysis results...")
        try:
            df = pd.read_csv(ubp_data_file)
            print(f"✅ Loaded {len(df)} UBP-encoded materials")
            
            with open(geometric_results_file, 'r') as f:
                geometric_results = json.load(f)
            print(f"✅ Loaded geometric analysis results")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        # Validation 1: NRCI Target Achievement
        print("\\nValidation 1: NRCI Target Achievement...")
        self._validate_nrci_targets(df)
        
        # Validation 2: Cross-Realm Coherence
        print("\\nValidation 2: Cross-Realm Coherence...")
        self._validate_cross_realm_coherence(df)
        
        # Validation 3: Fractal Dimension Validation
        print("\\nValidation 3: Fractal Dimension Validation...")
        self._validate_fractal_dimensions(geometric_results)
        
        # Validation 4: Sacred Geometry Resonance
        print("\\nValidation 4: Sacred Geometry Resonance...")
        self._validate_sacred_geometry_resonances(df)
        
        # Validation 5: UBP Energy Equation
        print("\\nValidation 5: UBP Energy Equation...")
        self._validate_ubp_energy_equation(df)
        
        # Validation 6: Temporal Coherence Synchronization
        print("\\nValidation 6: Temporal Coherence Synchronization...")
        self._validate_temporal_coherence(df)
        
        # Validation 7: GLR Framework Validation
        print("\\nValidation 7: GLR Framework Validation...")
        self._validate_glr_frameworks(df)
        
        # Validation 8: TGIC Constraint Validation
        print("\\nValidation 8: TGIC Constraint Validation...")
        self._validate_tgic_constraints(df)
        
        # Validation 9: Materials Property Correlations
        print("\\nValidation 9: Materials Property Correlations...")
        self._validate_materials_correlations(df)
        
        # Overall UBP System Validation
        print("\\nOverall UBP System Validation...")
        self._calculate_overall_validation_score()
        
        print("\\n✅ Comprehensive UBP validation complete!")
        
        return df, self.validation_results
    
    def _validate_nrci_targets(self, df):
        """Validate NRCI target achievement"""
        
        try:
            if 'nrci_calculated' not in df.columns:
                print("  ❌ NRCI data not available")
                self.validation_results['nrci_validation'] = {'status': 'failed', 'reason': 'no_data'}
                return
            
            nrci_values = df['nrci_calculated'].values
            target = self.validation_targets['nrci_target']
            
            # Calculate NRCI statistics
            nrci_stats = {
                'mean': float(np.mean(nrci_values)),
                'std': float(np.std(nrci_values)),
                'min': float(np.min(nrci_values)),
                'max': float(np.max(nrci_values)),
                'median': float(np.median(nrci_values))
            }
            
            # Count materials achieving target
            target_achieved = np.sum(nrci_values >= target)
            target_percentage = 100 * target_achieved / len(nrci_values)
            
            # Statistical significance test
            # Test if mean NRCI is significantly above random (0.5)
            t_stat, p_value = stats.ttest_1samp(nrci_values, 0.5)
            
            # Distribution analysis
            # Check if NRCI follows expected distribution
            normality_stat, normality_p = stats.normaltest(nrci_values)
            
            nrci_validation = {
                'status': 'passed' if target_percentage > 10 else 'failed',  # At least 10% should achieve target
                'statistics': nrci_stats,
                'target_achievement': {
                    'target': target,
                    'count_achieved': int(target_achieved),
                    'percentage_achieved': float(target_percentage),
                    'threshold_met': target_percentage > 10
                },
                'statistical_tests': {
                    'mean_vs_random': {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.01
                    },
                    'normality': {
                        'statistic': float(normality_stat),
                        'p_value': float(normality_p),
                        'is_normal': normality_p > 0.05
                    }
                }
            }
            
            self.validation_results['nrci_validation'] = nrci_validation
            
            print(f"  NRCI Statistics:")
            print(f"    Mean: {nrci_stats['mean']:.6f}")
            print(f"    Target (≥{target}): {target_achieved}/{len(nrci_values)} ({target_percentage:.1f}%)")
            print(f"    Statistical significance: p = {p_value:.2e}")
            
            status = "✅ PASSED" if nrci_validation['status'] == 'passed' else "❌ FAILED"
            print(f"  NRCI Validation: {status}")
            
        except Exception as e:
            print(f"  ❌ Error in NRCI validation: {e}")
            self.validation_results['nrci_validation'] = {'status': 'error', 'error': str(e)}
    
    def _validate_cross_realm_coherence(self, df):
        """Validate cross-realm coherence targets"""
        
        try:
            # Find realm coherence columns
            realm_coherence_cols = [col for col in df.columns if col.endswith('_coherence') and 
                                  any(realm in col for realm in self.crv_values.keys())]
            
            if len(realm_coherence_cols) < 2:
                print("  ❌ Insufficient realm coherence data")
                self.validation_results['coherence_validation'] = {'status': 'failed', 'reason': 'insufficient_data'}
                return
            
            # Calculate cross-realm coherence matrix
            coherence_data = df[realm_coherence_cols].values
            correlation_matrix = np.corrcoef(coherence_data.T)
            
            # Extract upper triangular correlations (excluding diagonal)
            upper_tri_indices = np.triu_indices_from(correlation_matrix, k=1)
            cross_correlations = correlation_matrix[upper_tri_indices]
            
            # Remove NaN values
            cross_correlations = cross_correlations[~np.isnan(cross_correlations)]
            
            if len(cross_correlations) == 0:
                print("  ❌ No valid cross-correlations found")
                self.validation_results['coherence_validation'] = {'status': 'failed', 'reason': 'no_valid_correlations'}
                return
            
            target = self.validation_targets['coherence_target']
            
            # Calculate coherence statistics
            coherence_stats = {
                'mean_correlation': float(np.mean(cross_correlations)),
                'std_correlation': float(np.std(cross_correlations)),
                'min_correlation': float(np.min(cross_correlations)),
                'max_correlation': float(np.max(cross_correlations)),
                'median_correlation': float(np.median(cross_correlations))
            }
            
            # Count correlations achieving target
            target_achieved = np.sum(cross_correlations >= target)
            target_percentage = 100 * target_achieved / len(cross_correlations)
            
            # Identify high-coherence realm pairs
            high_coherence_pairs = []
            realm_names = [col.replace('_coherence', '') for col in realm_coherence_cols]
            
            for i in range(len(realm_names)):
                for j in range(i+1, len(realm_names)):
                    correlation = correlation_matrix[i, j]
                    if not np.isnan(correlation) and correlation >= target:
                        high_coherence_pairs.append({
                            'realm1': realm_names[i],
                            'realm2': realm_names[j],
                            'correlation': float(correlation)
                        })
            
            coherence_validation = {
                'status': 'passed' if target_percentage > 20 else 'failed',  # At least 20% should achieve target
                'statistics': coherence_stats,
                'target_achievement': {
                    'target': target,
                    'count_achieved': int(target_achieved),
                    'total_pairs': len(cross_correlations),
                    'percentage_achieved': float(target_percentage),
                    'threshold_met': target_percentage > 20
                },
                'high_coherence_pairs': high_coherence_pairs,
                'correlation_matrix': correlation_matrix.tolist(),
                'realm_names': realm_names
            }
            
            self.validation_results['coherence_validation'] = coherence_validation
            
            print(f"  Cross-Realm Coherence:")
            print(f"    Mean correlation: {coherence_stats['mean_correlation']:.3f}")
            print(f"    Target (≥{target}): {target_achieved}/{len(cross_correlations)} pairs ({target_percentage:.1f}%)")
            print(f"    High-coherence pairs: {len(high_coherence_pairs)}")
            
            status = "✅ PASSED" if coherence_validation['status'] == 'passed' else "❌ FAILED"
            print(f"  Coherence Validation: {status}")
            
        except Exception as e:
            print(f"  ❌ Error in coherence validation: {e}")
            self.validation_results['coherence_validation'] = {'status': 'error', 'error': str(e)}
    
    def _validate_fractal_dimensions(self, geometric_results):
        """Validate fractal dimension targets"""
        
        try:
            if 'fractal_analysis' not in geometric_results.get('resonance_patterns', {}):
                print("  ❌ Fractal dimension data not available")
                self.validation_results['fractal_validation'] = {'status': 'failed', 'reason': 'no_data'}
                return
            
            fractal_data = geometric_results['resonance_patterns']['fractal_analysis']
            target = self.validation_targets['fractal_dimension_target']
            tolerance = self.validation_targets['fractal_tolerance']
            
            # Analyze fractal dimensions
            fractal_dimensions = []
            embedding_results = {}
            
            for embedding_name, analysis in fractal_data.items():
                if 'fractal_dimension' in analysis:
                    fractal_dim = analysis['fractal_dimension']
                    fractal_dimensions.append(fractal_dim)
                    
                    # Check if within target range
                    within_target = abs(fractal_dim - target) <= tolerance
                    
                    embedding_results[embedding_name] = {
                        'fractal_dimension': fractal_dim,
                        'target_deviation': abs(fractal_dim - target),
                        'within_target': within_target,
                        'r_squared': analysis.get('r_squared', 0.0)
                    }
            
            if len(fractal_dimensions) == 0:
                print("  ❌ No valid fractal dimensions found")
                self.validation_results['fractal_validation'] = {'status': 'failed', 'reason': 'no_valid_dimensions'}
                return
            
            # Calculate statistics
            fractal_stats = {
                'mean': float(np.mean(fractal_dimensions)),
                'std': float(np.std(fractal_dimensions)),
                'min': float(np.min(fractal_dimensions)),
                'max': float(np.max(fractal_dimensions)),
                'target': target,
                'tolerance': tolerance
            }
            
            # Count embeddings within target
            within_target_count = sum(1 for result in embedding_results.values() if result['within_target'])
            target_percentage = 100 * within_target_count / len(embedding_results)
            
            fractal_validation = {
                'status': 'passed' if within_target_count > 0 else 'failed',
                'statistics': fractal_stats,
                'embedding_results': embedding_results,
                'target_achievement': {
                    'target_range': [target - tolerance, target + tolerance],
                    'count_within_target': within_target_count,
                    'total_embeddings': len(embedding_results),
                    'percentage_within_target': float(target_percentage)
                }
            }
            
            self.validation_results['fractal_validation'] = fractal_validation
            
            print(f"  Fractal Dimensions:")
            print(f"    Mean: {fractal_stats['mean']:.3f}")
            print(f"    Target: {target} ± {tolerance}")
            print(f"    Within target: {within_target_count}/{len(embedding_results)} embeddings ({target_percentage:.1f}%)")
            
            status = "✅ PASSED" if fractal_validation['status'] == 'passed' else "❌ FAILED"
            print(f"  Fractal Validation: {status}")
            
        except Exception as e:
            print(f"  ❌ Error in fractal validation: {e}")
            self.validation_results['fractal_validation'] = {'status': 'error', 'error': str(e)}
    
    def _validate_sacred_geometry_resonances(self, df):
        """Validate sacred geometry resonance patterns"""
        
        try:
            # Find resonance columns
            resonance_cols = [col for col in df.columns if 'resonance' in col and col != 'total_resonance_potential']
            
            if len(resonance_cols) == 0:
                print("  ❌ No resonance data available")
                self.validation_results['resonance_validation'] = {'status': 'failed', 'reason': 'no_data'}
                return
            
            threshold = self.validation_targets['resonance_significance']
            
            # Analyze each sacred constant
            sacred_analysis = {}
            
            for const_name, const_value in self.sacred_constants.items():
                const_resonance_cols = [col for col in resonance_cols if const_name in col]
                
                if const_resonance_cols:
                    # Combine all resonances for this constant
                    const_resonances = df[const_resonance_cols].values.flatten()
                    const_resonances = const_resonances[~np.isnan(const_resonances)]
                    
                    if len(const_resonances) > 0:
                        # Calculate statistics
                        stats_dict = {
                            'mean': float(np.mean(const_resonances)),
                            'std': float(np.std(const_resonances)),
                            'max': float(np.max(const_resonances)),
                            'significant_count': int(np.sum(const_resonances >= threshold)),
                            'total_count': len(const_resonances),
                            'significance_percentage': float(100 * np.sum(const_resonances >= threshold) / len(const_resonances))
                        }
                        
                        # Test for non-random distribution
                        random_mean = 0.1  # Expected mean for random resonances
                        t_stat, p_value = stats.ttest_1samp(const_resonances, random_mean)
                        
                        sacred_analysis[const_name] = {
                            'constant_value': const_value,
                            'statistics': stats_dict,
                            'statistical_test': {
                                't_statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                        }
            
            # Overall resonance validation
            if 'total_resonance_potential' in df.columns:
                total_resonances = df['total_resonance_potential'].values
                total_resonances = total_resonances[~np.isnan(total_resonances)]
                
                overall_stats = {
                    'mean': float(np.mean(total_resonances)),
                    'std': float(np.std(total_resonances)),
                    'max': float(np.max(total_resonances)),
                    'significant_count': int(np.sum(total_resonances >= threshold)),
                    'significance_percentage': float(100 * np.sum(total_resonances >= threshold) / len(total_resonances))
                }
            else:
                overall_stats = {'mean': 0.0, 'significance_percentage': 0.0}
            
            # Determine validation status
            significant_constants = sum(1 for analysis in sacred_analysis.values() 
                                      if analysis['statistical_test']['significant'])
            
            resonance_validation = {
                'status': 'passed' if significant_constants >= 2 else 'failed',  # At least 2 constants should show significance
                'sacred_geometry_analysis': sacred_analysis,
                'overall_statistics': overall_stats,
                'significant_constants_count': significant_constants,
                'total_constants_analyzed': len(sacred_analysis)
            }
            
            self.validation_results['resonance_validation'] = resonance_validation
            
            print(f"  Sacred Geometry Resonances:")
            print(f"    Constants analyzed: {len(sacred_analysis)}")
            print(f"    Statistically significant: {significant_constants}")
            print(f"    Overall resonance mean: {overall_stats['mean']:.3f}")
            print(f"    Materials with significant resonance: {overall_stats.get('significance_percentage', 0):.1f}%")
            
            status = "✅ PASSED" if resonance_validation['status'] == 'passed' else "❌ FAILED"
            print(f"  Resonance Validation: {status}")
            
        except Exception as e:
            print(f"  ❌ Error in resonance validation: {e}")
            self.validation_results['resonance_validation'] = {'status': 'error', 'error': str(e)}
    
    def _validate_ubp_energy_equation(self, df):
        """Validate UBP energy equation against materials properties"""
        
        try:
            if 'ubp_energy_full' not in df.columns:
                print("  ❌ UBP energy data not available")
                self.validation_results['energy_validation'] = {'status': 'failed', 'reason': 'no_data'}
                return
            
            energy_values = df['ubp_energy_full'].values
            
            # Test correlations with fundamental properties
            property_correlations = {}
            fundamental_properties = ['formation_energy_per_atom', 'band_gap', 'total_magnetization', 
                                    'density', 'volume_per_atom']
            
            for prop in fundamental_properties:
                if prop in df.columns:
                    prop_values = df[prop].values
                    
                    # Remove NaN values
                    valid_mask = ~(np.isnan(energy_values) | np.isnan(prop_values))
                    if np.sum(valid_mask) > 10:
                        correlation = np.corrcoef(energy_values[valid_mask], prop_values[valid_mask])[0, 1]
                        
                        # Statistical significance test
                        r_stat, p_value = stats.pearsonr(energy_values[valid_mask], prop_values[valid_mask])
                        
                        property_correlations[prop] = {
                            'correlation': float(correlation),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'sample_size': int(np.sum(valid_mask))
                        }
            
            # Test UBP energy vs NRCI correlation
            nrci_correlation = None
            if 'nrci_calculated' in df.columns:
                nrci_values = df['nrci_calculated'].values
                valid_mask = ~(np.isnan(energy_values) | np.isnan(nrci_values))
                if np.sum(valid_mask) > 10:
                    nrci_correlation = {
                        'correlation': float(np.corrcoef(energy_values[valid_mask], nrci_values[valid_mask])[0, 1]),
                        'sample_size': int(np.sum(valid_mask))
                    }
            
            # Energy distribution analysis
            energy_stats = {
                'mean': float(np.nanmean(energy_values)),
                'std': float(np.nanstd(energy_values)),
                'min': float(np.nanmin(energy_values)),
                'max': float(np.nanmax(energy_values)),
                'range': float(np.nanmax(energy_values) - np.nanmin(energy_values))
            }
            
            # Check for expected energy scaling
            # UBP energy should scale with system size and complexity
            if 'nsites' in df.columns:
                nsites_values = df['nsites'].values
                valid_mask = ~(np.isnan(energy_values) | np.isnan(nsites_values))
                if np.sum(valid_mask) > 10:
                    size_correlation = np.corrcoef(energy_values[valid_mask], nsites_values[valid_mask])[0, 1]
                    property_correlations['system_size'] = {
                        'correlation': float(size_correlation),
                        'property': 'nsites'
                    }
            
            # Determine validation status
            significant_correlations = sum(1 for corr in property_correlations.values() 
                                         if corr.get('significant', False))
            
            energy_validation = {
                'status': 'passed' if significant_correlations >= 2 else 'failed',
                'energy_statistics': energy_stats,
                'property_correlations': property_correlations,
                'nrci_correlation': nrci_correlation,
                'significant_correlations_count': significant_correlations,
                'total_correlations_tested': len(property_correlations)
            }
            
            self.validation_results['energy_validation'] = energy_validation
            
            print(f"  UBP Energy Equation:")
            print(f"    Energy range: {energy_stats['min']:.2e} to {energy_stats['max']:.2e}")
            print(f"    Significant correlations: {significant_correlations}/{len(property_correlations)}")
            
            if nrci_correlation:
                print(f"    Energy-NRCI correlation: {nrci_correlation['correlation']:.3f}")
            
            status = "✅ PASSED" if energy_validation['status'] == 'passed' else "❌ FAILED"
            print(f"  Energy Validation: {status}")
            
        except Exception as e:
            print(f"  ❌ Error in energy validation: {e}")
            self.validation_results['energy_validation'] = {'status': 'error', 'error': str(e)}
    
    def _validate_temporal_coherence(self, df):
        """Validate temporal coherence synchronization"""
        
        try:
            # Check for temporal coherence indicators
            temporal_indicators = ['system_coherence', 'cross_realm_coherence', 'toggle_coherence']
            available_indicators = [col for col in temporal_indicators if col in df.columns]
            
            if len(available_indicators) == 0:
                print("  ❌ No temporal coherence data available")
                self.validation_results['temporal_validation'] = {'status': 'failed', 'reason': 'no_data'}
                return
            
            # Calculate Coherent Synchronization Cycle (CSC) validation
            csc_period = 1 / np.pi  # ~0.318309886 s
            
            # Analyze coherence stability
            coherence_analysis = {}
            
            for indicator in available_indicators:
                values = df[indicator].values
                values = values[~np.isnan(values)]
                
                if len(values) > 10:
                    # Calculate coherence statistics
                    coherence_stats = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'stability': float(1.0 - np.std(values)),  # Higher stability = lower variance
                        'above_threshold': int(np.sum(values >= self.validation_targets['temporal_sync_threshold'])),
                        'threshold_percentage': float(100 * np.sum(values >= self.validation_targets['temporal_sync_threshold']) / len(values))
                    }
                    
                    coherence_analysis[indicator] = coherence_stats
            
            # Overall temporal validation
            if coherence_analysis:
                mean_stability = np.mean([analysis['stability'] for analysis in coherence_analysis.values()])
                mean_threshold_percentage = np.mean([analysis['threshold_percentage'] for analysis in coherence_analysis.values()])
                
                temporal_validation = {
                    'status': 'passed' if mean_stability > 0.8 else 'failed',
                    'csc_period': csc_period,
                    'coherence_analysis': coherence_analysis,
                    'overall_stability': float(mean_stability),
                    'overall_threshold_achievement': float(mean_threshold_percentage)
                }
            else:
                temporal_validation = {
                    'status': 'failed',
                    'reason': 'no_valid_coherence_data'
                }
            
            self.validation_results['temporal_validation'] = temporal_validation
            
            if 'overall_stability' in temporal_validation:
                print(f"  Temporal Coherence:")
                print(f"    CSC period: {csc_period:.6f} s")
                print(f"    Overall stability: {temporal_validation['overall_stability']:.3f}")
                print(f"    Threshold achievement: {temporal_validation['overall_threshold_achievement']:.1f}%")
                
                status = "✅ PASSED" if temporal_validation['status'] == 'passed' else "❌ FAILED"
                print(f"  Temporal Validation: {status}")
            else:
                print("  ❌ Temporal validation failed - no valid data")
            
        except Exception as e:
            print(f"  ❌ Error in temporal validation: {e}")
            self.validation_results['temporal_validation'] = {'status': 'error', 'error': str(e)}
    
    def _validate_glr_frameworks(self, df):
        """Validate GLR (Golay-Leech-Resonance) frameworks"""
        
        try:
            # Check for realm-specific encoding features
            realm_features = {}
            
            for realm, crv in self.crv_values.items():
                realm_cols = [col for col in df.columns if realm in col]
                if realm_cols:
                    realm_data = df[realm_cols].select_dtypes(include=[np.number])
                    
                    if len(realm_data.columns) > 0:
                        # Calculate realm coherence metrics
                        realm_values = realm_data.values.flatten()
                        realm_values = realm_values[~np.isnan(realm_values)]
                        
                        if len(realm_values) > 0:
                            realm_features[realm] = {
                                'feature_count': len(realm_data.columns),
                                'mean_value': float(np.mean(realm_values)),
                                'std_value': float(np.std(realm_values)),
                                'crv_frequency': crv['freq'],
                                'crv_wavelength': crv['wavelength'],
                                'crv_toggle_bias': crv['toggle_bias']
                            }
            
            # Validate GLR spatial and temporal metrics
            glr_validation = {}
            
            for realm, features in realm_features.items():
                # Expected GLR metrics from UBP specification
                expected_metrics = {
                    'electromagnetic': {'spatial': 74.96, 'temporal': 91.0, 'nrci': 1.0},
                    'quantum': {'spatial': 74.65, 'temporal': 43.3, 'nrci': 0.875},
                    'gravitational': {'spatial': 85.59, 'temporal': 108.1, 'nrci': 0.915},
                    'biological': {'spatial': 48.79, 'temporal': 97.3, 'nrci': 0.911},
                    'cosmological': {'spatial': 62.22, 'temporal': 102.2, 'nrci': 0.797}
                }
                
                if realm in expected_metrics:
                    expected = expected_metrics[realm]
                    
                    # Estimate spatial metric from feature distribution
                    estimated_spatial = min(100, features['mean_value'] * 100)
                    spatial_deviation = abs(estimated_spatial - expected['spatial'])
                    
                    glr_validation[realm] = {
                        'expected_metrics': expected,
                        'estimated_spatial': float(estimated_spatial),
                        'spatial_deviation': float(spatial_deviation),
                        'within_tolerance': spatial_deviation < 20,  # 20% tolerance
                        'feature_representation': features
                    }
            
            # Overall GLR validation
            valid_realms = sum(1 for validation in glr_validation.values() if validation['within_tolerance'])
            
            glr_framework_validation = {
                'status': 'passed' if valid_realms >= 2 else 'failed',
                'realm_validations': glr_validation,
                'valid_realms_count': valid_realms,
                'total_realms_tested': len(glr_validation)
            }
            
            self.validation_results['glr_validation'] = glr_framework_validation
            
            print(f"  GLR Frameworks:")
            print(f"    Realms analyzed: {len(glr_validation)}")
            print(f"    Within tolerance: {valid_realms}/{len(glr_validation)}")
            
            status = "✅ PASSED" if glr_framework_validation['status'] == 'passed' else "❌ FAILED"
            print(f"  GLR Validation: {status}")
            
        except Exception as e:
            print(f"  ❌ Error in GLR validation: {e}")
            self.validation_results['glr_validation'] = {'status': 'error', 'error': str(e)}
    
    def _validate_tgic_constraints(self, df):
        """Validate TGIC (Triad Graph Interaction Constraint) implementation"""
        
        try:
            # TGIC enforces 3, 6, 9 structure
            tgic_structure = {'axes': 3, 'faces': 6, 'interactions_per_offbit': 9}
            
            # Check for TGIC-related features
            tgic_indicators = ['total_interactions', 'active_offbits_count', 'interaction_density']
            available_indicators = [col for col in tgic_indicators if col in df.columns]
            
            if len(available_indicators) == 0:
                print("  ❌ No TGIC constraint data available")
                self.validation_results['tgic_validation'] = {'status': 'failed', 'reason': 'no_data'}
                return
            
            tgic_analysis = {}
            
            # Validate interaction constraints
            if 'total_interactions' in df.columns and 'active_offbits_count' in df.columns:
                total_interactions = df['total_interactions'].values
                active_offbits = df['active_offbits_count'].values
                
                valid_mask = ~(np.isnan(total_interactions) | np.isnan(active_offbits))
                if np.sum(valid_mask) > 0:
                    # Calculate expected interactions (9 per OffBit)
                    expected_interactions = active_offbits[valid_mask] * tgic_structure['interactions_per_offbit']
                    actual_interactions = total_interactions[valid_mask]
                    
                    # Calculate deviation from TGIC constraint
                    interaction_ratios = actual_interactions / expected_interactions
                    interaction_ratios = interaction_ratios[~np.isnan(interaction_ratios)]
                    
                    if len(interaction_ratios) > 0:
                        tgic_analysis['interaction_constraint'] = {
                            'mean_ratio': float(np.mean(interaction_ratios)),
                            'std_ratio': float(np.std(interaction_ratios)),
                            'within_tolerance': float(np.sum(np.abs(interaction_ratios - 1.0) < 0.1) / len(interaction_ratios)),
                            'expected_interactions_per_offbit': tgic_structure['interactions_per_offbit']
                        }
            
            # Validate geometric structure (3, 6, 9)
            geometric_validation = {}
            
            # Check for 3-fold symmetry in coordination
            if 'coordination_number' in df.columns:
                coord_numbers = df['coordination_number'].values
                coord_numbers = coord_numbers[~np.isnan(coord_numbers)]
                
                if len(coord_numbers) > 0:
                    # Check for multiples of 3
                    multiples_of_3 = np.sum(coord_numbers % 3 == 0) / len(coord_numbers)
                    geometric_validation['coordination_3fold'] = float(multiples_of_3)
            
            # Check for 6-fold patterns in crystal systems
            if 'crystal_system_code' in df.columns:
                crystal_codes = df['crystal_system_code'].values
                crystal_codes = crystal_codes[~np.isnan(crystal_codes)]
                
                if len(crystal_codes) > 0:
                    # Hexagonal system (code 6) represents 6-fold symmetry
                    hexagonal_fraction = np.sum(crystal_codes == 6) / len(crystal_codes)
                    geometric_validation['hexagonal_6fold'] = float(hexagonal_fraction)
            
            # Overall TGIC validation
            tgic_constraints_met = 0
            total_constraints = 0
            
            if 'interaction_constraint' in tgic_analysis:
                total_constraints += 1
                if tgic_analysis['interaction_constraint']['within_tolerance'] > 0.5:
                    tgic_constraints_met += 1
            
            if geometric_validation:
                for validation_key, value in geometric_validation.items():
                    total_constraints += 1
                    if value > 0.1:  # At least 10% should show the pattern
                        tgic_constraints_met += 1
            
            tgic_validation = {
                'status': 'passed' if tgic_constraints_met >= 1 else 'failed',
                'tgic_structure': tgic_structure,
                'interaction_analysis': tgic_analysis,
                'geometric_validation': geometric_validation,
                'constraints_met': tgic_constraints_met,
                'total_constraints': total_constraints
            }
            
            self.validation_results['tgic_validation'] = tgic_validation
            
            print(f"  TGIC Constraints:")
            print(f"    Structure: {tgic_structure['axes']} axes, {tgic_structure['faces']} faces, {tgic_structure['interactions_per_offbit']} interactions")
            print(f"    Constraints met: {tgic_constraints_met}/{total_constraints}")
            
            status = "✅ PASSED" if tgic_validation['status'] == 'passed' else "❌ FAILED"
            print(f"  TGIC Validation: {status}")
            
        except Exception as e:
            print(f"  ❌ Error in TGIC validation: {e}")
            self.validation_results['tgic_validation'] = {'status': 'error', 'error': str(e)}
    
    def _validate_materials_correlations(self, df):
        """Validate UBP predictions against known materials properties"""
        
        try:
            # Test UBP quality score vs materials properties
            if 'ubp_quality_score' not in df.columns:
                print("  ❌ UBP quality score not available")
                self.validation_results['materials_validation'] = {'status': 'failed', 'reason': 'no_quality_score'}
                return
            
            quality_scores = df['ubp_quality_score'].values
            
            # Test correlations with key materials properties
            property_tests = {
                'formation_energy': 'formation_energy_per_atom',
                'band_gap': 'band_gap',
                'magnetization': 'total_magnetization',
                'density': 'density',
                'stability': 'energy_above_hull' if 'energy_above_hull' in df.columns else None
            }
            
            correlation_results = {}
            
            for test_name, prop_col in property_tests.items():
                if prop_col and prop_col in df.columns:
                    prop_values = df[prop_col].values
                    
                    valid_mask = ~(np.isnan(quality_scores) | np.isnan(prop_values))
                    if np.sum(valid_mask) > 10:
                        correlation = np.corrcoef(quality_scores[valid_mask], prop_values[valid_mask])[0, 1]
                        
                        # Statistical significance
                        r_stat, p_value = stats.pearsonr(quality_scores[valid_mask], prop_values[valid_mask])
                        
                        correlation_results[test_name] = {
                            'correlation': float(correlation),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'sample_size': int(np.sum(valid_mask)),
                            'property_column': prop_col
                        }
            
            # Test UBP realm assignments vs materials properties
            realm_validation = {}
            
            if 'primary_realm' in df.columns:
                for realm in df['primary_realm'].unique():
                    realm_mask = df['primary_realm'] == realm
                    realm_materials = df[realm_mask]
                    
                    if len(realm_materials) > 5:
                        # Analyze characteristic properties for this realm
                        realm_properties = {}
                        
                        if realm == 'quantum' and 'total_magnetization' in df.columns:
                            mag_values = realm_materials['total_magnetization'].values
                            mag_values = mag_values[~np.isnan(mag_values)]
                            if len(mag_values) > 0:
                                realm_properties['magnetization_mean'] = float(np.mean(mag_values))
                                realm_properties['magnetic_materials'] = float(np.sum(mag_values > 0.1) / len(mag_values))
                        
                        if realm == 'electromagnetic' and 'band_gap' in df.columns:
                            bg_values = realm_materials['band_gap'].values
                            bg_values = bg_values[~np.isnan(bg_values)]
                            if len(bg_values) > 0:
                                realm_properties['band_gap_mean'] = float(np.mean(bg_values))
                                realm_properties['semiconductors'] = float(np.sum((bg_values > 0.1) & (bg_values < 4.0)) / len(bg_values))
                        
                        if realm_properties:
                            realm_validation[realm] = {
                                'material_count': len(realm_materials),
                                'properties': realm_properties
                            }
            
            # Overall materials validation
            significant_correlations = sum(1 for result in correlation_results.values() if result['significant'])
            
            materials_validation = {
                'status': 'passed' if significant_correlations >= 1 else 'failed',
                'property_correlations': correlation_results,
                'realm_validation': realm_validation,
                'significant_correlations_count': significant_correlations,
                'total_correlations_tested': len(correlation_results)
            }
            
            self.validation_results['materials_validation'] = materials_validation
            
            print(f"  Materials Property Correlations:")
            print(f"    Properties tested: {len(correlation_results)}")
            print(f"    Significant correlations: {significant_correlations}")
            print(f"    Realms validated: {len(realm_validation)}")
            
            status = "✅ PASSED" if materials_validation['status'] == 'passed' else "❌ FAILED"
            print(f"  Materials Validation: {status}")
            
        except Exception as e:
            print(f"  ❌ Error in materials validation: {e}")
            self.validation_results['materials_validation'] = {'status': 'error', 'error': str(e)}
    
    def _calculate_overall_validation_score(self):
        """Calculate overall UBP system validation score"""
        
        try:
            # Define validation weights
            validation_weights = {
                'nrci_validation': 0.25,      # Core UBP metric
                'coherence_validation': 0.20,  # Cross-realm coherence
                'fractal_validation': 0.15,    # Geometric structure
                'resonance_validation': 0.15,  # Sacred geometry
                'energy_validation': 0.10,     # Energy equation
                'temporal_validation': 0.05,   # Temporal coherence
                'glr_validation': 0.05,        # GLR frameworks
                'tgic_validation': 0.03,       # TGIC constraints
                'materials_validation': 0.02   # Materials correlations
            }
            
            # Calculate weighted score
            total_score = 0.0
            total_weight = 0.0
            validation_breakdown = {}
            
            for validation_name, weight in validation_weights.items():
                if validation_name in self.validation_results:
                    result = self.validation_results[validation_name]
                    
                    if result.get('status') == 'passed':
                        score = 1.0
                    elif result.get('status') == 'failed':
                        score = 0.0
                    else:  # error or partial
                        score = 0.0
                    
                    total_score += score * weight
                    total_weight += weight
                    
                    validation_breakdown[validation_name] = {
                        'score': score,
                        'weight': weight,
                        'weighted_score': score * weight,
                        'status': result.get('status', 'unknown')
                    }
            
            # Normalize score
            if total_weight > 0:
                overall_score = total_score / total_weight
            else:
                overall_score = 0.0
            
            # Determine overall status
            if overall_score >= 0.8:
                overall_status = 'excellent'
            elif overall_score >= 0.6:
                overall_status = 'good'
            elif overall_score >= 0.4:
                overall_status = 'partial'
            else:
                overall_status = 'failed'
            
            # Count validations passed
            validations_passed = sum(1 for breakdown in validation_breakdown.values() if breakdown['score'] == 1.0)
            total_validations = len(validation_breakdown)
            
            overall_validation = {
                'overall_score': float(overall_score),
                'overall_status': overall_status,
                'validations_passed': validations_passed,
                'total_validations': total_validations,
                'pass_percentage': float(100 * validations_passed / total_validations) if total_validations > 0 else 0.0,
                'validation_breakdown': validation_breakdown,
                'validation_weights': validation_weights
            }
            
            self.validation_results['overall_validation'] = overall_validation
            
            print(f"  Overall UBP System Validation:")
            print(f"    Score: {overall_score:.3f} ({overall_status.upper()})")
            print(f"    Validations passed: {validations_passed}/{total_validations} ({overall_validation['pass_percentage']:.1f}%)")
            
            # Detailed breakdown
            print(f"\\n  Validation Breakdown:")
            for name, breakdown in validation_breakdown.items():
                status_symbol = "✅" if breakdown['score'] == 1.0 else "❌"
                print(f"    {status_symbol} {name}: {breakdown['score']:.1f} (weight: {breakdown['weight']:.2f})")
            
        except Exception as e:
            print(f"  ❌ Error calculating overall validation: {e}")
            self.validation_results['overall_validation'] = {'status': 'error', 'error': str(e)}
    
    def save_validation_results(self, filename="ubp_validation_results.json"):
        """Save comprehensive validation results"""
        
        print(f"\\n💾 Saving UBP validation results...")
        
        try:
            # Save main validation results
            with open(filename, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            
            # Create validation summary report
            summary_filename = filename.replace('.json', '_summary.txt')
            with open(summary_filename, 'w') as f:
                f.write("UBP PRINCIPLES VALIDATION SUMMARY\\n")
                f.write("="*50 + "\\n\\n")
                
                if 'overall_validation' in self.validation_results:
                    overall = self.validation_results['overall_validation']
                    f.write(f"Overall Score: {overall['overall_score']:.3f} ({overall['overall_status'].upper()})\\n")
                    f.write(f"Validations Passed: {overall['validations_passed']}/{overall['total_validations']}\\n\\n")
                
                # Individual validation results
                f.write("INDIVIDUAL VALIDATIONS:\\n")
                for validation_name, result in self.validation_results.items():
                    if validation_name != 'overall_validation':
                        status = result.get('status', 'unknown')
                        f.write(f"  {validation_name}: {status.upper()}\\n")
                
                # UBP targets summary
                f.write("\\nUBP TARGETS:\\n")
                for target_name, target_value in self.validation_targets.items():
                    f.write(f"  {target_name}: {target_value}\\n")
            
            # Create visualization of validation results
            self._create_validation_visualization()
            
            print(f"✅ Saved validation results to {filename}")
            print(f"✅ Saved summary to {summary_filename}")
            print(f"✅ Created validation visualization")
            
        except Exception as e:
            print(f"❌ Error saving validation results: {e}")
    
    def _create_validation_visualization(self):
        """Create visualization of validation results"""
        
        try:
            if 'overall_validation' not in self.validation_results:
                return
            
            overall = self.validation_results['overall_validation']
            breakdown = overall['validation_breakdown']
            
            # Create validation summary plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Validation scores bar chart
            validation_names = list(breakdown.keys())
            scores = [breakdown[name]['score'] for name in validation_names]
            weights = [breakdown[name]['weight'] for name in validation_names]
            
            # Clean up names for display
            display_names = [name.replace('_validation', '').replace('_', ' ').title() for name in validation_names]
            
            bars = ax1.bar(range(len(display_names)), scores, color=['green' if s == 1.0 else 'red' for s in scores])
            ax1.set_xlabel('Validation Category')
            ax1.set_ylabel('Score (0 = Failed, 1 = Passed)')
            ax1.set_title('UBP Validation Results')
            ax1.set_xticks(range(len(display_names)))
            ax1.set_xticklabels(display_names, rotation=45, ha='right')
            ax1.set_ylim(0, 1.1)
            
            # Add score labels on bars
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{score:.1f}', ha='center', va='bottom')
            
            # Weighted contribution pie chart
            weighted_scores = [breakdown[name]['weighted_score'] for name in validation_names]
            colors = ['green' if breakdown[name]['score'] == 1.0 else 'red' for name in validation_names]
            
            ax2.pie(weights, labels=display_names, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Validation Weight Distribution')
            
            plt.tight_layout()
            plt.savefig('ubp_validation_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create detailed validation heatmap
            plt.figure(figsize=(12, 8))
            
            # Prepare data for heatmap
            validation_data = []
            validation_labels = []
            
            for validation_name, result in self.validation_results.items():
                if validation_name != 'overall_validation' and isinstance(result, dict):
                    validation_labels.append(validation_name.replace('_validation', '').replace('_', ' ').title())
                    
                    # Extract key metrics for heatmap
                    row_data = []
                    
                    # Status (0 or 1)
                    row_data.append(1.0 if result.get('status') == 'passed' else 0.0)
                    
                    # Additional metrics based on validation type
                    if 'statistics' in result:
                        stats = result['statistics']
                        if 'mean' in stats:
                            row_data.append(min(1.0, stats['mean']))
                    else:
                        row_data.append(0.5)
                    
                    if 'target_achievement' in result:
                        achievement = result['target_achievement']
                        if 'percentage_achieved' in achievement:
                            row_data.append(achievement['percentage_achieved'] / 100.0)
                        else:
                            row_data.append(0.0)
                    else:
                        row_data.append(0.0)
                    
                    validation_data.append(row_data)
            
            if validation_data:
                validation_array = np.array(validation_data)
                
                sns.heatmap(validation_array, 
                           xticklabels=['Status', 'Mean Value', 'Target Achievement'],
                           yticklabels=validation_labels,
                           annot=True, fmt='.2f', cmap='RdYlGn', 
                           cbar_kws={'label': 'Validation Score'})
                
                plt.title('UBP Validation Detailed Heatmap')
                plt.tight_layout()
                plt.savefig('ubp_validation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            print(f"  ⚠️  Error creating validation visualization: {e}")

def main():
    """Main execution function"""
    
    print("Starting UBP principles validation analysis...")
    print("Phase 5: Comprehensive UBP Framework Validation")
    print()
    
    # Initialize validation analyzer
    analyzer = UBPValidationAnalyzer()
    
    # Perform comprehensive validation
    df, validation_results = analyzer.perform_comprehensive_validation(
        "ubp_encoded_inorganic_materials.csv",
        "ubp_geometric_analysis_results.json"
    )
    
    if df is not None and validation_results:
        # Save validation results
        analyzer.save_validation_results()
        
        print("\\n" + "="*80)
        print("PHASE 5 COMPLETE")
        print("="*80)
        
        if 'overall_validation' in validation_results:
            overall = validation_results['overall_validation']
            print(f"✅ UBP validation complete: {overall['overall_score']:.3f} ({overall['overall_status'].upper()})")
            print(f"✅ Validations passed: {overall['validations_passed']}/{overall['total_validations']}")
        else:
            print("✅ UBP validation analysis complete")
        
        print("✅ Comprehensive validation results generated")
        print("✅ Ready for Phase 6: Predictive models for materials discovery")
        
        return df, validation_results
    else:
        print("❌ UBP validation failed")
        return None, None

if __name__ == "__main__":
    dataset, validation = main()
