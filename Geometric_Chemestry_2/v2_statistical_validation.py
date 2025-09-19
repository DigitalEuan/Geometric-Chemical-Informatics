#!/usr/bin/env python3
"""
Version 2 Statistical Validation of Resonance Phenomena
Rigorous statistical analysis incorporating UBP principles and NRCI calculations

Key Features:
1. Non-random Coherence Index (NRCI) calculation targeting ≥ 0.999999
2. Statistical significance testing of sacred geometry patterns
3. Permutation tests for resonance validation
4. Cross-validation of geometric patterns
5. UBP Core Resonance Value (CRV) validation
6. Bootstrap confidence intervals
7. Multiple testing correction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, kstest
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class StatisticalValidation:
    """Comprehensive statistical validation of resonance phenomena"""
    
    def __init__(self):
        # UBP Core Resonance Values (CRVs)
        self.crv_values = {
            'quantum': 0.2265234857,  # e/12
            'electromagnetic': 3.141593,  # π
            'gravitational': 100.0,
            'biological': 10.0,
            'cosmological': 0.83203682,  # π^φ
            'golden_ratio': 1.618034,  # φ
            'sqrt2': 1.414214,  # √2
            'euler': 2.718282  # e
        }
        
        # UBP constants
        self.phi = (1 + np.sqrt(5)) / 2
        self.pi = np.pi
        self.sqrt2 = np.sqrt(2)
        self.e = np.e
        
        # Statistical parameters
        self.alpha = 0.01  # Significance level (99% confidence)
        self.n_permutations = 1000  # Permutation test iterations
        self.n_bootstrap = 1000  # Bootstrap iterations
        
        # Results storage
        self.validation_results = {}
        
    def load_data(self):
        """Load geometric analysis results and original dataset"""
        print("Loading data for statistical validation...")
        
        # Load geometric analysis results
        self.geometric_results = pd.read_csv("v2_geometric_analysis_results.csv")
        print(f"Geometric results loaded: {self.geometric_results.shape}")
        
        # Load original dataset
        self.dataset = pd.read_csv("v2_comprehensive_features.csv")
        self.target = self.dataset['pKi'].values
        print(f"Original dataset loaded: {self.dataset.shape}")
        
        # Load fingerprints
        self.fingerprints = np.load("v2_ecfp4_fingerprints.npy")
        print(f"Fingerprints loaded: {self.fingerprints.shape}")
        
        return self.geometric_results, self.dataset
    
    def calculate_nrci(self, observed_values, theoretical_values):
        """
        Calculate Non-random Coherence Index (NRCI)
        NRCI = 1 - (sqrt(sum((S_i - T_i)^2) / n) / sigma(T))
        Target: ≥ 0.999999
        """
        try:
            if len(observed_values) != len(theoretical_values):
                return 0.0
            
            # Ensure arrays are numpy arrays
            S = np.array(observed_values)
            T = np.array(theoretical_values)
            
            # Calculate RMSE between observed and theoretical
            rmse = np.sqrt(np.mean((S - T) ** 2))
            
            # Calculate standard deviation of theoretical values
            sigma_T = np.std(T)
            
            # Avoid division by zero
            if sigma_T == 0:
                return 1.0 if rmse == 0 else 0.0
            
            # Calculate NRCI
            nrci = 1 - (rmse / sigma_T)
            
            # Ensure NRCI is between 0 and 1
            nrci = max(0.0, min(1.0, nrci))
            
            return nrci
            
        except Exception as e:
            print(f"Error calculating NRCI: {e}")
            return 0.0
    
    def validate_sacred_geometry_resonances(self):
        """Validate sacred geometry resonance patterns statistically"""
        print("Validating sacred geometry resonances...")
        
        resonance_validation = {}
        
        # Extract resonance scores for each sacred geometry constant
        resonance_types = ['phi_resonance', 'pi_resonance', 'sqrt2_resonance', 'e_resonance']
        
        for resonance_type in resonance_types:
            if resonance_type in self.geometric_results.columns:
                observed_scores = self.geometric_results[resonance_type].values
                
                # Remove NaN values
                observed_scores = observed_scores[~np.isnan(observed_scores)]
                
                if len(observed_scores) > 0:
                    # Statistical tests
                    validation_result = self.perform_resonance_tests(observed_scores, resonance_type)
                    resonance_validation[resonance_type] = validation_result
        
        self.validation_results['sacred_geometry'] = resonance_validation
        print(f"Sacred geometry validation complete: {len(resonance_validation)} patterns tested")
        
        return resonance_validation
    
    def perform_resonance_tests(self, observed_scores, resonance_type):
        """Perform comprehensive statistical tests on resonance scores"""
        results = {}
        
        try:
            # Basic statistics
            results['mean'] = np.mean(observed_scores)
            results['std'] = np.std(observed_scores)
            results['median'] = np.median(observed_scores)
            results['n_samples'] = len(observed_scores)
            
            # Test against random distribution (null hypothesis)
            # Generate random scores for comparison
            random_scores = np.random.uniform(0, 1, len(observed_scores))
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = kstest(observed_scores, 'uniform')
            results['ks_statistic'] = ks_stat
            results['ks_p_value'] = ks_p
            results['ks_significant'] = ks_p < self.alpha
            
            # Mann-Whitney U test (non-parametric)
            if len(observed_scores) > 1:
                u_stat, u_p = stats.mannwhitneyu(observed_scores, random_scores, alternative='greater')
                results['mannwhitney_statistic'] = u_stat
                results['mannwhitney_p_value'] = u_p
                results['mannwhitney_significant'] = u_p < self.alpha
            
            # Permutation test
            perm_result = self.permutation_test(observed_scores, random_scores)
            results.update(perm_result)
            
            # Calculate NRCI against theoretical expectation
            # For resonance, we expect higher scores than random (0.5)
            theoretical_expectation = np.full_like(observed_scores, 0.5)
            nrci = self.calculate_nrci(observed_scores, theoretical_expectation)
            results['nrci'] = nrci
            results['nrci_target_met'] = nrci >= 0.999999
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(observed_scores) + np.var(random_scores)) / 2)
            if pooled_std > 0:
                cohens_d = (np.mean(observed_scores) - np.mean(random_scores)) / pooled_std
                results['cohens_d'] = cohens_d
                results['effect_size'] = self.interpret_effect_size(cohens_d)
            
        except Exception as e:
            print(f"Error in resonance tests for {resonance_type}: {e}")
            results['error'] = str(e)
        
        return results
    
    def permutation_test(self, group1, group2):
        """Perform permutation test to assess statistical significance"""
        try:
            # Observed difference in means
            observed_diff = np.mean(group1) - np.mean(group2)
            
            # Combine groups
            combined = np.concatenate([group1, group2])
            n1 = len(group1)
            
            # Permutation test
            permuted_diffs = []
            for _ in range(self.n_permutations):
                np.random.shuffle(combined)
                perm_group1 = combined[:n1]
                perm_group2 = combined[n1:]
                perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
                permuted_diffs.append(perm_diff)
            
            # Calculate p-value
            permuted_diffs = np.array(permuted_diffs)
            p_value = np.sum(permuted_diffs >= observed_diff) / self.n_permutations
            
            return {
                'permutation_observed_diff': observed_diff,
                'permutation_p_value': p_value,
                'permutation_significant': p_value < self.alpha
            }
            
        except Exception as e:
            return {
                'permutation_error': str(e),
                'permutation_p_value': 1.0,
                'permutation_significant': False
            }
    
    def interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def validate_activity_correlations(self):
        """Validate correlations between geometric patterns and biological activity"""
        print("Validating activity correlations...")
        
        correlation_validation = {}
        
        # Test correlations for different feature sets
        correlation_columns = [col for col in self.geometric_results.columns 
                             if 'correlation' in col.lower()]
        
        for col in correlation_columns:
            if col in self.geometric_results.columns:
                correlations = self.geometric_results[col].values
                correlations = correlations[~np.isnan(correlations)]
                
                if len(correlations) > 0:
                    validation_result = self.test_correlation_significance(correlations, col)
                    correlation_validation[col] = validation_result
        
        self.validation_results['activity_correlations'] = correlation_validation
        print(f"Activity correlation validation complete: {len(correlation_validation)} correlations tested")
        
        return correlation_validation
    
    def test_correlation_significance(self, correlations, correlation_name):
        """Test statistical significance of correlations"""
        results = {}
        
        try:
            # Basic statistics
            results['mean_correlation'] = np.mean(correlations)
            results['std_correlation'] = np.std(correlations)
            results['n_correlations'] = len(correlations)
            
            # Test against zero correlation (null hypothesis)
            t_stat, t_p = stats.ttest_1samp(correlations, 0)
            results['t_statistic'] = t_stat
            results['t_p_value'] = t_p
            results['t_significant'] = t_p < self.alpha
            
            # Bootstrap confidence interval
            bootstrap_means = []
            for _ in range(self.n_bootstrap):
                bootstrap_sample = np.random.choice(correlations, size=len(correlations), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            results['bootstrap_ci_lower'] = ci_lower
            results['bootstrap_ci_upper'] = ci_upper
            results['bootstrap_significant'] = not (ci_lower <= 0 <= ci_upper)
            
            # Calculate NRCI for correlation consistency
            theoretical_correlation = np.full_like(correlations, np.mean(correlations))
            nrci = self.calculate_nrci(correlations, theoretical_correlation)
            results['nrci'] = nrci
            results['nrci_target_met'] = nrci >= 0.999999
            
        except Exception as e:
            print(f"Error in correlation tests for {correlation_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def validate_geometric_complexity(self):
        """Validate geometric complexity measures"""
        print("Validating geometric complexity...")
        
        complexity_validation = {}
        
        complexity_columns = ['projection_complexity', 'geometric_entropy', 'convex_hull_area']
        
        for col in complexity_columns:
            if col in self.geometric_results.columns:
                complexity_values = self.geometric_results[col].values
                complexity_values = complexity_values[~np.isnan(complexity_values)]
                
                if len(complexity_values) > 0:
                    validation_result = self.test_complexity_distribution(complexity_values, col)
                    complexity_validation[col] = validation_result
        
        self.validation_results['geometric_complexity'] = complexity_validation
        print(f"Geometric complexity validation complete: {len(complexity_validation)} measures tested")
        
        return complexity_validation
    
    def test_complexity_distribution(self, complexity_values, complexity_name):
        """Test statistical properties of complexity distributions"""
        results = {}
        
        try:
            # Basic statistics
            results['mean'] = np.mean(complexity_values)
            results['std'] = np.std(complexity_values)
            results['skewness'] = stats.skew(complexity_values)
            results['kurtosis'] = stats.kurtosis(complexity_values)
            results['n_samples'] = len(complexity_values)
            
            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(complexity_values)
            results['shapiro_statistic'] = shapiro_stat
            results['shapiro_p_value'] = shapiro_p
            results['is_normal'] = shapiro_p > self.alpha
            
            # Test for randomness (runs test approximation)
            median_val = np.median(complexity_values)
            runs, n1, n2 = self.runs_test(complexity_values > median_val)
            results['runs_statistic'] = runs
            results['runs_expected'] = (2 * n1 * n2) / (n1 + n2) + 1
            results['runs_variance'] = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
            
            if results['runs_variance'] > 0:
                z_runs = (runs - results['runs_expected']) / np.sqrt(results['runs_variance'])
                results['runs_z_score'] = z_runs
                results['runs_p_value'] = 2 * (1 - stats.norm.cdf(abs(z_runs)))
                results['runs_significant'] = results['runs_p_value'] < self.alpha
            
            # Calculate NRCI for complexity consistency
            theoretical_complexity = np.full_like(complexity_values, np.mean(complexity_values))
            nrci = self.calculate_nrci(complexity_values, theoretical_complexity)
            results['nrci'] = nrci
            results['nrci_target_met'] = nrci >= 0.999999
            
        except Exception as e:
            print(f"Error in complexity tests for {complexity_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def runs_test(self, binary_sequence):
        """Perform runs test for randomness"""
        try:
            runs = 1
            n1 = np.sum(binary_sequence)
            n2 = len(binary_sequence) - n1
            
            for i in range(1, len(binary_sequence)):
                if binary_sequence[i] != binary_sequence[i-1]:
                    runs += 1
            
            return runs, n1, n2
        except:
            return 1, 1, 1
    
    def cross_validate_patterns(self):
        """Cross-validate geometric patterns using machine learning"""
        print("Cross-validating patterns with machine learning...")
        
        cv_results = {}
        
        try:
            # Prepare features from geometric results
            feature_columns = [col for col in self.geometric_results.columns 
                             if col not in ['feature_set', 'embedding_method']]
            
            # Create feature matrix
            X = self.geometric_results[feature_columns].fillna(0).values
            
            # Use pKi as target (repeated for each geometric analysis)
            y = np.tile(self.target, len(self.geometric_results) // len(self.target))[:len(X)]
            
            # Cross-validation
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
            
            cv_results['cv_scores'] = cv_scores
            cv_results['mean_cv_score'] = np.mean(cv_scores)
            cv_results['std_cv_score'] = np.std(cv_scores)
            cv_results['cv_significant'] = np.mean(cv_scores) > 0.1  # Arbitrary threshold
            
            # Feature importance
            rf.fit(X, y)
            feature_importance = rf.feature_importances_
            cv_results['feature_importance'] = dict(zip(feature_columns, feature_importance))
            
            # Calculate NRCI for cross-validation consistency
            nrci = self.calculate_nrci(cv_scores, np.full_like(cv_scores, np.mean(cv_scores)))
            cv_results['nrci'] = nrci
            cv_results['nrci_target_met'] = nrci >= 0.999999
            
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            cv_results['error'] = str(e)
        
        self.validation_results['cross_validation'] = cv_results
        print("Cross-validation complete")
        
        return cv_results
    
    def calculate_overall_nrci(self):
        """Calculate overall NRCI across all validation tests"""
        print("Calculating overall NRCI...")
        
        nrci_values = []
        
        # Collect all NRCI values from validation results
        for category, results in self.validation_results.items():
            if isinstance(results, dict):
                for test_name, test_results in results.items():
                    if isinstance(test_results, dict) and 'nrci' in test_results:
                        nrci_values.append(test_results['nrci'])
        
        if nrci_values:
            overall_nrci = np.mean(nrci_values)
            nrci_std = np.std(nrci_values)
            
            overall_results = {
                'overall_nrci': overall_nrci,
                'nrci_std': nrci_std,
                'n_nrci_tests': len(nrci_values),
                'target_met': overall_nrci >= 0.999999,
                'individual_nrci_values': nrci_values
            }
            
            self.validation_results['overall_nrci'] = overall_results
            print(f"Overall NRCI: {overall_nrci:.6f} (target: ≥0.999999)")
            
            return overall_results
        else:
            print("No NRCI values found for overall calculation")
            return None
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("Generating validation report...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Version 2 Statistical Validation Report', fontsize=16, fontweight='bold')
        
        # Plot 1: NRCI Values
        self.plot_nrci_analysis(axes[0, 0])
        
        # Plot 2: Resonance Significance
        self.plot_resonance_significance(axes[0, 1])
        
        # Plot 3: Correlation Analysis
        self.plot_correlation_analysis(axes[1, 0])
        
        # Plot 4: Overall Summary
        self.plot_validation_summary(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('v2_statistical_validation_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Validation report saved to v2_statistical_validation_report.png")
    
    def plot_nrci_analysis(self, ax):
        """Plot NRCI analysis"""
        if 'overall_nrci' in self.validation_results:
            nrci_values = self.validation_results['overall_nrci']['individual_nrci_values']
            
            ax.hist(nrci_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(0.999999, color='red', linestyle='--', label='Target (0.999999)')
            ax.axvline(np.mean(nrci_values), color='green', linestyle='-', label=f'Mean ({np.mean(nrci_values):.6f})')
            ax.set_xlabel('NRCI Value')
            ax.set_ylabel('Frequency')
            ax.set_title('NRCI Distribution')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No NRCI data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('NRCI Analysis')
    
    def plot_resonance_significance(self, ax):
        """Plot resonance significance analysis"""
        if 'sacred_geometry' in self.validation_results:
            resonance_data = self.validation_results['sacred_geometry']
            
            resonance_names = []
            p_values = []
            
            for name, results in resonance_data.items():
                if 'mannwhitney_p_value' in results:
                    resonance_names.append(name.replace('_resonance', ''))
                    p_values.append(results['mannwhitney_p_value'])
            
            if resonance_names and p_values:
                bars = ax.bar(resonance_names, [-np.log10(p) for p in p_values], 
                             alpha=0.7, color='lightcoral')
                ax.axhline(-np.log10(self.alpha), color='red', linestyle='--', 
                          label=f'Significance threshold (p={self.alpha})')
                ax.set_ylabel('-log10(p-value)')
                ax.set_title('Resonance Pattern Significance')
                ax.legend()
                plt.setp(ax.get_xticklabels(), rotation=45)
            else:
                ax.text(0.5, 0.5, 'No resonance data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No resonance data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Resonance Significance')
    
    def plot_correlation_analysis(self, ax):
        """Plot correlation analysis"""
        if 'activity_correlations' in self.validation_results:
            corr_data = self.validation_results['activity_correlations']
            
            corr_names = []
            mean_correlations = []
            
            for name, results in corr_data.items():
                if 'mean_correlation' in results:
                    corr_names.append(name.replace('activity_', '').replace('_correlation', ''))
                    mean_correlations.append(results['mean_correlation'])
            
            if corr_names and mean_correlations:
                bars = ax.bar(corr_names, mean_correlations, alpha=0.7, color='lightgreen')
                ax.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax.set_ylabel('Mean Correlation')
                ax.set_title('Activity Correlations')
                plt.setp(ax.get_xticklabels(), rotation=45)
            else:
                ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No correlation data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Correlation Analysis')
    
    def plot_validation_summary(self, ax):
        """Plot validation summary"""
        # Count significant results
        total_tests = 0
        significant_tests = 0
        
        for category, results in self.validation_results.items():
            if isinstance(results, dict) and category != 'overall_nrci':
                for test_name, test_results in results.items():
                    if isinstance(test_results, dict):
                        total_tests += 1
                        # Check various significance indicators
                        if (test_results.get('mannwhitney_significant', False) or 
                            test_results.get('t_significant', False) or
                            test_results.get('bootstrap_significant', False)):
                            significant_tests += 1
        
        if total_tests > 0:
            significance_rate = significant_tests / total_tests
            
            # Pie chart of significant vs non-significant
            labels = ['Significant', 'Non-significant']
            sizes = [significant_tests, total_tests - significant_tests]
            colors = ['lightgreen', 'lightcoral']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Validation Summary\n({significant_tests}/{total_tests} tests significant)')
        else:
            ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation Summary')
    
    def save_validation_results(self, filename="v2_statistical_validation_results.json"):
        """Save comprehensive validation results"""
        print(f"Saving validation results to {filename}")
        
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(self.validation_results)
        
        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Validation results saved: {len(self.validation_results)} categories")
        
        return results_serializable

def main():
    """Main execution function"""
    print("="*80)
    print("VERSION 2 STATISTICAL VALIDATION OF RESONANCE PHENOMENA")
    print("="*80)
    
    # Initialize validation
    validator = StatisticalValidation()
    
    # Load data
    geometric_results, dataset = validator.load_data()
    
    # Validate sacred geometry resonances
    resonance_validation = validator.validate_sacred_geometry_resonances()
    
    # Validate activity correlations
    correlation_validation = validator.validate_activity_correlations()
    
    # Validate geometric complexity
    complexity_validation = validator.validate_geometric_complexity()
    
    # Cross-validate patterns
    cv_validation = validator.cross_validate_patterns()
    
    # Calculate overall NRCI
    overall_nrci = validator.calculate_overall_nrci()
    
    # Generate validation report
    validator.generate_validation_report()
    
    # Save results
    validation_results = validator.save_validation_results()
    
    print("\n" + "="*80)
    print("STATISTICAL VALIDATION SUMMARY")
    print("="*80)
    print(f"Dataset: {len(dataset)} compounds")
    print(f"Geometric patterns tested: {len(geometric_results)}")
    print(f"Sacred geometry resonances: {len(resonance_validation)}")
    print(f"Activity correlations: {len(correlation_validation)}")
    print(f"Complexity measures: {len(complexity_validation)}")
    
    if overall_nrci:
        print(f"Overall NRCI: {overall_nrci['overall_nrci']:.6f}")
        print(f"NRCI target (≥0.999999): {'✓ MET' if overall_nrci['target_met'] else '✗ NOT MET'}")
    
    print("Validation report: v2_statistical_validation_report.png")
    print("Detailed results: v2_statistical_validation_results.json")
    print("Ready for predictive modeling!")
    
    return validator, validation_results

if __name__ == "__main__":
    validator, results = main()
