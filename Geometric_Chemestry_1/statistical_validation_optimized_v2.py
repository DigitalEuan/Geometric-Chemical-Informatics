#!/usr/bin/env python3
"""
Statistical Validation of Geometric Patterns in Chemical Space
Focused analysis with permutation testing and significance validation
"""

import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class StatisticalValidator:
    """Statistical validation of geometric patterns and biological relationships"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.pi = np.pi
        self.sqrt2 = np.sqrt(2)
        self.e = np.e
        
    def perform_umap_analysis(self, df, morgan_fps):
        """Perform UMAP and return embeddings with statistical analysis"""
        print("Performing UMAP analysis...")
        
        # Prepare features
        descriptor_cols = [col for col in df.columns if col not in 
                          ['canonical_smiles', 'molecule_chembl_id', 'standard_value', 
                           'target_chembl_id', 'pIC50']]
        
        descriptors = df[descriptor_cols].values
        combined_features = np.hstack([descriptors, morgan_fps])
        scaled_features = self.scaler.fit_transform(combined_features)
        
        # UMAP projections
        umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = umap_2d.fit_transform(scaled_features)
        
        umap_3d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
        embedding_3d = umap_3d.fit_transform(scaled_features)
        
        print(f"2D embedding shape: {embedding_2d.shape}")
        print(f"3D embedding shape: {embedding_3d.shape}")
        
        return embedding_2d, embedding_3d, scaled_features
    
    def calculate_distance_activity_correlation(self, embedding, pic50_values, sample_size=100000):
        """Calculate correlation between geometric distance and activity similarity using sampling"""
        print(f"Calculating distance-activity correlations with sample size {sample_size}...")
        
        n_points = len(pic50_values)
        
        # Generate random pairs of indices
        indices1 = np.random.randint(0, n_points, sample_size)
        indices2 = np.random.randint(0, n_points, sample_size)
        
        # Ensure we don't have i == j
        mask = indices1 == indices2
        while np.any(mask):
            indices2[mask] = np.random.randint(0, n_points, np.sum(mask))
            mask = indices1 == indices2

        # Calculate distances and activity differences for the sampled pairs
        distances = np.linalg.norm(embedding[indices1] - embedding[indices2], axis=1)
        activity_diffs = np.abs(pic50_values[indices1] - pic50_values[indices2])
        
        # Calculate correlations
        spearman_corr, spearman_p = spearmanr(distances, activity_diffs)
        pearson_corr, pearson_p = pearsonr(distances, activity_diffs)
        
        return {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'distances': distances,
            'activity_diffs': activity_diffs,
            'n_pairs': sample_size
        }
    
    def permutation_test_correlation(self, distances, activity_diffs, n_permutations=100):
        """Perform permutation test to validate correlation significance"""
        print(f"Performing permutation test with {n_permutations} iterations...")
        
        # Original correlation
        original_corr, _ = spearmanr(distances, activity_diffs)
        
        # Permutation test
        permuted_correlations = []
        for i in range(n_permutations):
            shuffled_activity = np.random.permutation(activity_diffs)
            perm_corr, _ = spearmanr(distances, shuffled_activity)
            permuted_correlations.append(perm_corr)
        
        permuted_correlations = np.array(permuted_correlations)
        
        # Calculate p-value
        if original_corr < 0:
            p_value = np.sum(permuted_correlations <= original_corr) / n_permutations
        else:
            p_value = np.sum(permuted_correlations >= original_corr) / n_permutations
        
        return {
            'original_correlation': original_corr,
            'permutation_p_value': p_value,
            'permuted_correlations': permuted_correlations,
            'is_significant': p_value < 0.05
        }
    
    def analyze_resonance_patterns(self, embedding, pic50_values, tolerance=0.1, sample_size=100000):
        """Analyze resonance patterns with statistical validation using sampling"""
        print(f"Analyzing resonance patterns with sample size {sample_size}...")
        
        n_points = len(pic50_values)
        
        resonance_targets = {
            'pi': self.pi,
            'phi': self.phi,
            'sqrt2': self.sqrt2,
            'e': self.e
        }
        
        results = {}
        
        # Generate random pairs of indices
        indices1 = np.random.randint(0, n_points, sample_size)
        indices2 = np.random.randint(0, n_points, sample_size)
        
        # Ensure we don't have i == j
        mask = indices1 == indices2
        while np.any(mask):
            indices2[mask] = np.random.randint(0, n_points, np.sum(mask))
            mask = indices1 == indices2

        distances = np.linalg.norm(embedding[indices1] - embedding[indices2], axis=1)
        activity_diffs = np.abs(pic50_values[indices1] - pic50_values[indices2])

        for name, target in resonance_targets.items():
            # Find resonant pairs
            resonant_mask = np.abs(distances - target) < tolerance
            resonant_pairs = activity_diffs[resonant_mask]
            non_resonant_pairs = activity_diffs[~resonant_mask]
            
            if len(resonant_pairs) > 10 and len(non_resonant_pairs) > 10:
                # Statistical test
                statistic, p_value = mannwhitneyu(resonant_pairs, non_resonant_pairs, 
                                                alternative='two-sided')
                
                results[name] = {
                    'n_resonant_pairs': len(resonant_pairs),
                    'n_non_resonant_pairs': len(non_resonant_pairs),
                    'resonant_mean_activity_diff': np.mean(resonant_pairs),
                    'non_resonant_mean_activity_diff': np.mean(non_resonant_pairs),
                    'mann_whitney_statistic': statistic,
                    'mann_whitney_p_value': p_value,
                    'is_significant': p_value < 0.05,
                    'effect_size': (np.mean(resonant_pairs) - np.mean(non_resonant_pairs)) / 
                                  np.sqrt((np.var(resonant_pairs) + np.var(non_resonant_pairs)) / 2)
                }
                
                print(f"{name} resonance: {len(resonant_pairs)} pairs, p={p_value:.4f}")
        
        return results
    
    def cluster_analysis(self, embedding, pic50_values, target_ids):
        """Analyze clustering patterns by target and activity"""
        print("Performing cluster analysis...")
        
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(embedding)
        cluster_labels = clustering.labels_
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Calculate silhouette score (excluding noise points)
        if n_clusters > 1:
            mask = cluster_labels != -1
            if np.sum(mask) > 1:
                silhouette_avg = silhouette_score(embedding[mask], cluster_labels[mask])
            else:
                silhouette_avg = 0
        else:
            silhouette_avg = 0
        
        # Analyze cluster composition
        cluster_analysis = {}
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise
                continue
                
            mask = cluster_labels == cluster_id
            cluster_pic50 = pic50_values[mask]
            cluster_targets = target_ids[mask]
            
            cluster_analysis[cluster_id] = {
                'size': np.sum(mask),
                'mean_pic50': np.mean(cluster_pic50),
                'std_pic50': np.std(cluster_pic50),
                'unique_targets': len(np.unique(cluster_targets)),
                'target_distribution': dict(zip(*np.unique(cluster_targets, return_counts=True)))
            }
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette_avg,
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis
        }
    
    def predictive_modeling(self, features, pic50_values, embedding_2d, embedding_3d):
        """Build predictive models using different feature sets"""
        print("Building predictive models...")
        
        # Prepare different feature sets
        feature_sets = {
            'molecular_descriptors': features[:, :27],  # First 27 are molecular descriptors
            'morgan_fingerprints': features[:, 27:],   # Rest are Morgan fingerprints
            'combined_features': features,
            'umap_2d': embedding_2d,
            'umap_3d': embedding_3d,
            'combined_with_umap_2d': np.hstack([features, embedding_2d]),
            'combined_with_umap_3d': np.hstack([features, embedding_3d])
        }
        
        results = {}
        
        for name, X in feature_sets.items():
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, pic50_values, test_size=0.2, random_state=42
            )
            
            # Random Forest model
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            # Predictions
            y_pred = rf.predict(X_test)
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'n_features': X.shape[1]
            }
            
            print(f"{name}: R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        
        return results
    
    def create_visualizations(self, embedding_2d, embedding_3d, pic50_values, target_ids, corr_2d):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 2D UMAP colored by pIC50
        scatter = axes[0, 0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                   c=pic50_values, cmap='viridis', alpha=0.6, s=20)
        axes[0, 0].set_title('2D UMAP - Colored by pIC50')
        axes[0, 0].set_xlabel('UMAP Dimension 1')
        axes[0, 0].set_ylabel('UMAP Dimension 2')
        plt.colorbar(scatter, ax=axes[0, 0], label='pIC50')
        
        # 2D UMAP colored by target
        unique_targets = np.unique(target_ids)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_targets)))
        for i, target in enumerate(unique_targets):
            mask = target_ids == target
            axes[0, 1].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                             c=[colors[i]], label=target, alpha=0.6, s=20)
        axes[0, 1].set_title('2D UMAP - Colored by Target')
        axes[0, 1].set_xlabel('UMAP Dimension 1')
        axes[0, 1].set_ylabel('UMAP Dimension 2')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # pIC50 distribution
        axes[1, 0].hist(pic50_values, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('pIC50 Distribution')
        axes[1, 0].set_xlabel('pIC50')
        axes[1, 0].set_ylabel('Frequency')
        
        # Distance vs Activity correlation
        distances = corr_2d['distances']
        activity_diffs = corr_2d['activity_diffs']
        
        axes[1, 1].scatter(distances, activity_diffs, 
                          alpha=0.3, s=1)
        axes[1, 1].set_title('Geometric Distance vs Activity Difference')
        axes[1, 1].set_xlabel('UMAP Distance')
        axes[1, 1].set_ylabel('pIC50 Difference')
        
        plt.tight_layout()
        plt.savefig('geometric_analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved visualization to geometric_analysis_summary.png")

def main():
    """Main execution function"""
    print("Loading processed dataset...")
    df = pd.read_csv('kinase_compounds_features.csv')
    morgan_fps = np.load('morgan_fingerprints.npy')
    
    print(f"Dataset: {len(df)} compounds")
    
    # Initialize validator
    validator = StatisticalValidator()
    
    # Perform UMAP analysis
    embedding_2d, embedding_3d, features = validator.perform_umap_analysis(df, morgan_fps)
    
    # Extract target values
    pic50_values = df['pIC50'].values
    target_ids = df['target_chembl_id'].values
    
    # Distance-activity correlation analysis
    print("\n" + "="*60)
    print("DISTANCE-ACTIVITY CORRELATION ANALYSIS")
    print("="*60)
    
    corr_2d = validator.calculate_distance_activity_correlation(embedding_2d, pic50_values)
    corr_3d = validator.calculate_distance_activity_correlation(embedding_3d, pic50_values)
    
    print(f"2D Analysis:")
    print(f"  Spearman correlation: {corr_2d['spearman_correlation']:.4f} (p={corr_2d['spearman_p_value']:.2e})")
    print(f"  Pearson correlation: {corr_2d['pearson_correlation']:.4f} (p={corr_2d['pearson_p_value']:.2e})")
    
    print(f"3D Analysis:")
    print(f"  Spearman correlation: {corr_3d['spearman_correlation']:.4f} (p={corr_3d['spearman_p_value']:.2e})")
    print(f"  Pearson correlation: {corr_3d['pearson_correlation']:.4f} (p={corr_3d['pearson_p_value']:.2e})")
    
    # Permutation testing
    print("\n" + "="*60)
    print("PERMUTATION TEST VALIDATION")
    print("="*60)
    
    perm_test_2d = validator.permutation_test_correlation(
        corr_2d['distances'], corr_2d['activity_diffs'], n_permutations=100
    )
    
    print(f"2D Permutation Test:")
    print(f"  Original correlation: {perm_test_2d['original_correlation']:.4f}")
    print(f"  Permutation p-value: {perm_test_2d['permutation_p_value']:.4f}")
    print(f"  Significant: {perm_test_2d['is_significant']}")
    
    # Resonance pattern analysis
    print("\n" + "="*60)
    print("RESONANCE PATTERN ANALYSIS")
    print("="*60)
    
    resonance_2d = validator.analyze_resonance_patterns(embedding_2d, pic50_values)
    
    for pattern, results in resonance_2d.items():
        print(f"{pattern.upper()} Resonance:")
        print(f"  Resonant pairs: {results['n_resonant_pairs']}")
        print(f"  Mean activity diff (resonant): {results['resonant_mean_activity_diff']:.4f}")
        print(f"  Mean activity diff (non-resonant): {results['non_resonant_mean_activity_diff']:.4f}")
        print(f"  Mann-Whitney p-value: {results['mann_whitney_p_value']:.4f}")
        print(f"  Effect size: {results['effect_size']:.4f}")
        print(f"  Significant: {results['is_significant']}")
        print()
    
    # Cluster analysis
    print("="*60)
    print("CLUSTER ANALYSIS")
    print("="*60)
    
    cluster_results = validator.cluster_analysis(embedding_2d, pic50_values, target_ids)
    
    print(f"Number of clusters: {cluster_results['n_clusters']}")
    print(f"Noise points: {cluster_results['n_noise']}")
    print(f"Silhouette score: {cluster_results['silhouette_score']:.4f}")
    
    # Predictive modeling
    print("\n" + "="*60)
    print("PREDICTIVE MODELING RESULTS")
    print("="*60)
    
    model_results = validator.predictive_modeling(features, pic50_values, embedding_2d, embedding_3d)
    
    # Create visualizations
    validator.create_visualizations(embedding_2d, embedding_3d, pic50_values, target_ids, corr_2d)
    
    # Save results
    results_summary = {
        'correlation_2d': corr_2d,
        'correlation_3d': corr_3d,
        'permutation_test_2d': perm_test_2d,
        'resonance_analysis_2d': resonance_2d,
        'cluster_analysis': cluster_results,
        'predictive_models': model_results
    }
    
    # Save embeddings with results
    df_results = df.copy()
    df_results['UMAP1'] = embedding_2d[:, 0]
    df_results['UMAP2'] = embedding_2d[:, 1]
    df_results['UMAP3_1'] = embedding_3d[:, 0]
    df_results['UMAP3_2'] = embedding_3d[:, 1]
    df_results['UMAP3_3'] = embedding_3d[:, 2]
    df_results['cluster_label'] = cluster_results['cluster_labels']
    
    df_results.to_csv('final_geometric_analysis_results.csv', index=False)
    
    print(f"\nAnalysis complete. Results saved to:")
    print(f"  - final_geometric_analysis_results.csv")
    print(f"  - geometric_analysis_summary.png")

if __name__ == "__main__":
    main()

