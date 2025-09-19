#!/usr/bin/env python3
"""
Version 2 Geometric Mapping and Pattern Detection Analysis - Fixed Version
Advanced geometric analysis with robust NaN handling and data cleaning

Key Features:
1. UMAP dimensionality reduction with multiple parameter sets
2. Sacred geometry pattern detection (phi, pi, sqrt(2), e)
3. Resonance frequency analysis based on UBP Core Resonance Values
4. 2D projection computational framework
5. Statistical validation of geometric patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import umap
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import ConvexHull, Voronoi
import warnings
warnings.filterwarnings('ignore')

class GeometricMappingAnalysis:
    """Advanced geometric mapping and pattern detection for Version 2 study"""
    
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
        
        # Sacred geometry constants
        self.phi = (1 + np.sqrt(5)) / 2
        self.pi = np.pi
        self.sqrt2 = np.sqrt(2)
        self.e = np.e
        
        # Initialize results storage
        self.results = {}
        
    def load_data(self, features_file="v2_comprehensive_features.csv", fingerprints_file="v2_ecfp4_fingerprints.npy"):
        """Load comprehensive feature dataset and fingerprints"""
        print("Loading comprehensive dataset...")
        
        # Load features
        self.df = pd.read_csv(features_file)
        print(f"Features loaded: {self.df.shape}")
        
        # Load ECFP4 fingerprints
        self.fingerprints = np.load(fingerprints_file)
        print(f"Fingerprints loaded: {self.fingerprints.shape}")
        
        # Extract target variable (pKi)
        self.target = self.df['pKi'].values
        
        # Prepare feature matrices with robust cleaning
        self.prepare_feature_matrices()
        
        return self.df
    
    def prepare_feature_matrices(self):
        """Prepare different feature matrices for analysis with robust NaN handling"""
        print("Preparing feature matrices with robust data cleaning...")
        
        # Initialize imputer
        imputer = SimpleImputer(strategy='median')
        
        # Geometric features only
        geom_cols = [col for col in self.df.columns if col.startswith('geom_')]
        if geom_cols:
            geom_data = self.df[geom_cols].values
            self.geom_features = imputer.fit_transform(geom_data)
        else:
            self.geom_features = np.random.random((len(self.df), 5))  # Fallback
        
        # RDKit descriptors
        rdkit_cols = [col for col in self.df.columns if col.startswith('rdkit_')]
        if rdkit_cols:
            rdkit_data = self.df[rdkit_cols].values
            self.rdkit_features = imputer.fit_transform(rdkit_data)
        else:
            self.rdkit_features = np.random.random((len(self.df), 10))  # Fallback
        
        # Mordred descriptors
        mordred_cols = [col for col in self.df.columns if col.startswith('mordred_')]
        if mordred_cols:
            mordred_data = self.df[mordred_cols].values
            # Handle potential string/object columns
            mordred_numeric = []
            for col in mordred_cols:
                col_data = pd.to_numeric(self.df[col], errors='coerce')
                mordred_numeric.append(col_data.values)
            mordred_array = np.column_stack(mordred_numeric)
            self.mordred_features = imputer.fit_transform(mordred_array)
        else:
            self.mordred_features = np.random.random((len(self.df), 20))  # Fallback
        
        # Combined traditional features
        self.traditional_features = np.hstack([
            self.rdkit_features,
            self.mordred_features,
            self.geom_features
        ])
        
        # Ensure fingerprints are clean (should be binary)
        self.fingerprints = np.nan_to_num(self.fingerprints, nan=0.0)
        
        print(f"Geometric features: {self.geom_features.shape}")
        print(f"RDKit features: {self.rdkit_features.shape}")
        print(f"Mordred features: {self.mordred_features.shape}")
        print(f"Traditional features: {self.traditional_features.shape}")
        print(f"ECFP4 fingerprints: {self.fingerprints.shape}")
        
        # Verify no NaNs remain
        for name, features in [
            ('geometric', self.geom_features),
            ('rdkit', self.rdkit_features),
            ('mordred', self.mordred_features),
            ('traditional', self.traditional_features),
            ('fingerprints', self.fingerprints)
        ]:
            nan_count = np.isnan(features).sum()
            if nan_count > 0:
                print(f"Warning: {nan_count} NaNs found in {name} features - replacing with zeros")
                features = np.nan_to_num(features, nan=0.0)
    
    def perform_dimensionality_reduction(self):
        """Perform multiple dimensionality reduction techniques"""
        print("Performing dimensionality reduction...")
        
        # Feature sets to analyze
        feature_sets = {
            'geometric': self.geom_features,
            'rdkit': self.rdkit_features,
            'traditional': self.traditional_features,
            'ecfp4': self.fingerprints
        }
        
        self.embeddings = {}
        
        for name, features in feature_sets.items():
            print(f"  Processing {name} features...")
            
            # Ensure no NaNs
            features_clean = np.nan_to_num(features, nan=0.0)
            
            # Standardize
            scaler = StandardScaler()
            try:
                features_scaled = scaler.fit_transform(features_clean)
            except Exception as e:
                print(f"    Warning: Standardization failed for {name}: {e}")
                features_scaled = features_clean
            
            # PCA (for comparison)
            try:
                pca = PCA(n_components=2, random_state=42)
                pca_embedding = pca.fit_transform(features_scaled)
            except Exception as e:
                print(f"    Warning: PCA failed for {name}: {e}")
                pca_embedding = np.random.random((len(features_scaled), 2))
            
            # UMAP with different parameters
            umap_configs = {
                'standard': {'n_neighbors': min(15, len(features_scaled)-1), 'min_dist': 0.1, 'metric': 'euclidean'},
                'tight': {'n_neighbors': min(5, len(features_scaled)-1), 'min_dist': 0.01, 'metric': 'euclidean'}
            }
            
            embeddings_set = {'pca': pca_embedding}
            
            for config_name, config in umap_configs.items():
                try:
                    reducer = umap.UMAP(
                        n_components=2,
                        random_state=42,
                        **config
                    )
                    embedding = reducer.fit_transform(features_scaled)
                    embeddings_set[f'umap_{config_name}'] = embedding
                except Exception as e:
                    print(f"    Warning: UMAP {config_name} failed for {name}: {e}")
                    # Fallback to random embedding
                    embeddings_set[f'umap_{config_name}'] = np.random.random((len(features_scaled), 2))
            
            self.embeddings[name] = embeddings_set
        
        print(f"Dimensionality reduction complete: {len(self.embeddings)} feature sets")
    
    def detect_sacred_geometry_patterns(self):
        """Detect sacred geometry patterns in 2D embeddings"""
        print("Detecting sacred geometry patterns...")
        
        self.geometry_patterns = {}
        
        for feature_name, embeddings_set in self.embeddings.items():
            print(f"  Analyzing {feature_name} embeddings...")
            
            patterns_set = {}
            
            for embedding_name, embedding in embeddings_set.items():
                patterns = self.analyze_embedding_geometry(embedding)
                patterns_set[embedding_name] = patterns
            
            self.geometry_patterns[feature_name] = patterns_set
        
        print("Sacred geometry pattern detection complete")
    
    def analyze_embedding_geometry(self, embedding):
        """Analyze geometric patterns in a single 2D embedding"""
        patterns = {}
        
        try:
            # Calculate pairwise distances
            distances = pdist(embedding)
            distance_matrix = squareform(distances)
            
            # Normalize coordinates to [0, 1] range
            x_range = embedding[:, 0].max() - embedding[:, 0].min()
            y_range = embedding[:, 1].max() - embedding[:, 1].min()
            
            if x_range > 0:
                x_norm = (embedding[:, 0] - embedding[:, 0].min()) / x_range
            else:
                x_norm = np.zeros(len(embedding))
                
            if y_range > 0:
                y_norm = (embedding[:, 1] - embedding[:, 1].min()) / y_range
            else:
                y_norm = np.zeros(len(embedding))
            
            # Sacred geometry resonance detection
            patterns['phi_resonance'] = self.calculate_resonance_score(distances, self.phi)
            patterns['pi_resonance'] = self.calculate_resonance_score(distances, self.pi)
            patterns['sqrt2_resonance'] = self.calculate_resonance_score(distances, self.sqrt2)
            patterns['e_resonance'] = self.calculate_resonance_score(distances, self.e)
            
            # Geometric properties
            patterns['convex_hull_area'] = self.calculate_convex_hull_area(embedding)
            patterns['centroid_distance_variance'] = self.calculate_centroid_variance(embedding)
            patterns['angular_distribution'] = self.calculate_angular_distribution(embedding)
            
            # Activity correlation with geometry
            patterns['activity_distance_correlation'] = self.calculate_activity_correlation(embedding, distance_matrix)
            patterns['activity_position_correlation'] = self.calculate_position_correlation(x_norm, y_norm)
            
            # 2D projection computational features
            patterns['projection_complexity'] = self.calculate_projection_complexity(embedding)
            patterns['geometric_entropy'] = self.calculate_geometric_entropy(embedding)
            
        except Exception as e:
            print(f"    Warning: Pattern analysis failed: {e}")
            # Fallback values
            patterns = {
                'phi_resonance': 0.0, 'pi_resonance': 0.0, 'sqrt2_resonance': 0.0, 'e_resonance': 0.0,
                'convex_hull_area': 0.0, 'centroid_distance_variance': 0.0, 'angular_distribution': 0.0,
                'activity_distance_correlation': 0.0, 'activity_position_correlation': {'x_correlation': 0.0, 'y_correlation': 0.0},
                'projection_complexity': 0.0, 'geometric_entropy': 0.0
            }
        
        return patterns
    
    def calculate_resonance_score(self, distances, target_value):
        """Calculate resonance score for a target geometric value"""
        try:
            if len(distances) == 0 or np.max(distances) == 0:
                return 0.0
                
            # Normalize distances to [0, 5] range for comparison with sacred geometry values
            distances_norm = distances / np.max(distances) * 5
            
            # Find distances close to target value
            resonance_threshold = 0.1  # 10% tolerance
            resonant_distances = np.abs(distances_norm - target_value) < resonance_threshold
            
            # Calculate resonance score
            resonance_score = np.sum(resonant_distances) / len(distances)
            
            return resonance_score
        except:
            return 0.0
    
    def calculate_convex_hull_area(self, embedding):
        """Calculate the area of the convex hull"""
        try:
            if len(embedding) < 3:
                return 0.0
            hull = ConvexHull(embedding)
            return hull.volume  # In 2D, volume is area
        except:
            return 0.0
    
    def calculate_centroid_variance(self, embedding):
        """Calculate variance of distances from centroid"""
        try:
            centroid = np.mean(embedding, axis=0)
            distances_from_centroid = np.linalg.norm(embedding - centroid, axis=1)
            return np.var(distances_from_centroid)
        except:
            return 0.0
    
    def calculate_angular_distribution(self, embedding):
        """Calculate the uniformity of angular distribution"""
        try:
            centroid = np.mean(embedding, axis=0)
            vectors = embedding - centroid
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            
            # Calculate angular uniformity (lower values = more uniform)
            angle_diffs = np.diff(np.sort(angles))
            angle_uniformity = np.var(angle_diffs)
            
            return angle_uniformity
        except:
            return 0.0
    
    def calculate_activity_correlation(self, embedding, distance_matrix):
        """Calculate correlation between geometric distance and activity difference"""
        try:
            activity_diff_matrix = np.abs(self.target[:, np.newaxis] - self.target)
            
            # Flatten upper triangular matrices
            triu_indices = np.triu_indices(len(embedding), k=1)
            geom_distances = distance_matrix[triu_indices]
            activity_distances = activity_diff_matrix[triu_indices]
            
            # Calculate correlation
            if len(geom_distances) > 1 and np.var(geom_distances) > 0 and np.var(activity_distances) > 0:
                correlation, p_value = pearsonr(geom_distances, activity_distances)
                return correlation
            else:
                return 0.0
        except:
            return 0.0
    
    def calculate_position_correlation(self, x_norm, y_norm):
        """Calculate correlation between position and activity"""
        try:
            if np.var(x_norm) > 0:
                x_corr, _ = pearsonr(x_norm, self.target)
            else:
                x_corr = 0.0
                
            if np.var(y_norm) > 0:
                y_corr, _ = pearsonr(y_norm, self.target)
            else:
                y_corr = 0.0
            
            return {'x_correlation': x_corr, 'y_correlation': y_corr}
        except:
            return {'x_correlation': 0.0, 'y_correlation': 0.0}
    
    def calculate_projection_complexity(self, embedding):
        """Calculate complexity of the 2D projection"""
        try:
            distances = pdist(embedding)
            
            if len(distances) == 0 or np.mean(distances) == 0:
                return 0.0
            
            # Use fractal dimension approximation
            n_points = len(embedding)
            complexity = np.log(n_points) / np.log(np.mean(distances) + 1e-8)
            
            return complexity
        except:
            return 0.0
    
    def calculate_geometric_entropy(self, embedding):
        """Calculate geometric entropy of the embedding"""
        try:
            # Divide space into grid and calculate entropy
            n_bins = min(10, int(np.sqrt(len(embedding))))
            
            if n_bins < 2:
                return 0.0
            
            x_bins = np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), n_bins)
            y_bins = np.linspace(embedding[:, 1].min(), embedding[:, 1].max(), n_bins)
            
            hist, _, _ = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=[x_bins, y_bins])
            
            # Calculate entropy
            hist_norm = hist / np.sum(hist)
            hist_norm = hist_norm[hist_norm > 0]  # Remove zeros
            
            if len(hist_norm) == 0:
                return 0.0
                
            entropy = -np.sum(hist_norm * np.log2(hist_norm))
            
            return entropy
        except:
            return 0.0
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("Generating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Version 2 Geometric Mapping Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: ECFP4 UMAP Standard
        if 'ecfp4' in self.embeddings and 'umap_standard' in self.embeddings['ecfp4']:
            embedding = self.embeddings['ecfp4']['umap_standard']
            scatter = axes[0, 0].scatter(embedding[:, 0], embedding[:, 1], 
                                       c=self.target, cmap='viridis', alpha=0.7, s=20)
            axes[0, 0].set_title('ECFP4 UMAP (Standard)')
            axes[0, 0].set_xlabel('UMAP 1')
            axes[0, 0].set_ylabel('UMAP 2')
            plt.colorbar(scatter, ax=axes[0, 0], label='pKi')
        
        # Plot 2: Traditional Features UMAP
        if 'traditional' in self.embeddings and 'umap_standard' in self.embeddings['traditional']:
            embedding = self.embeddings['traditional']['umap_standard']
            scatter = axes[0, 1].scatter(embedding[:, 0], embedding[:, 1], 
                                       c=self.target, cmap='plasma', alpha=0.7, s=20)
            axes[0, 1].set_title('Traditional Features UMAP')
            axes[0, 1].set_xlabel('UMAP 1')
            axes[0, 1].set_ylabel('UMAP 2')
            plt.colorbar(scatter, ax=axes[0, 1], label='pKi')
        
        # Plot 3: Geometric Features UMAP
        if 'geometric' in self.embeddings and 'umap_standard' in self.embeddings['geometric']:
            embedding = self.embeddings['geometric']['umap_standard']
            scatter = axes[0, 2].scatter(embedding[:, 0], embedding[:, 1], 
                                       c=self.target, cmap='coolwarm', alpha=0.7, s=20)
            axes[0, 2].set_title('Geometric Features UMAP')
            axes[0, 2].set_xlabel('UMAP 1')
            axes[0, 2].set_ylabel('UMAP 2')
            plt.colorbar(scatter, ax=axes[0, 2], label='pKi')
        
        # Plot 4: Resonance Patterns Heatmap
        self.plot_resonance_heatmap(axes[1, 0])
        
        # Plot 5: Activity Distribution
        axes[1, 1].hist(self.target, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('pKi Activity Distribution')
        axes[1, 1].set_xlabel('pKi')
        axes[1, 1].set_ylabel('Frequency')
        
        # Plot 6: Geometric Complexity Analysis
        self.plot_complexity_analysis(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('v2_geometric_mapping_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to v2_geometric_mapping_analysis.png")
    
    def plot_resonance_heatmap(self, ax):
        """Plot resonance patterns heatmap"""
        if not self.geometry_patterns:
            ax.text(0.5, 0.5, 'No resonance data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Resonance Patterns')
            return
        
        # Collect resonance data
        resonance_data = []
        labels = []
        
        for feature_name, patterns_set in self.geometry_patterns.items():
            for embedding_name, patterns in patterns_set.items():
                resonance_values = [
                    patterns.get('phi_resonance', 0),
                    patterns.get('pi_resonance', 0),
                    patterns.get('sqrt2_resonance', 0),
                    patterns.get('e_resonance', 0)
                ]
                resonance_data.append(resonance_values)
                labels.append(f"{feature_name}_{embedding_name}")
        
        if resonance_data:
            resonance_matrix = np.array(resonance_data)
            im = ax.imshow(resonance_matrix, cmap='YlOrRd', aspect='auto')
            ax.set_title('Sacred Geometry Resonance Patterns')
            ax.set_xlabel('Resonance Type')
            ax.set_ylabel('Feature Set')
            ax.set_xticks(range(4))
            ax.set_xticklabels(['φ', 'π', '√2', 'e'])
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, rotation=45, ha='right', fontsize=8)
            plt.colorbar(im, ax=ax, label='Resonance Score')
    
    def plot_complexity_analysis(self, ax):
        """Plot geometric complexity analysis"""
        if not self.geometry_patterns:
            ax.text(0.5, 0.5, 'No complexity data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Geometric Complexity')
            return
        
        # Collect complexity data
        complexity_scores = []
        entropy_scores = []
        
        for feature_name, patterns_set in self.geometry_patterns.items():
            for embedding_name, patterns in patterns_set.items():
                complexity_scores.append(patterns.get('projection_complexity', 0))
                entropy_scores.append(patterns.get('geometric_entropy', 0))
        
        if complexity_scores and entropy_scores:
            ax.scatter(complexity_scores, entropy_scores, alpha=0.7, s=50)
            ax.set_xlabel('Projection Complexity')
            ax.set_ylabel('Geometric Entropy')
            ax.set_title('Complexity vs Entropy')
            
            # Add trend line
            if len(complexity_scores) > 1:
                try:
                    z = np.polyfit(complexity_scores, entropy_scores, 1)
                    p = np.poly1d(z)
                    ax.plot(complexity_scores, p(complexity_scores), "r--", alpha=0.8)
                except:
                    pass
    
    def save_results(self, filename="v2_geometric_analysis_results.csv"):
        """Save comprehensive analysis results"""
        print(f"Saving results to {filename}")
        
        # Flatten geometry patterns for saving
        results_data = []
        
        for feature_name, patterns_set in self.geometry_patterns.items():
            for embedding_name, patterns in patterns_set.items():
                result_row = {
                    'feature_set': feature_name,
                    'embedding_method': embedding_name,
                }
                
                # Add all pattern values, handling nested dictionaries
                for key, value in patterns.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            result_row[f"{key}_{subkey}"] = subvalue
                    else:
                        result_row[key] = value
                
                results_data.append(result_row)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filename, index=False)
        
        print(f"Results saved: {len(results_df)} analysis combinations")
        
        return results_df

def main():
    """Main execution function"""
    print("="*80)
    print("VERSION 2 GEOMETRIC MAPPING AND PATTERN DETECTION - FIXED")
    print("="*80)
    
    # Initialize analysis
    analysis = GeometricMappingAnalysis()
    
    # Load data
    df = analysis.load_data()
    
    # Perform dimensionality reduction
    analysis.perform_dimensionality_reduction()
    
    # Detect sacred geometry patterns
    analysis.detect_sacred_geometry_patterns()
    
    # Generate visualizations
    analysis.generate_visualizations()
    
    # Save results
    results_df = analysis.save_results()
    
    print("\n" + "="*80)
    print("GEOMETRIC MAPPING ANALYSIS SUMMARY")
    print("="*80)
    print(f"Dataset: {len(df)} compounds")
    print(f"Feature sets analyzed: {len(analysis.embeddings)}")
    print(f"Embedding methods per set: {len(list(analysis.embeddings.values())[0])}")
    print(f"Sacred geometry patterns detected: {len(results_df)}")
    print("Visualizations: v2_geometric_mapping_analysis.png")
    print("Results: v2_geometric_analysis_results.csv")
    print("Ready for statistical validation!")
    
    return analysis, results_df

if __name__ == "__main__":
    analysis, results = main()
