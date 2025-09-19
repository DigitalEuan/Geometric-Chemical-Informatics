#!/usr/bin/env python3
"""
UMAP and Correlation Analysis
"""

import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

class UmapCorrelation:
    """UMAP and correlation analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def perform_umap_analysis(self, df, morgan_fps):
        """Perform UMAP and return embeddings"""
        print("Performing UMAP analysis...")
        
        descriptor_cols = [col for col in df.columns if col not in 
                          ['canonical_smiles', 'molecule_chembl_id', 'standard_value', 
                           'target_chembl_id', 'pIC50']]
        
        descriptors = df[descriptor_cols].values
        combined_features = np.hstack([descriptors, morgan_fps])
        scaled_features = self.scaler.fit_transform(combined_features)
        
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
        
        indices1 = np.random.randint(0, n_points, sample_size)
        indices2 = np.random.randint(0, n_points, sample_size)
        
        mask = indices1 == indices2
        while np.any(mask):
            indices2[mask] = np.random.randint(0, n_points, np.sum(mask))
            mask = indices1 == indices2

        distances = np.linalg.norm(embedding[indices1] - embedding[indices2], axis=1)
        activity_diffs = np.abs(pic50_values[indices1] - pic50_values[indices2])
        
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

def main():
    """Main execution function"""
    print("Loading processed dataset...")
    df = pd.read_csv('kinase_compounds_features.csv')
    morgan_fps = np.load('morgan_fingerprints.npy')
    
    print(f"Dataset: {len(df)} compounds")
    
    analyzer = UmapCorrelation()
    
    embedding_2d, embedding_3d, features = analyzer.perform_umap_analysis(df, morgan_fps)
    
    pic50_values = df['pIC50'].values
    
    print("\n" + "="*60)
    print("DISTANCE-ACTIVITY CORRELATION ANALYSIS")
    print("="*60)
    
    corr_2d = analyzer.calculate_distance_activity_correlation(embedding_2d, pic50_values)
    corr_3d = analyzer.calculate_distance_activity_correlation(embedding_3d, pic50_values)
    
    print(f"2D Analysis:")
    print(f"  Spearman correlation: {corr_2d['spearman_correlation']:.4f} (p={corr_2d['spearman_p_value']:.2e})")
    print(f"  Pearson correlation: {corr_2d['pearson_correlation']:.4f} (p={corr_2d['pearson_p_value']:.2e})")
    
    print(f"3D Analysis:")
    print(f"  Spearman correlation: {corr_3d['spearman_correlation']:.4f} (p={corr_3d['spearman_p_value']:.2e})")
    print(f"  Pearson correlation: {corr_3d['pearson_correlation']:.4f} (p={corr_3d['pearson_p_value']:.2e})")
    
    # Save embeddings and features
    np.save('embedding_2d.npy', embedding_2d)
    np.save('embedding_3d.npy', embedding_3d)
    np.save('scaled_features.npy', features)
    
    print("\nEmbeddings and features saved.")

if __name__ == "__main__":
    main()
    main()
