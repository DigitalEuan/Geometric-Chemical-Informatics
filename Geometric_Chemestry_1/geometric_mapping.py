#!/usr/bin/env python3
"""
Geometric Mapping and Sacred Geometry Analysis for Chemical Space
Implements UMAP projection and resonance pattern detection
"""

import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

class GeometricMapper:
    """Advanced geometric mapping and pattern analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.umap_2d = None
        self.umap_3d = None
        self.embedding_2d = None
        self.embedding_3d = None
        
        # Sacred geometry constants
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.pi = np.pi
        self.sqrt2 = np.sqrt(2)
        self.sqrt3 = np.sqrt(3)
        self.e = np.e
        
        # Resonance tolerance
        self.tolerance = 0.1
        
    def prepare_features(self, df, morgan_fps):
        """Prepare combined feature matrix for dimensionality reduction"""
        print("Preparing feature matrix...")
        
        # Get molecular descriptor columns
        descriptor_cols = [col for col in df.columns if col not in 
                          ['canonical_smiles', 'molecule_chembl_id', 'standard_value', 
                           'target_chembl_id', 'pIC50']]
        
        # Extract descriptors
        descriptors = df[descriptor_cols].values
        
        # Combine descriptors with Morgan fingerprints
        combined_features = np.hstack([descriptors, morgan_fps])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(combined_features)
        
        print(f"Feature matrix shape: {scaled_features.shape}")
        return scaled_features
    
    def perform_umap_projection(self, features, n_neighbors=15, min_dist=0.1, random_state=42):
        """Perform UMAP dimensionality reduction for 2D and 3D"""
        print("Performing UMAP dimensionality reduction...")
        
        # 2D UMAP
        self.umap_2d = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
            metric='euclidean'
        )
        self.embedding_2d = self.umap_2d.fit_transform(features)
        
        # 3D UMAP
        self.umap_3d = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=3,
            random_state=random_state,
            metric='euclidean'
        )
        self.embedding_3d = self.umap_3d.fit_transform(features)
        
        print(f"2D embedding shape: {self.embedding_2d.shape}")
        print(f"3D embedding shape: {self.embedding_3d.shape}")
        
        return self.embedding_2d, self.embedding_3d
    
    def detect_resonance_distances(self, embedding, target_values):
        """Detect pairs of compounds with resonant distances"""
        print("Detecting resonance patterns...")
        
        # Calculate pairwise distances
        distances = pdist(embedding)
        distance_matrix = squareform(distances)
        
        # Define resonance targets
        resonance_targets = {
            'pi': self.pi,
            'phi': self.phi,
            'sqrt2': self.sqrt2,
            'sqrt3': self.sqrt3,
            'e': self.e,
            'pi/2': self.pi/2,
            'phi^2': self.phi**2,
            '2*pi': 2*self.pi
        }
        
        resonant_pairs = {}
        
        for name, target in resonance_targets.items():
            pairs = []
            
            # Find pairs with distances close to target
            for i in range(len(embedding)):
                for j in range(i+1, len(embedding)):
                    dist = distance_matrix[i, j]
                    if abs(dist - target) < self.tolerance:
                        pairs.append({
                            'compound1_idx': i,
                            'compound2_idx': j,
                            'distance': dist,
                            'target_distance': target,
                            'deviation': abs(dist - target),
                            'activity1': target_values[i],
                            'activity2': target_values[j],
                            'activity_diff': abs(target_values[i] - target_values[j])
                        })
            
            if pairs:
                resonant_pairs[name] = pairs
                print(f"Found {len(pairs)} {name} resonant pairs")
        
        return resonant_pairs, distance_matrix
    
    def detect_geometric_patterns(self, embedding):
        """Detect geometric patterns like triangles and polygons"""
        print("Detecting geometric patterns...")
        
        patterns = {
            'equilateral_triangles': [],
            'right_triangles': [],
            'squares': [],
            'pentagons': []
        }
        
        n_points = len(embedding)
        
        # Check triangles
        for i in range(n_points):
            for j in range(i+1, n_points):
                for k in range(j+1, n_points):
                    # Calculate triangle side lengths
                    d1 = np.linalg.norm(embedding[i] - embedding[j])
                    d2 = np.linalg.norm(embedding[j] - embedding[k])
                    d3 = np.linalg.norm(embedding[k] - embedding[i])
                    
                    sides = sorted([d1, d2, d3])
                    
                    # Check for equilateral triangle
                    if abs(sides[0] - sides[1]) < self.tolerance and abs(sides[1] - sides[2]) < self.tolerance:
                        patterns['equilateral_triangles'].append({
                            'points': [i, j, k],
                            'side_length': np.mean(sides),
                            'coordinates': [embedding[i], embedding[j], embedding[k]]
                        })
                    
                    # Check for right triangle (Pythagorean theorem)
                    if abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < self.tolerance:
                        patterns['right_triangles'].append({
                            'points': [i, j, k],
                            'sides': sides,
                            'coordinates': [embedding[i], embedding[j], embedding[k]]
                        })
        
        print(f"Found {len(patterns['equilateral_triangles'])} equilateral triangles")
        print(f"Found {len(patterns['right_triangles'])} right triangles")
        
        return patterns
    
    def analyze_activity_correlation(self, embedding, target_values):
        """Analyze correlation between geometric distance and activity similarity"""
        print("Analyzing activity-distance correlation...")
        
        # Calculate pairwise distances in embedding space
        distances = pdist(embedding)
        
        # Calculate pairwise activity differences
        activity_diffs = []
        for i in range(len(target_values)):
            for j in range(i+1, len(target_values)):
                activity_diffs.append(abs(target_values[i] - target_values[j]))
        
        activity_diffs = np.array(activity_diffs)
        
        # Calculate correlations
        spearman_corr, spearman_p = spearmanr(distances, activity_diffs)
        pearson_corr, pearson_p = pearsonr(distances, activity_diffs)
        
        correlation_results = {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'n_pairs': len(distances)
        }
        
        print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
        print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
        
        return correlation_results, distances, activity_diffs
    
    def create_interactive_plots(self, df, embedding_2d, embedding_3d, resonant_pairs):
        """Create interactive visualizations"""
        print("Creating interactive visualizations...")
        
        # Prepare data
        pIC50_values = df['pIC50'].values
        target_names = df['target_chembl_id'].values
        
        # Create color scale based on pIC50
        colors = pIC50_values
        
        # 2D Plot
        fig_2d = go.Figure()
        
        # Add main scatter plot
        fig_2d.add_trace(go.Scatter(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title="pIC50"),
                line=dict(width=0.5, color='white')
            ),
            text=[f"pIC50: {pic50:.2f}<br>Target: {target}" 
                  for pic50, target in zip(pIC50_values, target_names)],
            hovertemplate='<b>Compound</b><br>' +
                         'UMAP1: %{x:.3f}<br>' +
                         'UMAP2: %{y:.3f}<br>' +
                         '%{text}<br>' +
                         '<extra></extra>',
            name='Compounds'
        ))
        
        # Add resonant pairs as lines
        colors_resonance = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, (resonance_type, pairs) in enumerate(resonant_pairs.items()):
            if i < len(colors_resonance):
                for pair in pairs[:10]:  # Limit to first 10 pairs for clarity
                    idx1, idx2 = pair['compound1_idx'], pair['compound2_idx']
                    fig_2d.add_trace(go.Scatter(
                        x=[embedding_2d[idx1, 0], embedding_2d[idx2, 0]],
                        y=[embedding_2d[idx1, 1], embedding_2d[idx2, 1]],
                        mode='lines',
                        line=dict(color=colors_resonance[i], width=2, dash='dash'),
                        name=f'{resonance_type} resonance',
                        showlegend=(pair == pairs[0])  # Only show legend for first pair
                    ))
        
        fig_2d.update_layout(
            title='2D UMAP Projection of Chemical Space with Resonance Patterns',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            width=800,
            height=600
        )
        
        # 3D Plot
        fig_3d = go.Figure()
        
        fig_3d.add_trace(go.Scatter3d(
            x=embedding_3d[:, 0],
            y=embedding_3d[:, 1],
            z=embedding_3d[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title="pIC50"),
                line=dict(width=0.5, color='white')
            ),
            text=[f"pIC50: {pic50:.2f}<br>Target: {target}" 
                  for pic50, target in zip(pIC50_values, target_names)],
            hovertemplate='<b>Compound</b><br>' +
                         'UMAP1: %{x:.3f}<br>' +
                         'UMAP2: %{y:.3f}<br>' +
                         'UMAP3: %{z:.3f}<br>' +
                         '%{text}<br>' +
                         '<extra></extra>',
            name='Compounds'
        ))
        
        fig_3d.update_layout(
            title='3D UMAP Projection of Chemical Space',
            scene=dict(
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                zaxis_title='UMAP Dimension 3'
            ),
            width=800,
            height=600
        )
        
        return fig_2d, fig_3d
    
    def generate_summary_report(self, df, resonant_pairs, correlation_results, patterns):
        """Generate comprehensive analysis summary"""
        report = {
            'dataset_summary': {
                'total_compounds': len(df),
                'unique_targets': df['target_chembl_id'].nunique(),
                'pic50_range': [df['pIC50'].min(), df['pIC50'].max()],
                'pic50_mean': df['pIC50'].mean(),
                'pic50_std': df['pIC50'].std()
            },
            'resonance_analysis': {
                'total_resonant_pairs': sum(len(pairs) for pairs in resonant_pairs.values()),
                'resonance_types': list(resonant_pairs.keys()),
                'pairs_per_type': {k: len(v) for k, v in resonant_pairs.items()}
            },
            'correlation_analysis': correlation_results,
            'geometric_patterns': {
                'equilateral_triangles': len(patterns['equilateral_triangles']),
                'right_triangles': len(patterns['right_triangles'])
            }
        }
        
        return report

def main():
    """Main execution function"""
    print("Loading processed dataset...")
    df = pd.read_csv('kinase_compounds_features.csv')
    morgan_fps = np.load('morgan_fingerprints.npy')
    
    print(f"Dataset: {len(df)} compounds with {morgan_fps.shape[1]} fingerprint bits")
    
    # Initialize mapper
    mapper = GeometricMapper()
    
    # Prepare features
    features = mapper.prepare_features(df, morgan_fps)
    
    # Perform UMAP projection
    embedding_2d, embedding_3d = mapper.perform_umap_projection(features)
    
    # Detect resonance patterns
    resonant_pairs_2d, distance_matrix_2d = mapper.detect_resonance_distances(
        embedding_2d, df['pIC50'].values
    )
    resonant_pairs_3d, distance_matrix_3d = mapper.detect_resonance_distances(
        embedding_3d, df['pIC50'].values
    )
    
    # Detect geometric patterns
    patterns_2d = mapper.detect_geometric_patterns(embedding_2d)
    patterns_3d = mapper.detect_geometric_patterns(embedding_3d)
    
    # Analyze correlations
    corr_2d, distances_2d, activity_diffs_2d = mapper.analyze_activity_correlation(
        embedding_2d, df['pIC50'].values
    )
    corr_3d, distances_3d, activity_diffs_3d = mapper.analyze_activity_correlation(
        embedding_3d, df['pIC50'].values
    )
    
    # Create visualizations
    fig_2d, fig_3d = mapper.create_interactive_plots(
        df, embedding_2d, embedding_3d, resonant_pairs_2d
    )
    
    # Save plots
    fig_2d.write_html('chemical_space_2d.html')
    fig_3d.write_html('chemical_space_3d.html')
    print("Saved interactive plots to HTML files")
    
    # Generate summary report
    report_2d = mapper.generate_summary_report(df, resonant_pairs_2d, corr_2d, patterns_2d)
    report_3d = mapper.generate_summary_report(df, resonant_pairs_3d, corr_3d, patterns_3d)
    
    # Save embeddings and results
    np.save('umap_2d_embedding.npy', embedding_2d)
    np.save('umap_3d_embedding.npy', embedding_3d)
    
    # Add embeddings to dataframe
    df_with_embeddings = df.copy()
    df_with_embeddings['UMAP1'] = embedding_2d[:, 0]
    df_with_embeddings['UMAP2'] = embedding_2d[:, 1]
    df_with_embeddings['UMAP3_1'] = embedding_3d[:, 0]
    df_with_embeddings['UMAP3_2'] = embedding_3d[:, 1]
    df_with_embeddings['UMAP3_3'] = embedding_3d[:, 2]
    
    df_with_embeddings.to_csv('kinase_compounds_with_embeddings.csv', index=False)
    print("Saved dataset with embeddings")
    
    # Print summary
    print("\n" + "="*60)
    print("GEOMETRIC MAPPING ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n2D Analysis:")
    print(f"  Resonant pairs found: {sum(len(pairs) for pairs in resonant_pairs_2d.values())}")
    print(f"  Spearman correlation: {corr_2d['spearman_correlation']:.4f} (p={corr_2d['spearman_p_value']:.2e})")
    print(f"  Equilateral triangles: {len(patterns_2d['equilateral_triangles'])}")
    
    print(f"\n3D Analysis:")
    print(f"  Resonant pairs found: {sum(len(pairs) for pairs in resonant_pairs_3d.values())}")
    print(f"  Spearman correlation: {corr_3d['spearman_correlation']:.4f} (p={corr_3d['spearman_p_value']:.2e})")
    print(f"  Equilateral triangles: {len(patterns_3d['equilateral_triangles'])}")
    
    print(f"\nResonance breakdown (2D):")
    for resonance_type, pairs in resonant_pairs_2d.items():
        if pairs:
            avg_activity_diff = np.mean([p['activity_diff'] for p in pairs])
            print(f"  {resonance_type}: {len(pairs)} pairs, avg activity diff: {avg_activity_diff:.3f}")

if __name__ == "__main__":
    main()
