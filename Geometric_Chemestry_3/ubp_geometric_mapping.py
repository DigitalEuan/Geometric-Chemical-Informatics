#!/usr/bin/env python3
"""
UBP Geometric Mapping and Resonance Detection Analysis
Phase 4: Create the "Periodic Neighborhood" map using UBP-enhanced geometric analysis

This system implements:
1. Multi-dimensional geometric mapping (UMAP, t-SNE, PCA)
2. Sacred geometry pattern detection
3. UBP resonance field analysis
4. Periodic Neighborhood visualization
5. Cross-realm coherence mapping
6. Fractal dimension analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import umap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

class UBPGeometricMapper:
    """UBP-enhanced geometric mapping and resonance detection system"""
    
    def __init__(self):
        # Sacred geometry constants for pattern detection
        self.sacred_constants = {
            'phi': (1 + np.sqrt(5)) / 2,      # Golden ratio
            'pi': np.pi,                      # Pi
            'e': np.e,                        # Euler's number
            'sqrt2': np.sqrt(2),              # Square root of 2
            'sqrt3': np.sqrt(3),              # Square root of 3
            'sqrt5': np.sqrt(5)               # Square root of 5
        }
        
        # UBP realm-specific wavelengths for visualization
        self.realm_colors = {
            'quantum': '#FF6B6B',           # Red (655 nm)
            'electromagnetic': '#4ECDC4',   # Cyan (635 nm)
            'gravitational': '#45B7D1',     # Blue (1000 nm)
            'biological': '#96CEB4',        # Green (700 nm)
            'cosmological': '#FFEAA7',      # Yellow (800 nm)
            'nuclear': '#DDA0DD',           # Purple (600 nm)
            'optical': '#98D8C8'            # Light green (600 nm)
        }
        
        # Geometric mapping results
        self.embeddings = {}
        self.resonance_patterns = {}
        self.periodic_neighborhoods = {}
        
    def perform_geometric_mapping(self, ubp_data_file: str):
        """Perform comprehensive geometric mapping analysis"""
        
        print("="*80)
        print("UBP GEOMETRIC MAPPING AND RESONANCE DETECTION")
        print("="*80)
        print("Phase 4: Creating the Periodic Neighborhood Map")
        print()
        
        # Load UBP-encoded data
        print(f"Loading UBP-encoded data from {ubp_data_file}...")
        try:
            df = pd.read_csv(ubp_data_file)
            print(f"✅ Loaded {len(df)} UBP-encoded materials with {len(df.columns)} features")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        # Prepare feature matrices
        print("\\nStep 1: Preparing feature matrices...")
        feature_matrices = self._prepare_feature_matrices(df)
        
        # Perform dimensionality reduction
        print("\\nStep 2: Performing multi-dimensional geometric mapping...")
        self._perform_dimensionality_reduction(df, feature_matrices)
        
        # Detect sacred geometry patterns
        print("\\nStep 3: Detecting sacred geometry patterns...")
        self._detect_sacred_patterns(df)
        
        # Analyze UBP resonance fields
        print("\\nStep 4: Analyzing UBP resonance fields...")
        self._analyze_resonance_fields(df)
        
        # Create Periodic Neighborhood map
        print("\\nStep 5: Creating Periodic Neighborhood visualization...")
        self._create_periodic_neighborhood_map(df)
        
        # Perform cross-realm coherence analysis
        print("\\nStep 6: Analyzing cross-realm coherence...")
        self._analyze_cross_realm_coherence(df)
        
        # Calculate fractal dimensions
        print("\\nStep 7: Calculating fractal dimensions...")
        self._calculate_fractal_dimensions(df)
        
        print("\\n✅ Geometric mapping and resonance detection complete!")
        
        return df
    
    def _prepare_feature_matrices(self, df):
        """Prepare different feature matrices for geometric mapping"""
        
        feature_matrices = {}
        
        try:
            # 1. UBP-specific features
            ubp_features = [col for col in df.columns if any(x in col for x in 
                           ['ubp', 'nrci', 'realm', 'coherence', 'toggle', 'offbit'])]
            if ubp_features:
                ubp_matrix = df[ubp_features].select_dtypes(include=[np.number]).fillna(0)
                feature_matrices['ubp'] = StandardScaler().fit_transform(ubp_matrix)
                print(f"  ✅ UBP features: {len(ubp_features)} features")
            
            # 2. Sacred geometry resonances
            resonance_features = [col for col in df.columns if 'resonance' in col]
            if resonance_features:
                resonance_matrix = df[resonance_features].fillna(0)
                feature_matrices['resonance'] = StandardScaler().fit_transform(resonance_matrix)
                print(f"  ✅ Resonance features: {len(resonance_features)} features")
            
            # 3. Crystallographic features
            crystal_features = [col for col in df.columns if any(x in col for x in 
                              ['spacegroup', 'crystal', 'lattice', 'symmetry', 'coordination'])]
            if crystal_features:
                crystal_matrix = df[crystal_features].select_dtypes(include=[np.number]).fillna(0)
                feature_matrices['crystallographic'] = StandardScaler().fit_transform(crystal_matrix)
                print(f"  ✅ Crystallographic features: {len(crystal_features)} features")
            
            # 4. Electronic properties
            electronic_features = [col for col in df.columns if any(x in col for x in 
                                  ['band_gap', 'magnetization', 'electronic', 'formation_energy'])]
            if electronic_features:
                electronic_matrix = df[electronic_features].select_dtypes(include=[np.number]).fillna(0)
                feature_matrices['electronic'] = StandardScaler().fit_transform(electronic_matrix)
                print(f"  ✅ Electronic features: {len(electronic_features)} features")
            
            # 5. Combined feature matrix
            all_numeric_features = df.select_dtypes(include=[np.number]).columns
            # Exclude ID columns and text columns
            exclude_cols = ['material_id', 'primary_realm_code', 'crystal_system_code']
            numeric_features = [col for col in all_numeric_features if col not in exclude_cols]
            
            if numeric_features:
                combined_matrix = df[numeric_features].fillna(0)
                feature_matrices['combined'] = StandardScaler().fit_transform(combined_matrix)
                print(f"  ✅ Combined features: {len(numeric_features)} features")
            
        except Exception as e:
            print(f"  ⚠️  Error preparing feature matrices: {e}")
            # Fallback to basic features
            basic_features = ['band_gap', 'total_magnetization', 'formation_energy_per_atom', 
                            'coordination_number', 'density']
            available_features = [f for f in basic_features if f in df.columns]
            if available_features:
                basic_matrix = df[available_features].fillna(0)
                feature_matrices['basic'] = StandardScaler().fit_transform(basic_matrix)
                print(f"  ✅ Fallback basic features: {len(available_features)} features")
        
        return feature_matrices
    
    def _perform_dimensionality_reduction(self, df, feature_matrices):
        """Perform multiple dimensionality reduction techniques"""
        
        self.embeddings = {}
        
        for matrix_name, matrix in feature_matrices.items():
            print(f"  Processing {matrix_name} features...")
            
            try:
                # UMAP embedding (primary method for UBP)
                umap_reducer = umap.UMAP(
                    n_neighbors=15,
                    min_dist=0.1,
                    n_components=2,
                    metric='euclidean',
                    random_state=42
                )
                umap_embedding = umap_reducer.fit_transform(matrix)
                self.embeddings[f'{matrix_name}_umap'] = umap_embedding
                
                # t-SNE embedding
                if matrix.shape[0] > 50:  # t-SNE needs sufficient samples
                    tsne_reducer = TSNE(
                        n_components=2,
                        perplexity=min(30, matrix.shape[0] // 4),
                        random_state=42,
                        n_iter=1000
                    )
                    tsne_embedding = tsne_reducer.fit_transform(matrix)
                    self.embeddings[f'{matrix_name}_tsne'] = tsne_embedding
                
                # PCA embedding
                pca_reducer = PCA(n_components=2, random_state=42)
                pca_embedding = pca_reducer.fit_transform(matrix)
                self.embeddings[f'{matrix_name}_pca'] = pca_embedding
                
                # Store explained variance for PCA
                if hasattr(pca_reducer, 'explained_variance_ratio_'):
                    explained_var = pca_reducer.explained_variance_ratio_
                    print(f"    PCA explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}")
                
            except Exception as e:
                print(f"    ⚠️  Error with {matrix_name}: {e}")
                continue
        
        print(f"  ✅ Generated {len(self.embeddings)} embeddings")
    
    def _detect_sacred_patterns(self, df):
        """Detect sacred geometry patterns in the embeddings"""
        
        self.resonance_patterns = {}
        
        for embedding_name, embedding in self.embeddings.items():
            print(f"  Analyzing {embedding_name}...")
            
            try:
                patterns = {}
                
                # Calculate pairwise distances
                from scipy.spatial.distance import pdist, squareform
                distances = squareform(pdist(embedding))
                
                # Detect sacred geometry ratios in distances
                for const_name, const_value in self.sacred_constants.items():
                    # Find distance ratios that match sacred constants
                    ratios = []
                    for i in range(len(distances)):
                        for j in range(i+1, len(distances)):
                            if distances[i,j] > 0:
                                for k in range(j+1, len(distances)):
                                    if distances[i,k] > 0:
                                        ratio = distances[i,j] / distances[i,k]
                                        # Check if ratio matches sacred constant (within tolerance)
                                        if abs(ratio - const_value) < 0.1 * const_value:
                                            ratios.append(ratio)
                    
                    if ratios:
                        patterns[f'{const_name}_ratio_count'] = len(ratios)
                        patterns[f'{const_name}_ratio_mean'] = np.mean(ratios)
                        patterns[f'{const_name}_ratio_std'] = np.std(ratios)
                    else:
                        patterns[f'{const_name}_ratio_count'] = 0
                        patterns[f'{const_name}_ratio_mean'] = 0
                        patterns[f'{const_name}_ratio_std'] = 0
                
                # Detect geometric clusters
                if len(embedding) > 10:
                    # K-means clustering
                    n_clusters = min(8, len(embedding) // 10)
                    if n_clusters >= 2:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = kmeans.fit_predict(embedding)
                        silhouette = silhouette_score(embedding, cluster_labels)
                        
                        patterns['cluster_count'] = n_clusters
                        patterns['cluster_silhouette'] = silhouette
                        patterns['cluster_centers'] = kmeans.cluster_centers_.tolist()
                
                # Calculate geometric properties
                patterns['embedding_span_x'] = np.max(embedding[:, 0]) - np.min(embedding[:, 0])
                patterns['embedding_span_y'] = np.max(embedding[:, 1]) - np.min(embedding[:, 1])
                patterns['embedding_aspect_ratio'] = patterns['embedding_span_x'] / patterns['embedding_span_y'] if patterns['embedding_span_y'] > 0 else 1.0
                
                # Check for sacred geometry in aspect ratio
                aspect_ratio = patterns['embedding_aspect_ratio']
                for const_name, const_value in self.sacred_constants.items():
                    if abs(aspect_ratio - const_value) < 0.1 * const_value:
                        patterns[f'aspect_ratio_{const_name}_match'] = True
                    else:
                        patterns[f'aspect_ratio_{const_name}_match'] = False
                
                self.resonance_patterns[embedding_name] = patterns
                
            except Exception as e:
                print(f"    ⚠️  Error detecting patterns in {embedding_name}: {e}")
                continue
        
        print(f"  ✅ Detected patterns in {len(self.resonance_patterns)} embeddings")
    
    def _analyze_resonance_fields(self, df):
        """Analyze UBP resonance fields across the geometric space"""
        
        print("  Analyzing UBP resonance field topology...")
        
        try:
            # Get the best embedding for analysis (prefer UBP features)
            best_embedding_name = None
            for name in self.embeddings.keys():
                if 'ubp' in name and 'umap' in name:
                    best_embedding_name = name
                    break
            
            if not best_embedding_name:
                best_embedding_name = list(self.embeddings.keys())[0]
            
            embedding = self.embeddings[best_embedding_name]
            
            # Create resonance field analysis
            resonance_analysis = {}
            
            # 1. NRCI field analysis
            if 'nrci_calculated' in df.columns:
                nrci_values = df['nrci_calculated'].values
                
                # Create NRCI field map
                from scipy.interpolate import griddata
                
                # Create grid for interpolation
                x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
                y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
                
                grid_x, grid_y = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
                
                # Interpolate NRCI values
                nrci_field = griddata(embedding, nrci_values, (grid_x, grid_y), method='cubic', fill_value=0)
                
                resonance_analysis['nrci_field'] = {
                    'grid_x': grid_x.tolist(),
                    'grid_y': grid_y.tolist(),
                    'field_values': nrci_field.tolist(),
                    'field_mean': float(np.nanmean(nrci_field)),
                    'field_std': float(np.nanstd(nrci_field)),
                    'field_max': float(np.nanmax(nrci_field)),
                    'field_min': float(np.nanmin(nrci_field))
                }
            
            # 2. UBP energy field analysis
            if 'ubp_energy_full' in df.columns:
                energy_values = df['ubp_energy_full'].values
                
                # Remove outliers for better visualization
                energy_q99 = np.percentile(energy_values, 99)
                energy_q1 = np.percentile(energy_values, 1)
                energy_clipped = np.clip(energy_values, energy_q1, energy_q99)
                
                energy_field = griddata(embedding, energy_clipped, (grid_x, grid_y), method='cubic', fill_value=0)
                
                resonance_analysis['energy_field'] = {
                    'field_values': energy_field.tolist(),
                    'field_mean': float(np.nanmean(energy_field)),
                    'field_std': float(np.nanstd(energy_field)),
                    'original_range': [float(energy_values.min()), float(energy_values.max())]
                }
            
            # 3. Resonance potential field
            if 'total_resonance_potential' in df.columns:
                resonance_values = df['total_resonance_potential'].values
                resonance_field = griddata(embedding, resonance_values, (grid_x, grid_y), method='cubic', fill_value=0)
                
                resonance_analysis['resonance_field'] = {
                    'field_values': resonance_field.tolist(),
                    'field_mean': float(np.nanmean(resonance_field)),
                    'field_std': float(np.nanstd(resonance_field))
                }
            
            # 4. Realm distribution analysis
            if 'primary_realm' in df.columns:
                realm_distribution = df['primary_realm'].value_counts().to_dict()
                resonance_analysis['realm_distribution'] = realm_distribution
                
                # Calculate realm centroids in embedding space
                realm_centroids = {}
                for realm in df['primary_realm'].unique():
                    realm_mask = df['primary_realm'] == realm
                    if np.sum(realm_mask) > 0:
                        centroid = np.mean(embedding[realm_mask], axis=0)
                        realm_centroids[realm] = centroid.tolist()
                
                resonance_analysis['realm_centroids'] = realm_centroids
            
            # Store analysis results
            self.resonance_patterns['field_analysis'] = resonance_analysis
            
            print(f"    ✅ Resonance field analysis complete")
            
        except Exception as e:
            print(f"    ⚠️  Error in resonance field analysis: {e}")
    
    def _create_periodic_neighborhood_map(self, df):
        """Create the main Periodic Neighborhood visualization"""
        
        print("  Creating Periodic Neighborhood map...")
        
        try:
            # Get the best embedding for the main map
            best_embedding_name = None
            for name in self.embeddings.keys():
                if 'ubp' in name and 'umap' in name:
                    best_embedding_name = name
                    break
            
            if not best_embedding_name:
                best_embedding_name = list(self.embeddings.keys())[0]
            
            embedding = self.embeddings[best_embedding_name]
            
            # Create interactive Periodic Neighborhood map
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Periodic Neighborhood Map', 'NRCI Distribution', 
                              'UBP Energy Field', 'Resonance Potential'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Main Periodic Neighborhood map (colored by realm)
            if 'primary_realm' in df.columns:
                for realm in df['primary_realm'].unique():
                    realm_mask = df['primary_realm'] == realm
                    if np.sum(realm_mask) > 0:
                        realm_embedding = embedding[realm_mask]
                        color = self.realm_colors.get(realm, '#888888')
                        
                        fig.add_trace(
                            go.Scatter(
                                x=realm_embedding[:, 0],
                                y=realm_embedding[:, 1],
                                mode='markers',
                                name=f'{realm.title()} Realm',
                                marker=dict(
                                    color=color,
                                    size=8,
                                    opacity=0.7,
                                    line=dict(width=1, color='white')
                                ),
                                text=[f"ID: {df.iloc[i]['material_id']}<br>Formula: {df.iloc[i]['formula']}" 
                                     for i in np.where(realm_mask)[0]],
                                hovertemplate='%{text}<br>Realm: ' + realm + '<extra></extra>'
                            ),
                            row=1, col=1
                        )
            
            # NRCI distribution
            if 'nrci_calculated' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        mode='markers',
                        name='NRCI',
                        marker=dict(
                            color=df['nrci_calculated'],
                            colorscale='Viridis',
                            size=6,
                            colorbar=dict(title="NRCI", x=0.48),
                            showscale=True
                        ),
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # UBP Energy field
            if 'ubp_energy_full' in df.columns:
                # Clip extreme values for better visualization
                energy_values = df['ubp_energy_full'].values
                energy_clipped = np.clip(energy_values, 
                                       np.percentile(energy_values, 1),
                                       np.percentile(energy_values, 99))
                
                fig.add_trace(
                    go.Scatter(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        mode='markers',
                        name='UBP Energy',
                        marker=dict(
                            color=energy_clipped,
                            colorscale='Plasma',
                            size=6,
                            colorbar=dict(title="UBP Energy", x=1.02),
                            showscale=True
                        ),
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Resonance potential
            if 'total_resonance_potential' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        mode='markers',
                        name='Resonance',
                        marker=dict(
                            color=df['total_resonance_potential'],
                            colorscale='Cividis',
                            size=6,
                            colorbar=dict(title="Resonance", x=1.02, y=0.25),
                            showscale=True
                        ),
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="UBP Periodic Neighborhood Map - Inorganic Materials",
                height=800,
                showlegend=True,
                legend=dict(x=0.02, y=0.98)
            )
            
            # Update axes
            for i in range(1, 3):
                for j in range(1, 3):
                    fig.update_xaxes(title="Dimension 1", row=i, col=j)
                    fig.update_yaxes(title="Dimension 2", row=i, col=j)
            
            # Save interactive map
            fig.write_html("ubp_periodic_neighborhood_map.html")
            print(f"    ✅ Saved interactive map to ubp_periodic_neighborhood_map.html")
            
            # Create static version for overview
            plt.figure(figsize=(15, 10))
            
            # Main plot
            plt.subplot(2, 3, 1)
            if 'primary_realm' in df.columns:
                for realm in df['primary_realm'].unique():
                    realm_mask = df['primary_realm'] == realm
                    if np.sum(realm_mask) > 0:
                        realm_embedding = embedding[realm_mask]
                        color = self.realm_colors.get(realm, '#888888')
                        plt.scatter(realm_embedding[:, 0], realm_embedding[:, 1], 
                                  c=color, label=f'{realm.title()}', alpha=0.7, s=30)
            plt.title('Periodic Neighborhood Map')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
            
            # NRCI plot
            plt.subplot(2, 3, 2)
            if 'nrci_calculated' in df.columns:
                scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                    c=df['nrci_calculated'], cmap='viridis', s=30, alpha=0.7)
                plt.colorbar(scatter, label='NRCI')
            plt.title('NRCI Distribution')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
            # Energy plot
            plt.subplot(2, 3, 3)
            if 'ubp_energy_full' in df.columns:
                energy_clipped = np.clip(df['ubp_energy_full'].values,
                                       np.percentile(df['ubp_energy_full'].values, 1),
                                       np.percentile(df['ubp_energy_full'].values, 99))
                scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                    c=energy_clipped, cmap='plasma', s=30, alpha=0.7)
                plt.colorbar(scatter, label='UBP Energy')
            plt.title('UBP Energy Field')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
            # Resonance plot
            plt.subplot(2, 3, 4)
            if 'total_resonance_potential' in df.columns:
                scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                    c=df['total_resonance_potential'], cmap='cividis', s=30, alpha=0.7)
                plt.colorbar(scatter, label='Resonance')
            plt.title('Resonance Potential')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
            # Coherence plot
            plt.subplot(2, 3, 5)
            if 'system_coherence' in df.columns:
                scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                    c=df['system_coherence'], cmap='coolwarm', s=30, alpha=0.7)
                plt.colorbar(scatter, label='Coherence')
            plt.title('System Coherence')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
            # Quality score plot
            plt.subplot(2, 3, 6)
            if 'ubp_quality_score' in df.columns:
                scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                    c=df['ubp_quality_score'], cmap='RdYlGn', s=30, alpha=0.7)
                plt.colorbar(scatter, label='Quality')
            plt.title('UBP Quality Score')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
            plt.tight_layout()
            plt.savefig('ubp_periodic_neighborhood_overview.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Saved overview plot to ubp_periodic_neighborhood_overview.png")
            
        except Exception as e:
            print(f"    ⚠️  Error creating Periodic Neighborhood map: {e}")
    
    def _analyze_cross_realm_coherence(self, df):
        """Analyze coherence patterns across UBP realms"""
        
        print("  Analyzing cross-realm coherence patterns...")
        
        try:
            coherence_analysis = {}
            
            # Get realm-specific coherence values
            realm_coherences = {}
            for realm in ['quantum', 'electromagnetic', 'gravitational', 'biological', 
                         'cosmological', 'nuclear', 'optical']:
                coherence_col = f'{realm}_coherence'
                if coherence_col in df.columns:
                    realm_coherences[realm] = df[coherence_col].values
            
            if len(realm_coherences) > 1:
                # Calculate cross-realm correlations
                coherence_matrix = np.array(list(realm_coherences.values())).T
                correlation_matrix = np.corrcoef(coherence_matrix.T)
                
                coherence_analysis['cross_realm_correlations'] = {
                    'matrix': correlation_matrix.tolist(),
                    'realms': list(realm_coherences.keys()),
                    'mean_correlation': float(np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])),
                    'max_correlation': float(np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])),
                    'min_correlation': float(np.min(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
                }
                
                # Identify highly coherent realm pairs
                high_coherence_pairs = []
                realm_names = list(realm_coherences.keys())
                for i in range(len(realm_names)):
                    for j in range(i+1, len(realm_names)):
                        correlation = correlation_matrix[i, j]
                        if correlation > 0.95:  # UBP target
                            high_coherence_pairs.append({
                                'realm1': realm_names[i],
                                'realm2': realm_names[j],
                                'correlation': float(correlation)
                            })
                
                coherence_analysis['high_coherence_pairs'] = high_coherence_pairs
                
                print(f"    ✅ Cross-realm analysis: {len(high_coherence_pairs)} high-coherence pairs found")
            
            # Analyze coherence vs. material properties
            if 'system_coherence' in df.columns:
                properties = ['band_gap', 'total_magnetization', 'formation_energy_per_atom', 
                            'coordination_number', 'density']
                
                property_correlations = {}
                for prop in properties:
                    if prop in df.columns:
                        correlation = np.corrcoef(df['system_coherence'], df[prop])[0, 1]
                        if not np.isnan(correlation):
                            property_correlations[prop] = float(correlation)
                
                coherence_analysis['property_correlations'] = property_correlations
            
            self.resonance_patterns['coherence_analysis'] = coherence_analysis
            
        except Exception as e:
            print(f"    ⚠️  Error in cross-realm coherence analysis: {e}")
    
    def _calculate_fractal_dimensions(self, df):
        """Calculate fractal dimensions of the geometric embeddings"""
        
        print("  Calculating fractal dimensions...")
        
        try:
            fractal_analysis = {}
            
            for embedding_name, embedding in self.embeddings.items():
                if len(embedding) < 10:
                    continue
                
                try:
                    # Box-counting method for fractal dimension
                    def box_count(points, box_size):
                        """Count boxes containing points"""
                        x_min, x_max = points[:, 0].min(), points[:, 0].max()
                        y_min, y_max = points[:, 1].min(), points[:, 1].max()
                        
                        x_bins = int((x_max - x_min) / box_size) + 1
                        y_bins = int((y_max - y_min) / box_size) + 1
                        
                        # Create grid
                        x_edges = np.linspace(x_min, x_max, x_bins + 1)
                        y_edges = np.linspace(y_min, y_max, y_bins + 1)
                        
                        # Count occupied boxes
                        hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], 
                                                  bins=[x_edges, y_edges])
                        
                        return np.sum(hist > 0)
                    
                    # Calculate box counts for different scales
                    span_x = embedding[:, 0].max() - embedding[:, 0].min()
                    span_y = embedding[:, 1].max() - embedding[:, 1].min()
                    max_span = max(span_x, span_y)
                    
                    box_sizes = np.logspace(-2, 0, 10) * max_span
                    box_counts = []
                    
                    for box_size in box_sizes:
                        if box_size > 0:
                            count = box_count(embedding, box_size)
                            box_counts.append(count)
                    
                    # Fit power law to get fractal dimension
                    if len(box_counts) > 3:
                        log_sizes = np.log(1 / box_sizes[:len(box_counts)])
                        log_counts = np.log(box_counts)
                        
                        # Linear regression
                        coeffs = np.polyfit(log_sizes, log_counts, 1)
                        fractal_dim = coeffs[0]
                        
                        fractal_analysis[embedding_name] = {
                            'fractal_dimension': float(fractal_dim),
                            'box_sizes': box_sizes[:len(box_counts)].tolist(),
                            'box_counts': box_counts,
                            'r_squared': float(np.corrcoef(log_sizes, log_counts)[0, 1] ** 2)
                        }
                        
                        print(f"    {embedding_name}: D = {fractal_dim:.3f}")
                
                except Exception as e:
                    print(f"    ⚠️  Error calculating fractal dimension for {embedding_name}: {e}")
                    continue
            
            self.resonance_patterns['fractal_analysis'] = fractal_analysis
            
            # Check if fractal dimensions match UBP target (~2.3)
            target_fractal_dim = 2.3
            matching_embeddings = []
            
            for name, analysis in fractal_analysis.items():
                fractal_dim = analysis['fractal_dimension']
                if abs(fractal_dim - target_fractal_dim) < 0.3:
                    matching_embeddings.append(name)
            
            if matching_embeddings:
                print(f"    ✅ {len(matching_embeddings)} embeddings match UBP target fractal dimension (~2.3)")
            
        except Exception as e:
            print(f"    ⚠️  Error in fractal dimension calculation: {e}")
    
    def save_geometric_analysis_results(self, df, filename="ubp_geometric_analysis_results.json"):
        """Save all geometric analysis results"""
        
        print(f"\\n💾 Saving geometric analysis results...")
        
        try:
            # Prepare results for JSON serialization
            results = {
                'embeddings_info': {
                    name: {
                        'shape': embedding.shape,
                        'mean': embedding.mean(axis=0).tolist(),
                        'std': embedding.std(axis=0).tolist(),
                        'range': [embedding.min(axis=0).tolist(), embedding.max(axis=0).tolist()]
                    }
                    for name, embedding in self.embeddings.items()
                },
                'resonance_patterns': self.resonance_patterns,
                'analysis_summary': {
                    'total_materials': len(df),
                    'embeddings_generated': len(self.embeddings),
                    'patterns_detected': len(self.resonance_patterns),
                    'sacred_constants_used': self.sacred_constants,
                    'realm_colors': self.realm_colors
                }
            }
            
            # Save main results
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save embeddings separately (numpy arrays)
            embeddings_filename = filename.replace('.json', '_embeddings.npz')
            np.savez_compressed(embeddings_filename, **self.embeddings)
            
            # Create summary report
            summary_filename = filename.replace('.json', '_summary.txt')
            with open(summary_filename, 'w') as f:
                f.write("UBP GEOMETRIC ANALYSIS SUMMARY\\n")
                f.write("="*50 + "\\n\\n")
                
                f.write(f"Materials analyzed: {len(df)}\\n")
                f.write(f"Embeddings generated: {len(self.embeddings)}\\n")
                f.write(f"Pattern analyses: {len(self.resonance_patterns)}\\n\\n")
                
                # Embedding summary
                f.write("EMBEDDINGS:\\n")
                for name, embedding in self.embeddings.items():
                    f.write(f"  {name}: {embedding.shape[0]} points in {embedding.shape[1]}D\\n")
                
                # Pattern summary
                f.write("\\nPATTERN DETECTION:\\n")
                for pattern_name, patterns in self.resonance_patterns.items():
                    if isinstance(patterns, dict):
                        f.write(f"  {pattern_name}: {len(patterns)} metrics\\n")
                
                # Sacred geometry summary
                f.write("\\nSACRED GEOMETRY CONSTANTS:\\n")
                for name, value in self.sacred_constants.items():
                    f.write(f"  {name}: {value:.6f}\\n")
            
            print(f"✅ Saved analysis results to {filename}")
            print(f"✅ Saved embeddings to {embeddings_filename}")
            print(f"✅ Saved summary to {summary_filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main execution function"""
    
    print("Starting UBP geometric mapping and resonance detection...")
    print("Phase 4: Creating the Periodic Neighborhood Map")
    print()
    
    # Initialize geometric mapper
    mapper = UBPGeometricMapper()
    
    # Perform geometric mapping analysis
    df = mapper.perform_geometric_mapping("ubp_encoded_inorganic_materials.csv")
    
    if df is not None:
        # Save analysis results
        mapper.save_geometric_analysis_results(df)
        
        print("\\n" + "="*80)
        print("PHASE 4 COMPLETE")
        print("="*80)
        print(f"✅ Geometric mapping complete for {len(df)} materials")
        print(f"✅ Generated {len(mapper.embeddings)} dimensional embeddings")
        print(f"✅ Detected patterns in {len(mapper.resonance_patterns)} analyses")
        print("✅ Created Periodic Neighborhood visualization")
        print("✅ Ready for Phase 5: UBP principles validation")
        
        return df, mapper
    else:
        print("❌ Geometric mapping failed")
        return None, None

if __name__ == "__main__":
    dataset, geometric_mapper = main()
