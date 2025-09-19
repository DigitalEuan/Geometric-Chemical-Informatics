#!/usr/bin/env python3
"""
Geometric Computation Framework: 2D Projection as Novel Computational Paradigm
Exploring how 2D geometric arrangements can serve as computational substrates
"""

import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class GeometricComputationEngine:
    """Novel computational framework using 2D geometric projections"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.pi = np.pi
        self.sqrt2 = np.sqrt(2)
        self.e = np.e
        
        # Computational geometry constants
        self.resonance_frequencies = {
            'phi': self.phi,
            'pi': self.pi,
            'sqrt2': self.sqrt2,
            'e': self.e,
            'phi_squared': self.phi**2,
            'pi_half': self.pi/2,
            'golden_angle': 2*self.pi/self.phi**2
        }
        
    def create_geometric_substrate(self, embedding_2d, pic50_values):
        """Create a 2D geometric substrate for computation"""
        print("Creating geometric computational substrate...")
        
        # Normalize coordinates to unit square for computational consistency
        x_min, x_max = embedding_2d[:, 0].min(), embedding_2d[:, 0].max()
        y_min, y_max = embedding_2d[:, 1].min(), embedding_2d[:, 1].max()
        
        normalized_coords = np.column_stack([
            (embedding_2d[:, 0] - x_min) / (x_max - x_min),
            (embedding_2d[:, 1] - y_min) / (y_max - y_min)
        ])
        
        # Create computational substrate
        substrate = {
            'coordinates': normalized_coords,
            'values': pic50_values,
            'n_points': len(normalized_coords),
            'bounds': [(0, 1), (0, 1)],
            'density': len(normalized_coords) / 1.0  # points per unit area
        }
        
        return substrate
    
    def implement_geometric_operations(self, substrate):
        """Implement computational operations using geometric arrangements"""
        print("Implementing geometric computational operations...")
        
        coords = substrate['coordinates']
        values = substrate['values']
        
        operations = {}
        
        # 1. Voronoi-based computation
        print("  Computing Voronoi tessellation...")
        vor = Voronoi(coords)
        operations['voronoi'] = {
            'tessellation': vor,
            'cell_areas': self._calculate_voronoi_areas(vor, coords),
            'neighbor_graph': self._build_voronoi_neighbor_graph(vor)
        }
        
        # 2. Convex hull computation
        print("  Computing convex hull...")
        hull = ConvexHull(coords)
        operations['convex_hull'] = {
            'hull': hull,
            'boundary_points': hull.vertices,
            'interior_points': np.setdiff1d(np.arange(len(coords)), hull.vertices),
            'hull_area': hull.volume  # In 2D, volume is area
        }
        
        # 3. Distance-based field computation
        print("  Computing distance fields...")
        distance_matrix = squareform(pdist(coords))
        operations['distance_fields'] = {
            'matrix': distance_matrix,
            'nearest_neighbors': np.argsort(distance_matrix, axis=1)[:, 1:6],  # 5 nearest
            'local_density': self._calculate_local_density(distance_matrix)
        }
        
        # 4. Resonance-based computation
        print("  Computing resonance patterns...")
        operations['resonance_computation'] = self._compute_resonance_patterns(
            coords, values, distance_matrix
        )
        
        # 5. Geometric flow computation
        print("  Computing geometric flows...")
        operations['geometric_flows'] = self._compute_geometric_flows(coords, values)
        
        return operations
    
    def _calculate_voronoi_areas(self, vor, coords):
        """Calculate areas of Voronoi cells"""
        areas = []
        for i, point_region in enumerate(vor.point_region):
            region = vor.regions[point_region]
            if -1 not in region and len(region) > 0:
                polygon = vor.vertices[region]
                # Calculate area using shoelace formula
                area = 0.5 * abs(sum(polygon[i,0]*(polygon[i+1,1]-polygon[i-1,1]) 
                                   for i in range(len(polygon))))
                areas.append(area)
            else:
                areas.append(np.inf)  # Infinite area for unbounded cells
        return np.array(areas)
    
    def _build_voronoi_neighbor_graph(self, vor):
        """Build neighbor graph from Voronoi tessellation"""
        neighbor_graph = {}
        for i, point_region in enumerate(vor.point_region):
            neighbors = set()
            region = vor.regions[point_region]
            if -1 not in region:
                for ridge in vor.ridge_points:
                    if i in ridge:
                        neighbor = ridge[0] if ridge[1] == i else ridge[1]
                        neighbors.add(neighbor)
            neighbor_graph[i] = list(neighbors)
        return neighbor_graph
    
    def _calculate_local_density(self, distance_matrix, k=5):
        """Calculate local density using k-nearest neighbors"""
        densities = []
        for i in range(len(distance_matrix)):
            # Get k nearest neighbors (excluding self)
            nearest_distances = np.sort(distance_matrix[i])[1:k+1]
            # Density is inverse of mean distance to k nearest neighbors
            density = 1.0 / (np.mean(nearest_distances) + 1e-10)
            densities.append(density)
        return np.array(densities)
    
    def _compute_resonance_patterns(self, coords, values, distance_matrix):
        """Compute resonance-based computational patterns"""
        resonance_results = {}
        
        for name, target_distance in self.resonance_frequencies.items():
            # Find pairs with resonant distances
            resonant_pairs = []
            resonant_values = []
            
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    dist = distance_matrix[i, j]
                    if abs(dist - target_distance) < 0.1:  # tolerance
                        resonant_pairs.append((i, j))
                        # Compute resonance "energy"
                        energy = values[i] * values[j] / (dist + 1e-10)
                        resonant_values.append(energy)
            
            resonance_results[name] = {
                'pairs': resonant_pairs,
                'energies': resonant_values,
                'total_energy': sum(resonant_values),
                'n_resonances': len(resonant_pairs)
            }
        
        return resonance_results
    
    def _compute_geometric_flows(self, coords, values):
        """Compute geometric flows based on value gradients"""
        flows = []
        
        for i, point in enumerate(coords):
            # Calculate local gradient
            neighbors = []
            neighbor_values = []
            
            # Find nearby points within radius
            for j, other_point in enumerate(coords):
                if i != j:
                    dist = np.linalg.norm(point - other_point)
                    if dist < 0.1:  # local neighborhood
                        neighbors.append(other_point - point)  # direction vector
                        neighbor_values.append(values[j] - values[i])  # value difference
            
            if len(neighbors) > 0:
                # Compute weighted flow direction
                neighbors = np.array(neighbors)
                weights = np.array(neighbor_values)
                flow = np.sum(neighbors * weights[:, np.newaxis], axis=0)
                flows.append(flow)
            else:
                flows.append(np.array([0.0, 0.0]))
        
        return np.array(flows)
    
    def geometric_computation_engine(self, substrate, operations, query_type="similarity"):
        """Main computational engine using geometric arrangements"""
        print(f"Running geometric computation engine for {query_type}...")
        
        coords = substrate['coordinates']
        values = substrate['values']
        
        if query_type == "similarity":
            return self._compute_similarity_via_geometry(coords, values, operations)
        elif query_type == "prediction":
            return self._compute_prediction_via_geometry(coords, values, operations)
        elif query_type == "optimization":
            return self._compute_optimization_via_geometry(coords, values, operations)
        elif query_type == "pattern_discovery":
            return self._discover_patterns_via_geometry(coords, values, operations)
        else:
            raise ValueError(f"Unknown query type: {query_type}")
    
    def _compute_similarity_via_geometry(self, coords, values, operations):
        """Compute molecular similarity using geometric arrangements"""
        # Use Voronoi neighbors and resonance patterns
        voronoi_graph = operations['voronoi']['neighbor_graph']
        resonance_patterns = operations['resonance_computation']
        
        similarity_matrix = np.zeros((len(coords), len(coords)))
        
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                # Geometric similarity based on multiple factors
                
                # 1. Voronoi neighborhood overlap
                neighbors_i = set(voronoi_graph.get(i, []))
                neighbors_j = set(voronoi_graph.get(j, []))
                neighborhood_similarity = len(neighbors_i & neighbors_j) / len(neighbors_i | neighbors_j) if neighbors_i | neighbors_j else 0
                
                # 2. Resonance coupling
                resonance_coupling = 0
                for pattern_name, pattern_data in resonance_patterns.items():
                    if (i, j) in pattern_data['pairs'] or (j, i) in pattern_data['pairs']:
                        resonance_coupling += 1
                
                # 3. Distance-based similarity
                distance = np.linalg.norm(coords[i] - coords[j])
                distance_similarity = np.exp(-distance)
                
                # 4. Value similarity
                value_similarity = np.exp(-abs(values[i] - values[j]))
                
                # Combined similarity
                total_similarity = (neighborhood_similarity + resonance_coupling + 
                                  distance_similarity + value_similarity) / 4
                
                similarity_matrix[i, j] = similarity_matrix[j, i] = total_similarity
        
        return similarity_matrix
    
    def _compute_prediction_via_geometry(self, coords, values, operations):
        """Predict values using geometric computational methods"""
        distance_matrix = operations['distance_fields']['matrix']
        local_densities = operations['distance_fields']['local_density']
        
        predictions = []
        
        for i in range(len(coords)):
            # Geometric prediction based on local neighborhood
            neighbors = operations['distance_fields']['nearest_neighbors'][i]
            
            # Weight by inverse distance and local density
            weights = []
            neighbor_values = []
            
            for neighbor_idx in neighbors:
                dist = distance_matrix[i, neighbor_idx]
                density = local_densities[neighbor_idx]
                weight = density / (dist + 1e-10)
                weights.append(weight)
                neighbor_values.append(values[neighbor_idx])
            
            if weights:
                weights = np.array(weights)
                weights /= weights.sum()  # normalize
                prediction = np.sum(weights * neighbor_values)
            else:
                prediction = np.mean(values)  # fallback
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _discover_patterns_via_geometry(self, coords, values, operations):
        """Discover patterns using geometric computational methods"""
        patterns = {}
        
        # 1. Geometric clusters with similar values
        hull = operations['convex_hull']['hull']
        boundary_points = hull.vertices
        interior_points = operations['convex_hull']['interior_points']
        
        patterns['boundary_vs_interior'] = {
            'boundary_mean_value': np.mean(values[boundary_points]),
            'interior_mean_value': np.mean(values[interior_points]),
            'boundary_std': np.std(values[boundary_points]),
            'interior_std': np.std(values[interior_points])
        }
        
        # 2. Resonance hotspots
        resonance_data = operations['resonance_computation']
        hotspots = {}
        
        for pattern_name, pattern_info in resonance_data.items():
            if pattern_info['n_resonances'] > 0:
                # Find points involved in most resonances
                point_resonance_count = {}
                for pair in pattern_info['pairs']:
                    for point in pair:
                        point_resonance_count[point] = point_resonance_count.get(point, 0) + 1
                
                if point_resonance_count:
                    max_resonances = max(point_resonance_count.values())
                    hotspot_points = [p for p, count in point_resonance_count.items() 
                                    if count == max_resonances]
                    
                    hotspots[pattern_name] = {
                        'points': hotspot_points,
                        'coordinates': coords[hotspot_points],
                        'values': values[hotspot_points],
                        'resonance_count': max_resonances
                    }
        
        patterns['resonance_hotspots'] = hotspots
        
        # 3. Flow convergence points
        flows = operations['geometric_flows']
        flow_magnitudes = np.linalg.norm(flows, axis=1)
        
        # Find points with minimal flow (potential attractors)
        min_flow_threshold = np.percentile(flow_magnitudes, 10)
        attractor_points = np.where(flow_magnitudes <= min_flow_threshold)[0]
        
        patterns['flow_attractors'] = {
            'points': attractor_points,
            'coordinates': coords[attractor_points],
            'values': values[attractor_points],
            'mean_value': np.mean(values[attractor_points])
        }
        
        return patterns
    
    def visualize_geometric_computation(self, substrate, operations, computation_results):
        """Create comprehensive visualizations of the geometric computation"""
        print("Creating geometric computation visualizations...")
        
        coords = substrate['coordinates']
        values = substrate['values']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Basic 2D projection with values
        scatter = axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=values, 
                                   cmap='viridis', alpha=0.7, s=30)
        axes[0, 0].set_title('2D Geometric Substrate')
        axes[0, 0].set_xlabel('Normalized X')
        axes[0, 0].set_ylabel('Normalized Y')
        plt.colorbar(scatter, ax=axes[0, 0], label='pIC50')
        
        # 2. Voronoi tessellation
        vor = operations['voronoi']['tessellation']
        voronoi_plot_2d(vor, ax=axes[0, 1], show_vertices=False, line_colors='blue', 
                        line_width=1, point_size=2)
        axes[0, 1].scatter(coords[:, 0], coords[:, 1], c=values, cmap='viridis', s=20)
        axes[0, 1].set_title('Voronoi Computational Tessellation')
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Convex hull and boundary analysis
        hull = operations['convex_hull']['hull']
        for simplex in hull.simplices:
            axes[0, 2].plot(coords[simplex, 0], coords[simplex, 1], 'r-', alpha=0.7)
        
        boundary_points = hull.vertices
        interior_points = operations['convex_hull']['interior_points']
        
        axes[0, 2].scatter(coords[interior_points, 0], coords[interior_points, 1], 
                         c='blue', alpha=0.6, s=20, label='Interior')
        axes[0, 2].scatter(coords[boundary_points, 0], coords[boundary_points, 1], 
                         c='red', alpha=0.8, s=40, label='Boundary')
        axes[0, 2].set_title('Convex Hull Computation')
        axes[0, 2].legend()
        
        # 4. Resonance patterns
        resonance_data = operations['resonance_computation']
        axes[1, 0].scatter(coords[:, 0], coords[:, 1], c='lightgray', alpha=0.5, s=20)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (pattern_name, pattern_info) in enumerate(resonance_data.items()):
            if i < len(colors) and pattern_info['pairs']:
                for pair in pattern_info['pairs'][:50]:  # Limit for visibility
                    p1, p2 = pair
                    axes[1, 0].plot([coords[p1, 0], coords[p2, 0]], 
                                  [coords[p1, 1], coords[p2, 1]], 
                                  color=colors[i], alpha=0.3, linewidth=0.5)
        
        axes[1, 0].set_title('Resonance Pattern Computation')
        
        # 5. Geometric flows
        flows = operations['geometric_flows']
        # Subsample for visualization
        step = max(1, len(coords) // 100)
        axes[1, 1].quiver(coords[::step, 0], coords[::step, 1], 
                        flows[::step, 0], flows[::step, 1], 
                        alpha=0.6, scale=10)
        axes[1, 1].scatter(coords[:, 0], coords[:, 1], c=values, 
                         cmap='viridis', alpha=0.5, s=10)
        axes[1, 1].set_title('Geometric Flow Computation')
        
        # 6. Computational results (if similarity matrix available)
        if 'similarity' in str(type(computation_results)):
            im = axes[1, 2].imshow(computation_results[:50, :50], cmap='viridis', aspect='auto')
            axes[1, 2].set_title('Geometric Similarity Matrix (50x50 subset)')
            plt.colorbar(im, ax=axes[1, 2])
        else:
            axes[1, 2].text(0.5, 0.5, 'Computation Results\n(Non-matrix output)', 
                          ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Geometric Computation Output')
        
        plt.tight_layout()
        plt.savefig('geometric_computation_framework.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved geometric computation visualization")

def main():
    """Demonstrate the geometric computation framework"""
    print("Loading data for geometric computation framework...")
    
    # Load processed data
    df = pd.read_csv('kinase_compounds_features.csv')
    morgan_fps = np.load('morgan_fingerprints.npy')
    
    # Quick UMAP for demonstration
    print("Performing UMAP projection...")
    descriptor_cols = [col for col in df.columns if col not in 
                      ['canonical_smiles', 'molecule_chembl_id', 'standard_value', 
                       'target_chembl_id', 'pIC50']]
    
    descriptors = df[descriptor_cols].values
    combined_features = np.hstack([descriptors, morgan_fps])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)
    
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = umap_reducer.fit_transform(scaled_features)
    
    # Initialize geometric computation engine
    engine = GeometricComputationEngine()
    
    # Create computational substrate
    substrate = engine.create_geometric_substrate(embedding_2d, df['pIC50'].values)
    
    # Implement geometric operations
    operations = engine.implement_geometric_operations(substrate)
    
    # Run different computational queries
    print("\n" + "="*60)
    print("GEOMETRIC COMPUTATION RESULTS")
    print("="*60)
    
    # Similarity computation
    similarity_matrix = engine.geometric_computation_engine(substrate, operations, "similarity")
    print(f"Similarity computation: Generated {similarity_matrix.shape} matrix")
    
    # Pattern discovery
    patterns = engine.geometric_computation_engine(substrate, operations, "pattern_discovery")
    print(f"Pattern discovery: Found {len(patterns)} pattern types")
    
    for pattern_type, pattern_data in patterns.items():
        print(f"  {pattern_type}: {len(pattern_data) if isinstance(pattern_data, dict) else 'N/A'} elements")
    
    # Prediction
    predictions = engine.geometric_computation_engine(substrate, operations, "prediction")
    prediction_error = np.mean(np.abs(predictions - df['pIC50'].values))
    print(f"Prediction computation: MAE = {prediction_error:.4f}")
    
    # Create visualizations
    engine.visualize_geometric_computation(substrate, operations, similarity_matrix)
    
    print(f"\nGeometric Computation Framework Summary:")
    print(f"  Substrate points: {substrate['n_points']}")
    print(f"  Computational density: {substrate['density']:.2f} points/unit²")
    print(f"  Voronoi cells: {len(operations['voronoi']['cell_areas'])}")
    print(f"  Convex hull vertices: {len(operations['convex_hull']['boundary_points'])}")
    print(f"  Total resonance patterns: {sum(ops['n_resonances'] for ops in operations['resonance_computation'].values())}")
    
    # Save results
    results_df = df.copy()
    results_df['geometric_x'] = substrate['coordinates'][:, 0]
    results_df['geometric_y'] = substrate['coordinates'][:, 1]
    results_df['geometric_prediction'] = predictions
    results_df['local_density'] = operations['distance_fields']['local_density']
    
    results_df.to_csv('geometric_computation_results.csv', index=False)
    print(f"\nResults saved to geometric_computation_results.csv")

if __name__ == "__main__":
    main()
