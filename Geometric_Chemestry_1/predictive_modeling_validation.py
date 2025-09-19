#!/usr/bin/env python3
"""
Predictive Modeling and Validation Framework
Incorporating geometric computation concepts and comprehensive model evaluation
"""

import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class GeometricPredictiveFramework:
    """Advanced predictive modeling framework incorporating geometric insights"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.geometric_features = {}
        
        # Sacred geometry constants for feature engineering
        self.phi = (1 + np.sqrt(5)) / 2
        self.pi = np.pi
        self.sqrt2 = np.sqrt(2)
        self.e = np.e
        
    def prepare_comprehensive_features(self, df, morgan_fps):
        """Prepare comprehensive feature sets including geometric projections"""
        print("Preparing comprehensive feature sets...")
        
        # Extract molecular descriptors
        descriptor_cols = [col for col in df.columns if col not in 
                          ['canonical_smiles', 'molecule_chembl_id', 'standard_value', 
                           'target_chembl_id', 'pIC50']]
        
        molecular_descriptors = df[descriptor_cols].values
        
        # Combine molecular descriptors with Morgan fingerprints
        combined_features = np.hstack([molecular_descriptors, morgan_fps])
        scaled_features = self.scaler.fit_transform(combined_features)
        
        # Generate UMAP embeddings
        print("  Generating UMAP embeddings...")
        umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = umap_2d.fit_transform(scaled_features)
        
        umap_3d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
        embedding_3d = umap_3d.fit_transform(scaled_features)
        
        # Generate geometric features based on 2D projections
        print("  Computing geometric projection features...")
        geometric_features_2d = self._compute_geometric_features(embedding_2d, df['pIC50'].values)
        geometric_features_3d = self._compute_geometric_features(embedding_3d, df['pIC50'].values)
        
        # Create feature sets
        feature_sets = {
            'molecular_only': molecular_descriptors,
            'fingerprints_only': morgan_fps,
            'traditional_combined': combined_features,
            'umap_2d': embedding_2d,
            'umap_3d': embedding_3d,
            'geometric_2d': geometric_features_2d,
            'geometric_3d': geometric_features_3d,
            'molecular_plus_umap_2d': np.hstack([molecular_descriptors, embedding_2d]),
            'molecular_plus_geometric_2d': np.hstack([molecular_descriptors, geometric_features_2d]),
            'full_geometric_2d': np.hstack([scaled_features, embedding_2d, geometric_features_2d]),
            'full_geometric_3d': np.hstack([scaled_features, embedding_3d, geometric_features_3d])
        }
        
        # Store for later use
        self.geometric_features = {
            'embedding_2d': embedding_2d,
            'embedding_3d': embedding_3d,
            'geometric_2d': geometric_features_2d,
            'geometric_3d': geometric_features_3d
        }
        
        print(f"  Generated {len(feature_sets)} feature sets")
        for name, features in feature_sets.items():
            print(f"    {name}: {features.shape}")
        
        return feature_sets
    
    def _compute_geometric_features(self, embedding, pic50_values):
        """Compute geometric features from 2D/3D projections"""
        n_points = len(embedding)
        
        # Distance-based features
        distance_matrix = squareform(pdist(embedding))
        
        geometric_features = []
        
        for i in range(n_points):
            features = []
            
            # 1. Local density (inverse of mean distance to k nearest neighbors)
            k = min(10, n_points - 1)
            nearest_distances = np.sort(distance_matrix[i])[1:k+1]
            local_density = 1.0 / (np.mean(nearest_distances) + 1e-10)
            features.append(local_density)
            
            # 2. Distance to geometric center
            center = np.mean(embedding, axis=0)
            distance_to_center = np.linalg.norm(embedding[i] - center)
            features.append(distance_to_center)
            
            # 3. Resonance features (distances matching sacred geometry constants)
            resonance_counts = []
            for constant in [self.phi, self.pi, self.sqrt2, self.e]:
                count = np.sum(np.abs(distance_matrix[i] - constant) < 0.1)
                resonance_counts.append(count)
            features.extend(resonance_counts)
            
            # 4. Local activity variance
            neighbor_indices = np.argsort(distance_matrix[i])[1:k+1]
            neighbor_activities = pic50_values[neighbor_indices]
            local_activity_var = np.var(neighbor_activities)
            features.append(local_activity_var)
            
            # 5. Activity-weighted local density
            activity_weights = np.exp(-np.abs(pic50_values[neighbor_indices] - pic50_values[i]))
            weighted_density = np.sum(activity_weights / (nearest_distances + 1e-10))
            features.append(weighted_density)
            
            # 6. Geometric position features (for 2D/3D)
            features.extend(embedding[i])
            
            # 7. Convex hull membership (boundary vs interior)
            if embedding.shape[1] == 2:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(embedding)
                    is_boundary = i in hull.vertices
                    features.append(float(is_boundary))
                except:
                    features.append(0.0)
            else:
                features.append(0.0)  # Placeholder for 3D
            
            geometric_features.append(features)
        
        return np.array(geometric_features)
    
    def build_predictive_models(self, feature_sets, target_values):
        """Build and evaluate multiple predictive models"""
        print("Building predictive models...")
        
        # Define model configurations
        model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'elastic_net': {
                'model': ElasticNet(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            },
            'svr': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
        
        results = {}
        
        for feature_name, X in feature_sets.items():
            print(f"\n  Evaluating feature set: {feature_name}")
            feature_results = {}
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, target_values, test_size=0.2, random_state=42
            )
            
            for model_name, config in model_configs.items():
                try:
                    # Grid search for best parameters
                    grid_search = GridSearchCV(
                        config['model'], 
                        config['params'],
                        cv=5,
                        scoring='r2',
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    
                    # Predictions
                    y_pred_train = best_model.predict(X_train)
                    y_pred_test = best_model.predict(X_test)
                    
                    # Metrics
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    
                    # Cross-validation
                    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
                    
                    feature_results[model_name] = {
                        'best_params': grid_search.best_params_,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'test_mae': test_mae,
                        'test_rmse': test_rmse,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'model': best_model,
                        'predictions': y_pred_test,
                        'y_test': y_test
                    }
                    
                    print(f"    {model_name}: R² = {test_r2:.4f}, MAE = {test_mae:.4f}")
                    
                except Exception as e:
                    print(f"    {model_name}: Failed - {str(e)}")
                    feature_results[model_name] = None
            
            results[feature_name] = feature_results
        
        self.results = results
        return results
    
    def analyze_geometric_contribution(self, feature_sets, target_values):
        """Analyze the contribution of geometric features to prediction performance"""
        print("Analyzing geometric feature contributions...")
        
        # Compare performance with and without geometric features
        baseline_features = ['molecular_only', 'fingerprints_only', 'traditional_combined']
        geometric_features = ['geometric_2d', 'geometric_3d', 'molecular_plus_geometric_2d', 
                            'full_geometric_2d', 'full_geometric_3d']
        
        analysis = {}
        
        for baseline in baseline_features:
            if baseline in self.results:
                baseline_performance = {}
                for model_name, model_results in self.results[baseline].items():
                    if model_results:
                        baseline_performance[model_name] = model_results['test_r2']
                
                analysis[baseline] = {
                    'baseline_performance': baseline_performance,
                    'geometric_improvements': {}
                }
                
                # Compare with geometric variants
                for geom_feature in geometric_features:
                    if geom_feature in self.results:
                        improvements = {}
                        for model_name in baseline_performance:
                            if (model_name in self.results[geom_feature] and 
                                self.results[geom_feature][model_name]):
                                geom_r2 = self.results[geom_feature][model_name]['test_r2']
                                baseline_r2 = baseline_performance[model_name]
                                improvement = geom_r2 - baseline_r2
                                improvements[model_name] = improvement
                        
                        analysis[baseline]['geometric_improvements'][geom_feature] = improvements
        
        return analysis
    
    def identify_best_models(self):
        """Identify the best performing models across all feature sets"""
        print("Identifying best performing models...")
        
        all_performances = []
        
        for feature_name, feature_results in self.results.items():
            for model_name, model_data in feature_results.items():
                if model_data:
                    all_performances.append({
                        'feature_set': feature_name,
                        'model': model_name,
                        'test_r2': model_data['test_r2'],
                        'test_mae': model_data['test_mae'],
                        'cv_mean': model_data['cv_mean'],
                        'cv_std': model_data['cv_std']
                    })
        
        # Sort by test R²
        all_performances.sort(key=lambda x: x['test_r2'], reverse=True)
        
        print("\nTop 10 Model Performances:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Feature Set':<25} {'Model':<15} {'Test R²':<8} {'CV Mean':<8} {'MAE':<8}")
        print("-" * 80)
        
        for i, perf in enumerate(all_performances[:10]):
            print(f"{i+1:<4} {perf['feature_set']:<25} {perf['model']:<15} "
                  f"{perf['test_r2']:<8.4f} {perf['cv_mean']:<8.4f} {perf['test_mae']:<8.4f}")
        
        return all_performances
    
    def create_comprehensive_visualizations(self, target_values):
        """Create comprehensive visualizations of model performance and geometric insights"""
        print("Creating comprehensive visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Model performance comparison
        feature_sets = list(self.results.keys())
        models = ['random_forest', 'gradient_boosting', 'ridge', 'elastic_net', 'svr']
        
        performance_matrix = np.zeros((len(feature_sets), len(models)))
        
        for i, feature_set in enumerate(feature_sets):
            for j, model in enumerate(models):
                if (model in self.results[feature_set] and 
                    self.results[feature_set][model]):
                    performance_matrix[i, j] = self.results[feature_set][model]['test_r2']
                else:
                    performance_matrix[i, j] = np.nan
        
        im1 = axes[0, 0].imshow(performance_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Model Performance Heatmap (Test R²)')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Feature Sets')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].set_yticks(range(len(feature_sets)))
        axes[0, 0].set_yticklabels(feature_sets, rotation=0)
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Best model predictions vs actual
        best_performance = max([
            (feature_name, model_name, model_data)
            for feature_name, feature_results in self.results.items()
            for model_name, model_data in feature_results.items()
            if model_data
        ], key=lambda x: x[2]['test_r2'])
        
        best_feature, best_model, best_data = best_performance
        y_test = best_data['y_test']
        y_pred = best_data['predictions']
        
        axes[0, 1].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[0, 1].set_xlabel('Actual pIC50')
        axes[0, 1].set_ylabel('Predicted pIC
(Content truncated due to size limit. Use page ranges or line ranges to read remaining content)