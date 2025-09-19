#!/usr/bin/env python3
"""
Version 2 Predictive Modeling with Optimal Feature Sets
Advanced machine learning incorporating geometric features and UBP principles

Key Features:
1. Multiple feature set combinations (traditional + geometric + fingerprints)
2. Advanced ensemble methods with geometric weighting
3. UBP-inspired model architectures
4. Cross-validation with geometric consistency
5. Feature importance analysis for drug discovery insights
6. NRCI-based model evaluation
7. Geometric hypothesis generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import warnings
warnings.filterwarnings('ignore')

class PredictiveModeling:
    """Advanced predictive modeling with geometric features and UBP principles"""
    
    def __init__(self):
        # UBP Core Resonance Values for model weighting
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
        
        # Model configurations
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            'ridge': Ridge(random_state=42),
            'svr': SVR(),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        # Results storage
        self.results = {}
        self.best_models = {}
        
    def load_data(self):
        """Load all feature sets and target data"""
        print("Loading comprehensive dataset for predictive modeling...")
        
        # Load main dataset
        self.dataset = pd.read_csv("v2_comprehensive_features.csv")
        self.target = self.dataset['pKi'].values
        
        # Load fingerprints
        self.fingerprints = np.load("v2_ecfp4_fingerprints.npy")
        
        # Load geometric analysis results
        self.geometric_results = pd.read_csv("v2_geometric_analysis_results.csv")
        
        print(f"Dataset: {self.dataset.shape}")
        print(f"Fingerprints: {self.fingerprints.shape}")
        print(f"Geometric results: {self.geometric_results.shape}")
        
        # Prepare feature sets
        self.prepare_feature_sets()
        
        return self.dataset
    
    def prepare_feature_sets(self):
        """Prepare different feature combinations for modeling"""
        print("Preparing feature sets...")
        
        # Traditional molecular descriptors
        mordred_cols = [col for col in self.dataset.columns if col.startswith('mordred_')]
        self.traditional_features = self.dataset[mordred_cols].fillna(0).values
        
        # Geometric features from analysis results
        # Aggregate geometric features across embedding methods
        geometric_features = []
        feature_names = ['phi_resonance', 'pi_resonance', 'sqrt2_resonance', 'e_resonance',
                        'convex_hull_area', 'centroid_distance_variance', 'angular_distribution',
                        'activity_distance_correlation', 'projection_complexity', 'geometric_entropy']
        
        # Create geometric feature matrix by averaging across embedding methods
        for i in range(len(self.target)):
            compound_geometric_features = []
            for feature_name in feature_names:
                if feature_name in self.geometric_results.columns:
                    # Average across all embedding methods for this compound
                    feature_values = self.geometric_results[feature_name].values
                    # Handle potential NaN values
                    valid_values = feature_values[~np.isnan(feature_values)]
                    if len(valid_values) > 0:
                        avg_value = np.mean(valid_values)
                    else:
                        avg_value = 0.0
                    compound_geometric_features.append(avg_value)
                else:
                    compound_geometric_features.append(0.0)
            geometric_features.append(compound_geometric_features)
        
        self.geometric_features = np.array(geometric_features)
        
        # Feature set combinations
        self.feature_sets = {
            'traditional': self.traditional_features,
            'fingerprints': self.fingerprints,
            'geometric': self.geometric_features,
            'traditional_geometric': np.hstack([self.traditional_features, self.geometric_features]),
            'fingerprints_geometric': np.hstack([self.fingerprints, self.geometric_features]),
            'all_features': np.hstack([self.traditional_features, self.fingerprints, self.geometric_features])
        }
        
        print(f"Feature sets prepared:")
        for name, features in self.feature_sets.items():
            print(f"  {name}: {features.shape}")
    
    def calculate_nrci(self, y_true, y_pred):
        """Calculate NRCI for model evaluation"""
        try:
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            sigma_true = np.std(y_true)
            
            if sigma_true == 0:
                return 1.0 if rmse == 0 else 0.0
            
            nrci = 1 - (rmse / sigma_true)
            return max(0.0, min(1.0, nrci))
        except:
            return 0.0
    
    def ubp_weighted_ensemble(self, predictions_dict, weights=None):
        """Create UBP-weighted ensemble predictions"""
        if weights is None:
            # Use CRV values as weights
            weights = {
                'random_forest': self.crv_values['biological'],
                'gradient_boosting': self.crv_values['electromagnetic'],
                'extra_trees': self.crv_values['quantum'],
                'elastic_net': self.crv_values['golden_ratio'],
                'ridge': self.crv_values['sqrt2'],
                'svr': self.crv_values['euler'],
                'neural_network': self.crv_values['cosmological']
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted ensemble
        ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])
        for model_name, predictions in predictions_dict.items():
            if model_name in normalized_weights:
                ensemble_pred += normalized_weights[model_name] * predictions
        
        return ensemble_pred
    
    def train_and_evaluate_models(self):
        """Train and evaluate models on all feature sets"""
        print("Training and evaluating models...")
        
        self.results = {}
        
        for feature_set_name, features in self.feature_sets.items():
            print(f"\n  Processing feature set: {feature_set_name}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, self.target, test_size=0.2, random_state=42
            )
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            feature_results = {}
            predictions_dict = {}
            
            for model_name, model in self.models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Predictions
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    # Store predictions for ensemble
                    predictions_dict[model_name] = y_pred_test
                    
                    # Evaluate
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    train_nrci = self.calculate_nrci(y_train, y_pred_train)
                    test_nrci = self.calculate_nrci(y_test, y_pred_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    
                    feature_results[model_name] = {
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_nrci': train_nrci,
                        'test_nrci': test_nrci,
                        'cv_mean': np.mean(cv_scores),
                        'cv_std': np.std(cv_scores),
                        'model': model,
                        'scaler': scaler
                    }
                    
                    print(f"    {model_name}: R² = {test_r2:.4f}, NRCI = {test_nrci:.6f}")
                    
                except Exception as e:
                    print(f"    {model_name}: Error - {e}")
                    continue
            
            # UBP-weighted ensemble
            if len(predictions_dict) > 1:
                ensemble_pred = self.ubp_weighted_ensemble(predictions_dict)
                ensemble_r2 = r2_score(y_test, ensemble_pred)
                ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                ensemble_nrci = self.calculate_nrci(y_test, ensemble_pred)
                
                feature_results['ubp_ensemble'] = {
                    'test_r2': ensemble_r2,
                    'test_rmse': ensemble_rmse,
                    'test_nrci': ensemble_nrci,
                    'predictions': ensemble_pred
                }
                
                print(f"    UBP Ensemble: R² = {ensemble_r2:.4f}, NRCI = {ensemble_nrci:.6f}")
            
            self.results[feature_set_name] = feature_results
        
        print("\nModel training and evaluation complete!")
    
    def feature_importance_analysis(self):
        """Analyze feature importance across models and feature sets"""
        print("Analyzing feature importance...")
        
        importance_results = {}
        
        for feature_set_name, feature_results in self.results.items():
            print(f"\n  Analyzing {feature_set_name}...")
            
            # Get feature names
            if feature_set_name == 'traditional':
                feature_names = [col for col in self.dataset.columns if col.startswith('mordred_')]
            elif feature_set_name == 'fingerprints':
                feature_names = [f'ECFP4_{i}' for i in range(self.fingerprints.shape[1])]
            elif feature_set_name == 'geometric':
                feature_names = ['phi_resonance', 'pi_resonance', 'sqrt2_resonance', 'e_resonance',
                               'convex_hull_area', 'centroid_distance_variance', 'angular_distribution',
                               'activity_distance_correlation', 'projection_complexity', 'geometric_entropy']
            else:
                # Combined feature sets
                feature_names = []
                if 'traditional' in feature_set_name:
                    feature_names.extend([col for col in self.dataset.columns if col.startswith('mordred_')])
                if 'fingerprints' in feature_set_name:
                    feature_names.extend([f'ECFP4_{i}' for i in range(self.fingerprints.shape[1])])
                if 'geometric' in feature_set_name:
                    feature_names.extend(['phi_resonance', 'pi_resonance', 'sqrt2_resonance', 'e_resonance',
                                        'convex_hull_area', 'centroid_distance_variance', 'angular_distribution',
                                        'activity_distance_correlation', 'projection_complexity', 'geometric_entropy'])
            
            set_importance = {}
            
            # Tree-based models have feature_importances_
            for model_name in ['random_forest', 'gradient_boosting', 'extra_trees']:
                if model_name in feature_results and 'model' in feature_results[model_name]:
                    model = feature_results[model_name]['model']
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        
                        # Get top 10 most important features
                        top_indices = np.argsort(importances)[-10:][::-1]
                        top_features = [(feature_names[i] if i < len(feature_names) else f'Feature_{i}', 
                                       importances[i]) for i in top_indices]
                        
                        set_importance[model_name] = top_features
            
            importance_results[feature_set_name] = set_importance
        
        self.importance_results = importance_results
        print("Feature importance analysis complete!")
        
        return importance_results
    
    def identify_best_models(self):
        """Identify best performing models across feature sets"""
        print("Identifying best models...")
        
        best_models = {}
        
        # Find best model for each metric
        best_r2 = {'score': -np.inf, 'model': None, 'feature_set': None}
        best_nrci = {'score': -np.inf, 'model': None, 'feature_set': None}
        best_cv = {'score': -np.inf, 'model': None, 'feature_set': None}
        
        for feature_set_name, feature_results in self.results.items():
            for model_name, metrics in feature_results.items():
                if model_name == 'ubp_ensemble':
                    continue
                
                # Best R²
                if metrics.get('test_r2', -np.inf) > best_r2['score']:
                    best_r2 = {
                        'score': metrics['test_r2'],
                        'model': model_name,
                        'feature_set': feature_set_name,
                        'nrci': metrics.get('test_nrci', 0)
                    }
                
                # Best NRCI
                if metrics.get('test_nrci', -np.inf) > best_nrci['score']:
                    best_nrci = {
                        'score': metrics['test_nrci'],
                        'model': model_name,
                        'feature_set': feature_set_name,
                        'r2': metrics.get('test_r2', 0)
                    }
                
                # Best CV
                if metrics.get('cv_mean', -np.inf) > best_cv['score']:
                    best_cv = {
                        'score': metrics['cv_mean'],
                        'model': model_name,
                        'feature_set': feature_set_name,
                        'std': metrics.get('cv_std', 0)
                    }
        
        best_models = {
            'best_r2': best_r2,
            'best_nrci': best_nrci,
            'best_cv': best_cv
        }
        
        self.best_models = best_models
        
        print(f"Best R²: {best_r2['model']} on {best_r2['feature_set']} (R² = {best_r2['score']:.4f})")
        print(f"Best NRCI: {best_nrci['model']} on {best_nrci['feature_set']} (NRCI = {best_nrci['score']:.6f})")
        print(f"Best CV: {best_cv['model']} on {best_cv['feature_set']} (CV = {best_cv['score']:.4f})")
        
        return best_models
    
    def generate_visualizations(self):
        """Generate comprehensive visualization of modeling results"""
        print("Generating modeling visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Version 2 Predictive Modeling Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Model Performance Comparison
        self.plot_model_performance(axes[0, 0])
        
        # Plot 2: Feature Set Comparison
        self.plot_feature_set_comparison(axes[0, 1])
        
        # Plot 3: NRCI vs R² Scatter
        self.plot_nrci_vs_r2(axes[0, 2])
        
        # Plot 4: Feature Importance (Geometric Features)
        self.plot_geometric_importance(axes[1, 0])
        
        # Plot 5: Ensemble Performance
        self.plot_ensemble_performance(axes[1, 1])
        
        # Plot 6: Prediction vs Actual
        self.plot_prediction_accuracy(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('v2_predictive_modeling_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Modeling visualizations saved to v2_predictive_modeling_results.png")
    
    def plot_model_performance(self, ax):
        """Plot model performance comparison"""
        models = []
        r2_scores = []
        nrci_scores = []
        
        for feature_set_name, feature_results in self.results.items():
            for model_name, metrics in feature_results.items():
                if model_name != 'ubp_ensemble':
                    models.append(f"{model_name}_{feature_set_name}")
                    r2_scores.append(metrics.get('test_r2', 0))
                    nrci_scores.append(metrics.get('test_nrci', 0))
        
        if models:
            x_pos = np.arange(len(models))
            ax.bar(x_pos, r2_scores, alpha=0.7, label='R²')
            ax.set_xlabel('Model_FeatureSet')
            ax.set_ylabel('R² Score')
            ax.set_title('Model Performance (R²)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    
    def plot_feature_set_comparison(self, ax):
        """Plot feature set comparison"""
        feature_sets = []
        best_r2_per_set = []
        
        for feature_set_name, feature_results in self.results.items():
            r2_scores = [metrics.get('test_r2', 0) for model_name, metrics in feature_results.items() 
                        if model_name != 'ubp_ensemble']
            if r2_scores:
                feature_sets.append(feature_set_name)
                best_r2_per_set.append(max(r2_scores))
        
        if feature_sets:
            bars = ax.bar(feature_sets, best_r2_per_set, alpha=0.7, color='lightgreen')
            ax.set_xlabel('Feature Set')
            ax.set_ylabel('Best R² Score')
            ax.set_title('Best Performance by Feature Set')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def plot_nrci_vs_r2(self, ax):
        """Plot NRCI vs R² scatter"""
        r2_scores = []
        nrci_scores = []
        
        for feature_set_name, feature_results in self.results.items():
            for model_name, metrics in feature_results.items():
                if model_name != 'ubp_ensemble':
                    r2_scores.append(metrics.get('test_r2', 0))
                    nrci_scores.append(metrics.get('test_nrci', 0))
        
        if r2_scores and nrci_scores:
            ax.scatter(r2_scores, nrci_scores, alpha=0.7)
            ax.set_xlabel('R² Score')
            ax.set_ylabel('NRCI Score')
            ax.set_title('NRCI vs R² Relationship')
            
            # Add target lines
            ax.axhline(0.999999, color='red', linestyle='--', alpha=0.5, label='NRCI Target')
            ax.legend()
    
    def plot_geometric_importance(self, ax):
        """Plot geometric feature importance"""
        if hasattr(self, 'importance_results'):
            # Focus on geometric features
            geometric_importance = {}
            
            for feature_set_name, set_importance in self.importance_results.items():
                if 'geometric' in feature_set_name:
                    for model_name, features in set_importance.items():
                        for feature_name, importance in features:
                            if any(geo_feat in feature_name for geo_feat in 
                                  ['phi_resonance', 'pi_resonance', 'sqrt2_resonance', 'e_resonance']):
                                if feature_name not in geometric_importance:
                                    geometric_importance[feature_name] = []
                                geometric_importance[feature_name].append(importance)
            
            if geometric_importance:
                features = list(geometric_importance.keys())
                avg_importance = [np.mean(geometric_importance[feat]) for feat in features]
                
                bars = ax.bar(features, avg_importance, alpha=0.7, color='gold')
                ax.set_xlabel('Geometric Feature')
                ax.set_ylabel('Average Importance')
                ax.set_title('Sacred Geometry Feature Importance')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.text(0.5, 0.5, 'No geometric importance data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No importance data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Geometric Feature Importance')
    
    def plot_ensemble_performance(self, ax):
        """Plot ensemble performance"""
        ensemble_r2 = []
        individual_best_r2 = []
        feature_sets = []
        
        for feature_set_name, feature_results in self.results.items():
            if 'ubp_ensemble' in feature_results:
                ensemble_r2.append(feature_results['ubp_ensemble']['test_r2'])
                
                # Find best individual model for this feature set
                individual_r2 = [metrics.get('test_r2', 0) for model_name, metrics in feature_results.items() 
                               if model_name != 'ubp_ensemble']
                individual_best_r2.append(max(individual_r2) if individual_r2 else 0)
                feature_sets.append(feature_set_name)
        
        if ensemble_r2:
            x_pos = np.arange(len(feature_sets))
            width = 0.35
            
            ax.bar(x_pos - width/2, individual_best_r2, width, label='Best Individual', alpha=0.7)
            ax.bar(x_pos + width/2, ensemble_r2, width, label='UBP Ensemble', alpha=0.7)
            
            ax.set_xlabel('Feature Set')
            ax.set_ylabel('R² Score')
            ax.set_title('Ensemble vs Individual Models')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(feature_sets, rotation=45, ha='right')
            ax.legend()
    
    def plot_prediction_accuracy(self, ax):
        """Plot prediction vs actual for best model"""
        if self.best_models and 'best_r2' in self.best_models:
            best_info = self.best_models['best_r2']
            feature_set = best_info['feature_set']
            model_name = best_info['model']
            
            # Get the model and make predictions
            if (feature_set in self.results and 
                model_name in self.results[feature_set] and 
                'model' in self.results[feature_set][model_name]):
                
                # Use test set for visualization
                features = self.feature_sets[feature_set]
                X_train, X_test, y_train, y_test = train_test_split(
                    features, self.target, test_size=0.2, random_state=42
                )
                
                model = self.results[feature_set][model_name]['model']
                scaler = self.results[feature_set][model_name]['scaler']
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
                
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel('Actual pKi')
                ax.set_ylabel('Predicted pKi')
                ax.set_title(f'Best Model: {model_name} on {feature_set}')
                
                # Add R² annotation
                r2 = best_info['score']
                ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Best model data not available', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No best model identified', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Prediction Accuracy')
    
    def save_results(self, filename="v2_predictive_modeling_results.csv"):
        """Save comprehensive modeling results"""
        print(f"Saving modeling results to {filename}")
        
        # Flatten results for CSV
        results_data = []
        
        for feature_set_name, feature_results in self.results.items():
            for model_name, metrics in feature_results.items():
                if model_name != 'ubp_ensemble':  # Skip ensemble for CSV
                    result_row = {
                        'feature_set': feature_set_name,
                        'model': model_name,
                        'train_r2': metrics.get('train_r2', np.nan),
                        'test_r2': metrics.get('test_r2', np.nan),
                        'train_rmse': metrics.get('train_rmse', np.nan),
                        'test_rmse': metrics.get('test_rmse', np.nan),
                        'train_nrci': metrics.get('train_nrci', np.nan),
                        'test_nrci': metrics.get('test_nrci', np.nan),
                        'cv_mean': metrics.get('cv_mean', np.nan),
                        'cv_std': metrics.get('cv_std', np.nan)
                    }
                    results_data.append(result_row)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filename, index=False)
        
        print(f"Results saved: {len(results_df)} model evaluations")
        
        return results_df

def main():
    """Main execution function"""
    print("="*80)
    print("VERSION 2 PREDICTIVE MODELING WITH OPTIMAL FEATURE SETS")
    print("="*80)
    
    # Initialize modeling
    modeler = PredictiveModeling()
    
    # Load data
    dataset = modeler.load_data()
    
    # Train and evaluate models
    modeler.train_and_evaluate_models()
    
    # Feature importance analysis
    importance_results = modeler.feature_importance_analysis()
    
    # Identify best models
    best_models = modeler.identify_best_models()
    
    # Generate visualizations
    modeler.generate_visualizations()
    
    # Save results
    results_df = modeler.save_results()
    
    print("\n" + "="*80)
    print("PREDICTIVE MODELING SUMMARY")
    print("="*80)
    print(f"Dataset: {len(dataset)} compounds")
    print(f"Feature sets tested: {len(modeler.feature_sets)}")
    print(f"Models evaluated: {len(modeler.models)}")
    print(f"Total evaluations: {len(results_df)}")
    
    if best_models:
        print(f"\nBest performing models:")
        for metric, info in best_models.items():
            print(f"  {metric}: {info['model']} on {info['feature_set']} (score: {info['score']:.4f})")
    
    print("\nVisualization: v2_predictive_modeling_results.png")
    print("Results: v2_predictive_modeling_results.csv")
    print("Ready for geometric hypothesis generation!")
    
    return modeler, results_df

if __name__ == "__main__":
    modeler, results = main()
