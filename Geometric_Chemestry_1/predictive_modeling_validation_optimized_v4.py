#!/usr/bin/env python3
"""
Predictive Modeling and Validation Framework
Incorporating geometric computation concepts and comprehensive model evaluation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings("ignore")

class GeometricPredictiveFramework:
    """Advanced predictive modeling framework incorporating geometric insights"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.geometric_features = {}
        
        self.phi = (1 + np.sqrt(5)) / 2
        self.pi = np.pi
        self.sqrt2 = np.sqrt(2)
        self.e = np.e
        
    def prepare_comprehensive_features(self, df, morgan_fps):
        """Prepare comprehensive feature sets using pre-computed embeddings"""
        print("Preparing comprehensive feature sets from pre-computed embeddings...")
        
        molecular_descriptors = df[[col for col in df.columns if col not in 
                          ["canonical_smiles", "molecule_chembl_id", "standard_value", 
                           "target_chembl_id", "pIC50"]]].values
        
        scaled_features = np.load("scaled_features.npy")
        embedding_2d = np.load("embedding_2d.npy")
        embedding_3d = np.load("embedding_3d.npy")
        
        print("  Computing geometric projection features...")
        geometric_features_2d = self._compute_geometric_features(embedding_2d, df["pIC50"].values)
        geometric_features_3d = self._compute_geometric_features(embedding_3d, df["pIC50"].values)
        
        feature_sets = {
            "molecular_only": molecular_descriptors,
            "fingerprints_only": morgan_fps,
            "traditional_combined": np.hstack([molecular_descriptors, morgan_fps]),
            "umap_2d": embedding_2d,
            "umap_3d": embedding_3d,
            "geometric_2d": geometric_features_2d,
            "geometric_3d": geometric_features_3d,
            "molecular_plus_umap_2d": np.hstack([molecular_descriptors, embedding_2d]),
            "molecular_plus_geometric_2d": np.hstack([molecular_descriptors, geometric_features_2d]),
            "full_geometric_2d": np.hstack([scaled_features, embedding_2d, geometric_features_2d]),
            "full_geometric_3d": np.hstack([scaled_features, embedding_3d, geometric_features_3d])
        }
        
        self.geometric_features = {
            "embedding_2d": embedding_2d,
            "embedding_3d": embedding_3d,
            "geometric_2d": geometric_features_2d,
            "geometric_3d": geometric_features_3d
        }
        
        print(f"  Generated {len(feature_sets)} feature sets")
        for name, features in feature_sets.items():
            print(f"    {name}: {features.shape}")
        
        return feature_sets
    
    def _compute_geometric_features(self, embedding, pic50_values):
        """Compute geometric features from 2D/3D projections"""
        n_points = len(embedding)
        
        distance_matrix = squareform(pdist(embedding))
        
        geometric_features = []
        
        for i in range(n_points):
            features = []
            
            k = min(10, n_points - 1)
            nearest_distances = np.sort(distance_matrix[i])[1:k+1]
            local_density = 1.0 / (np.mean(nearest_distances) + 1e-10)
            features.append(local_density)
            
            center = np.mean(embedding, axis=0)
            distance_to_center = np.linalg.norm(embedding[i] - center)
            features.append(distance_to_center)
            
            resonance_counts = []
            for constant in [self.phi, self.pi, self.sqrt2, self.e]:
                count = np.sum(np.abs(distance_matrix[i] - constant) < 0.1)
                resonance_counts.append(count)
            features.extend(resonance_counts)
            
            neighbor_indices = np.argsort(distance_matrix[i])[1:k+1]
            neighbor_activities = pic50_values[neighbor_indices]
            local_activity_var = np.var(neighbor_activities)
            features.append(local_activity_var)
            
            activity_weights = np.exp(-np.abs(pic50_values[neighbor_indices] - pic50_values[i]))
            weighted_density = np.sum(activity_weights / (nearest_distances + 1e-10))
            features.append(weighted_density)
            
            features.extend(embedding[i])
            
            if embedding.shape[1] == 2:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(embedding)
                    is_boundary = i in hull.vertices
                    features.append(float(is_boundary))
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            geometric_features.append(features)
        
        return np.array(geometric_features)
    
    def build_predictive_models(self, feature_sets, target_values):
        """Build and evaluate multiple predictive models"""
        print("Building predictive models...")
        
        model_configs = {
            "random_forest": {
                "model": RandomForestRegressor(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [10, 20],
                    "min_samples_split": [2, 5]
                }
            },
            "gradient_boosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "params": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                }
            }
        }
        
        results = {}
        
        for feature_name, X in feature_sets.items():
            print(f"\n  Evaluating feature set: {feature_name}")
            feature_results = {}
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, target_values, test_size=0.2, random_state=42
            )
            
            for model_name, config in model_configs.items():
                try:
                    grid_search = GridSearchCV(
                        config["model"], 
                        config["params"],
                        cv=3,
                        scoring="r2",
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    
                    y_pred_train = best_model.predict(X_train)
                    y_pred_test = best_model.predict(X_test)
                    
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    
                    cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring="r2")
                    
                    feature_results[model_name] = {
                        "best_params": grid_search.best_params_,
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "test_mae": test_mae,
                        "test_rmse": test_rmse,
                        "cv_mean": cv_scores.mean(),
                        "cv_std": cv_scores.std(),
                        "model": best_model,
                        "predictions": y_pred_test,
                        "y_test": y_test
                    }
                    
                    print(f"    {model_name}: R2 = {test_r2:.4f}, MAE = {test_mae:.4f}")
                    
                except Exception as e:
                    print(f"    {model_name}: Failed - {str(e)}")
                    feature_results[model_name] = None
            
            results[feature_name] = feature_results
        
        self.results = results
        return results

    def identify_best_models(self):
        """Identify the best performing models across all feature sets"""
        print("Identifying best performing models...")
        
        all_performances = []
        
        for feature_name, feature_results in self.results.items():
            for model_name, model_data in feature_results.items():
                if model_data:
                    all_performances.append({
                        "feature_set": feature_name,
                        "model": model_name,
                        "test_r2": model_data["test_r2"],
                        "test_mae": model_data["test_mae"],
                        "cv_mean": model_data["cv_mean"],
                        "cv_std": model_data["cv_std"]
                    })
        
        all_performances.sort(key=lambda x: x["test_r2"], reverse=True)
        
        print("\nTop 5 Model Performances:")
        print("-" * 80)
        print("{:<4} {:<25} {:<15} {:<8} {:<8} {:<8}".format("Rank", "Feature Set", "Model", "Test R2", "CV Mean", "MAE"))
        print("-" * 80)
        
        for i, perf in enumerate(all_performances[:5]):
            print(f'{i+1:<4} {perf["feature_set"]:<25} {perf["model"]:<15} {perf["test_r2"]:<8.4f} {perf["cv_mean"]:<8.4f} {perf["test_mae"]:<8.4f}')
        
        return all_performances

    def create_comprehensive_visualizations(self, target_values):
        """Create comprehensive visualizations of model performance and geometric insights"""
        print("Creating comprehensive visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        best_performance = max([
            (feature_name, model_name, model_data)
            for feature_name, feature_results in self.results.items()
            for model_name, model_data in feature_results.items()
            if model_data
        ], key=lambda x: x[2]["test_r2"])
        
        best_feature, best_model, best_data = best_performance
        y_test = best_data["y_test"]
        y_pred = best_data["predictions"]
        
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        axes[0, 0].set_xlabel("Actual pIC50")
        axes[0, 0].set_ylabel("Predicted pIC50")
        axes[0, 0].set_title(f'Best Model: {best_model} + {best_feature}\nR2 = {best_data["test_r2"]:.4f}')
        
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("Predicted pIC50")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals Plot")
        
        if hasattr(best_data["model"], "feature_importances_"):
            importances = best_data["model"].feature_importances_
            indices = np.argsort(importances)[-15:]
            axes[1, 0].barh(range(len(indices)), importances[indices])
            axes[1, 0].set_title("Top 15 Feature Importances")
            axes[1, 0].set_xlabel("Importance")
        else:
            axes[1, 0].text(0.5, 0.5, "Feature importance\nnot available", 
                          ha="center", va="center", transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("Feature Importance")
        
        if "embedding_2d" in self.geometric_features:
            embedding_2d = self.geometric_features["embedding_2d"]
            scatter = axes[1, 1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                       c=target_values, cmap="viridis", alpha=0.6)
            axes[1, 1].set_title("2D UMAP Embedding")
            axes[1, 1].set_xlabel("UMAP 1")
            axes[1, 1].set_ylabel("UMAP 2")
            plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig("comprehensive_predictive_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print("Saved comprehensive analysis visualization")

def main():
    """Main execution function"""
    print("Loading data for predictive modeling...")
    
    df = pd.read_csv("kinase_compounds_features.csv")
    morgan_fps = np.load("morgan_fingerprints.npy")
    target_values = df["pIC50"].values
    
    print(f"Dataset: {len(df)} compounds, {morgan_fps.shape[1]} fingerprint bits")
    
    framework = GeometricPredictiveFramework()
    
    feature_sets = framework.prepare_comprehensive_features(df, morgan_fps)
    
    results = framework.build_predictive_models(feature_sets, target_values)
    
    best_models = framework.identify_best_models()
    
    framework.create_comprehensive_visualizations(target_values)
    
    print("\n" + "="*80)
    print("PREDICTIVE MODELING SUMMARY")
    print("="*80)
    
    best = best_models[0]
    print(f"\nBest Model Performance:")
    print(f'  Feature Set: {best["feature_set"]}')
    print(f'  Model: {best["model"]}')
    print(f'  Test R2: {best["test_r2"]:.4f}')
    print(f'  Test MAE: {best["test_mae"]:.4f}')
    print(f'  CV Mean ± Std: {best["cv_mean"]:.4f} ± {best["cv_std"]:.4f}')
    
    final_df = df.copy()
    best_feature_set = best["feature_set"]
    best_model_name = best["model"]
    best_model_obj = framework.results[best_feature_set][best_model_name]["model"]
    
    full_predictions = best_model_obj.predict(feature_sets[best_feature_set])
    
    final_df["predicted_pIC50"] = full_predictions
    final_df["prediction_error"] = np.abs(final_df["pIC50"] - final_df["predicted_pIC50"])
    
    if "embedding_2d" in framework.geometric_features:
        final_df["geometric_x"] = framework.geometric_features["embedding_2d"][:, 0]
        final_df["geometric_y"] = framework.geometric_features["embedding_2d"][:, 1]
    
    final_df.to_csv("final_predictive_modeling_results.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"  - final_predictive_modeling_results.csv")
    print(f"  - comprehensive_predictive_analysis.png")

if __name__ == "__main__":
    main()

