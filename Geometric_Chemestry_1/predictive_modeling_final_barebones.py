#!/usr/bin/env python3
"""
Barebones Predictive Modeling and Validation Framework
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class BarebonesPredictiveFramework:
    def __init__(self):
        self.scaler = StandardScaler()

    def prepare_features(self, df, morgan_fps):
        print("Preparing feature sets...")
        molecular_descriptors = df[[c for c in df.columns if c not in ["canonical_smiles", "molecule_chembl_id", "standard_value", "target_chembl_id", "pIC50"]]].values
        embedding_2d = np.load("embedding_2d.npy")
        
        feature_sets = {
            "molecular_only": molecular_descriptors,
            "umap_2d": embedding_2d,
            "molecular_plus_umap_2d": np.hstack([molecular_descriptors, embedding_2d]),
        }
        return feature_sets

    def build_model(self, feature_sets, target_values):
        print("Building predictive model...")
        X = feature_sets["molecular_plus_umap_2d"]
        y = target_values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"Model R2: {r2:.4f}")
        
        # Create and save visualization
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        plt.xlabel("Actual pIC50")
        plt.ylabel("Predicted pIC50")
        plt.title(f"Predictive Model Performance (R2 = {r2:.4f})")
        plt.savefig("comprehensive_predictive_analysis.png", dpi=150)
        print("Saved visualization.")

        # Save results
        results_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        results_df.to_csv("final_predictive_modeling_results.csv", index=False)
        print("Saved prediction results.")

def main():
    print("Loading data...")
    df = pd.read_csv("kinase_compounds_features.csv")
    morgan_fps = np.load("morgan_fingerprints.npy")
    target_values = df["pIC50"].values
    
    framework = BarebonesPredictiveFramework()
    feature_sets = framework.prepare_features(df, morgan_fps)
    framework.build_model(feature_sets, target_values)
    print("Analysis complete.")

if __name__ == "__main__":
    main()

