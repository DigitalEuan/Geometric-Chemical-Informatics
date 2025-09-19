#!/usr/bin/env python3
"""
Generate and save UMAP embeddings from the UBP encoded data.
"""

import pandas as pd
import numpy as np
import umap
import warnings
warnings.filterwarnings('ignore')

def generate_umap_embeddings():
    print("Generating UMAP embeddings...")
    try:
        df = pd.read_csv("ubp_encoded_inorganic_materials.csv")
        print(f"Loaded {len(df)} materials.")

        # Select features for UMAP (all numeric columns)
        numeric_cols = df.select_dtypes(include=np.number).columns
        # Exclude some non-feature columns
        exclude_cols = ['material_id', 'toggle_sum', 'active_offbits_count']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0)
        print(f"Using {len(feature_cols)} features for UMAP.")

        # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X)
        print("UMAP fitting complete.")

        # Save embeddings
        np.save("ubp_umap_embeddings.npy", embedding)
        print("Saved embeddings to ubp_umap_embeddings.npy")

    except Exception as e:
        print(f"Error generating embeddings: {e}")

if __name__ == "__main__":
    generate_umap_embeddings()

