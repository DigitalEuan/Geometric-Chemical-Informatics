# The Universal Binary Principle (UBP) in Geometric Chemical Informatics: Unified Data and Code Repository
By Euan R A Craig (DigitalEuan), New Zealand

This repository contains the full datasets, feature engineering pipelines, UBP encoding algorithms, and predictive modeling scripts supporting the comprehensive research papers on the application of the Universal Binary Principle (UBP) to both drug discovery and inorganic materials science.

The UBP framework represents a fundamentally new approach to chemical informatics, modeling reality as a deterministic, toggle-based computational system governed by geometric and informational coherence principles.

---

### Framework Validation Highlights

*   **Materials Coherence:** $\mathbf{79.8\%}$ of the 495 transition metal compounds achieved the high coherence target of $\text{NRCI} \geq 0.999999$.
*   **Hypothesis Generation:** The UBP-enhanced analysis generated 15 high-quality geometric hypotheses for drug discovery, validated by a mean NRCI score of 0.7496.
*   **Fundamental Correlation:** Scripts demonstrate the strong correlation found between the **UBP energy equation** (derived from first principles) and the biological activity of molecules.
*   **Geometric Mapping:** Code for creating the **"Periodic Neighborhood" map** for materials and the geometric maps of chemical space using UMAP.

---

## Repository Structure and Datasets

This repository contains the data and code for three studies:

### 1. Inorganic Materials Science Study
*   **Target:** 495 pure inorganic transition metal compounds (binary and ternary).
*   **Source:** Materials Project database via REST API.
*   **Data Included:** Raw property data, 89 traditional inorganic features (Basic, Crystallographic, Electronic, Topological), and 44 novel UBP-Specific features (NRCI, realm assignments, UBP energy calculations).
*   **Code:** UBP encoding algorithms (translating properties to UBP realms), UMAP script for Periodic Neighborhood map construction, and Random Forest models for metric prediction.

### 2. Drug Discovery Study (Kinase Inhibitors)
*   **Target:** 4,073 kinase inhibitors.
*   **Source:** ChEMBL database.
*   **Data Included:** Canonical SMILES strings and $\text{pIC}_{50}$ values for 10 kinase targets.
*   **Code:** Scripts for Morgan fingerprint and molecular descriptor calculation (RDKit), UMAP mapping, and the Gradient Boosting Regressor pipeline that achieved $\text{R}^{2}=0.83$ using geometric features.

### 3. Drug Discovery Study (Dopamine D2 Receptor)
*   **Target:** 1,000 unique compounds targeting the Dopamine D2 receptor.
*   **Source:** ChEMBL database.
*   **Data Included:** Compound structures and reported $\text{pKi}$ values.
*   **Code:** QSAR pipeline (Random Forest, Gradient Boosting) for baseline $\text{R}^{2}=0.6233$. Scripts for feature engineering (Mordred library), UBP molecular encoding (OffBits, Triad framework), and the UBP-enhanced geometric hypothesis generator.

---

## Reproducibility and Dependencies

The complete UBP implementation, including encoding algorithms and visualization tools, is available as open-source software. To reproduce the results, the following key dependencies are necessary:

*   **Dimensionality Reduction:** UMAP (Uniform Manifold Approximation and Projection).
*   **Machine Learning:** scikit-learn (Random Forest, Gradient Boosting).
*   **Cheminformatics:** RDKit [19] and Mordred.
*   **Materials Informatics:** pymatgen (Python Materials Genomics).
*   **Data Sources:** Access to the ChEMBL Database and Materials Project REST API.

---

## Citation and Acknowledgments

If you utilize this data or code in your research, please cite this repository, the Universal Binary Principal (UBP) by Euan R A Craig, New Zealand and acknowledge the following resources:

We thank the **Materials Project team** and the **ChEMBL Database** for providing access to the foundational datasets. Special thanks go to the developers of **UMAP, RDKit, Mordred, scikit-learn, and pymatgen** for their invaluable contributions to materials and chemical informatics
