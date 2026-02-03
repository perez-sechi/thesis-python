# Thesis Python Support Library

This project provides computational support and reproducible experiments for the doctoral thesis on **Network Visualization Methods for Cooperative Game Theory and Explainable Artificial Intelligence**. The notebooks generate figures, tables, and analytical results presented in the thesis chapters located at `thesis-latex/Capitulos/*.tex`.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Chapter-by-Chapter Guide](#chapter-by-chapter-guide)
  - [Chapter 3: Visualization Methods for Cooperative Games](#chapter-3-visualization-methods-for-cooperative-games)
  - [Chapter 4: Structural Analysis of Network Representation](#chapter-4-structural-analysis-of-network-representation)
  - [Chapter 5: Network Visualization of Fuzzy Measures](#chapter-5-network-visualization-of-fuzzy-measures)
  - [Chapter 6: Visual Interpretation of Models with SHAP](#chapter-6-visual-interpretation-of-models-with-shap)
  - [Chapter 7: Interaction Analysis with KWSHAP](#chapter-7-interaction-analysis-with-kwshap)
  - [Chapter 8: Aggregation of Explanations via Multi-Measure Fuzzy Systems](#chapter-8-aggregation-of-explanations-via-multi-measure-fuzzy-systems)
- [Data Sources](#data-sources)
- [Dependencies](#dependencies)
- [Core Library: cgt_perezsechi](#core-library-cgt_perezsechi)
- [License](#license)

---

## Project Overview

This repository contains Jupyter notebooks that implement the computational experiments supporting each chapter of the thesis. The main contributions include:

1. **Network visualization of cooperative games** using Shapley values (node relevance) and Interaction Indices (edge weights)
2. **Structural analysis** through weighted degree statistics, clustering coefficients, and community detection
3. **Extension to fuzzy measures** for modeling non-additive systems like parliamentary voting
4. **Explainable AI (XAI)** through SHAP value analysis and network visualization of feature interactions
5. **KWSHAP methodology** for discretizing continuous variables based on statistical significance
6. **Multi-Measure Fuzzy Systems (MMFS)** for aggregating multiple local explanations into interpretable networks

---

## Installation

1. **Clone the repository** and navigate to the `thesis-python` directory:
   ```bash
   cd thesis-python
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** The project depends on `cgt_perezsechi`, a custom library installed from GitHub that provides core functionality for visualization, normalization, and analysis.

4. **Install GraphViz** (required for network visualizations):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install graphviz graphviz-dev
   
   # macOS
   brew install graphviz
   ```

---

## Project Structure

```
thesis-python/
├── data/
│   ├── exact/              # Pre-computed Shapley and Grabisch values for canonical games
│   ├── input/              # Input data for cooperative games (airport, bankruptcy, etc.)
│   └── shap/               # SHAP values and interaction matrices for NHANES-I dataset
├── results/
│   ├── Cap3/ - Cap8/       # Generated figures for each chapter
├── run/
│   ├── Cap3/               # Notebooks for Chapter 3
│   ├── Cap4/               # Notebooks for Chapter 4
│   ├── Cap5/               # Notebooks for Chapter 5
│   ├── Cap6/               # Notebooks for Chapter 6
│   ├── Cap7/               # Notebooks for Chapter 7
│   └── Cap8/               # Notebooks for Chapter 8
└── requirements.txt        # Python dependencies
```

---

## Chapter-by-Chapter Guide

### Chapter 3: Visualization Methods for Cooperative Games
**Thesis Reference:** `thesis-latex/Capitulos/03Capitulo3.tex`

This chapter establishes the formal methodology for visualizing cooperative games as networks, introducing the pair $(\Psi, R)$ where $\Psi$ represents player relevance (Shapley values) and $R$ captures pairwise interactions (Grabisch Interaction Index).

#### Notebooks in `run/Cap3/`

| Notebook | Description | Thesis Usage |
|----------|-------------|--------------|
| `airport-runway.ipynb` | **Airport Runway Game** - Cost allocation among aircraft for runway construction. Computes Shapley values and Interaction matrices for 10 aircraft (SM1-SM5, MD1-MD3, LG1-LG2). | Generates Tables 3.1 and Figures 3.1-3.2 showing network visualization with filtered representations. |
| `bankruptcy.ipynb` | **Bankruptcy Game** - Division of remaining capital among creditors with varying loan amounts. | Generates Table 3.2 and Figures 3.3-3.4 demonstrating positive interaction networks. |
| `shoes.ipynb` | **Shoes Game** - Matching left and right shoes to form pairs. Demonstrates both positive (complementary) and negative (competitive) interactions. | Generates Table 3.3 and Figures 3.5-3.6 showing bipartite structure with mixed sign interactions. |
| `myerson-messages-chain.ipynb` | **Myerson Messages Game (Chain)** - Communication restricted to chain topology. Players rewarded for forming message pairs. | Generates Table 3.7 and Figures 3.11-3.12 for the chain communication structure. |
| `myerson-messages-cycle.ipynb` | **Myerson Messages Game (Cycle)** - Cycle topology where interaction depends on distance between players. | Generates Table 3.6 and Figures 3.9-3.10 for cyclic communication. |
| `myerson-messages-star.ipynb` | **Myerson Messages Game (Star)** - Star topology with central hub player. | Generates corresponding figures for star communication structure. |
| `myerson-unitary-cost-chain.ipynb` | **Myerson Unitary Cost Game (Chain)** - Each connected component incurs unit cost. Chain topology. | Generates Table 3.5 and Figures 3.7-3.8. |
| `myerson-unitary-cost-cycle.ipynb` | **Myerson Unitary Cost Game (Cycle)** - Unitary cost allocation in cycle topology. | Generates corresponding tables and figures. |
| `myerson-unitary-cost-star.ipynb` | **Myerson Unitary Cost Game (Star)** - Unitary cost allocation in star topology. | Generates corresponding tables and figures. |

**Key Concepts Implemented:**
- Shapley value computation: $Sh_i(N, v)$
- Interaction Index: $I_{i,j}(v)$ from Murofushi-Grabisch formulation
- Filtered representation: $(\Psi_{\alpha^+, \alpha^-}, R_{\beta^+, \beta^-})$ with threshold parameters
- Matrices $I^+$, $I^-$, $I^{|\cdot|}$ for positive, negative, and absolute interactions

---

### Chapter 4: Structural Analysis of Network Representation
**Thesis Reference:** `thesis-latex/Capitulos/04Capitulo4.tex`

This chapter demonstrates the analytical power of network representation through weighted degree statistics, clustering coefficients, and community detection using the Louvain algorithm.

#### Notebooks in `run/Cap4/`

| Notebook | Description | Thesis Usage |
|----------|-------------|--------------|
| `airport-runway-fuzzy.ipynb` | Computes clustering coefficients and applies community detection to the Airport game network. | Generates Table 4.1 (metrics) and Figure 4.1 (community partition). |
| `bankruptcy-fuzzy.ipynb` | Structural analysis of the Bankruptcy game with weighted degree statistics and Gini coefficient. | Generates Table 4.2 and Figure 4.2. |
| `shoes-fuzzy.ipynb` | Analysis of bipartite structure in the Shoes game, revealing two natural communities. | Generates Table 4.3 and Figure 4.3. |
| `myerson-*-fuzzy.ipynb` | Fuzzy analysis variants for all six Myerson games (chain/cycle/star × messages/cost). | Generates corresponding metrics tables and community visualizations. |

**Key Metrics Computed:**
- **Weighted Degree** $s_i = \sum_{j \in N} r_{i,j}$ for matrices $I^+$, $I^-$, $I$, $I^{|\cdot|}$
- **Descriptive Statistics:** Min, Max, Mean, Median, Q1, Q3, Standard Deviation, Skewness, Kurtosis
- **Gini Index:** Measures inequality in relational influence distribution
- **Average Clustering Coefficient** $\bar{c}$: Local cohesion measure
- **Community Detection:** Louvain algorithm on aggregated adjacency matrix $A^{agg}$

---

### Chapter 5: Network Visualization of Fuzzy Measures
**Thesis Reference:** `thesis-latex/Capitulos/05Capitulo5.tex`

This chapter extends the visualization methodology to fuzzy measures, demonstrated through a parliamentary voting scenario where coalitions form governments.

#### Notebooks in `run/Cap5/`

| Notebook | Description | Thesis Usage |
|----------|-------------|--------------|
| `parliament-profile.ipynb` | Computes Shapley values and Interaction Indices for a 10-party parliament with 100 seats. Models government formation as a fuzzy measure where $\mu(T) = 1$ if coalition $T$ reaches majority. | Generates Table 5.1 ($\Psi = Sh$, $R = I$) and Figures 5.1-5.2 (complete and filtered visualizations). |
| `parliament-clustering-fuzzy.ipynb` | Applies community detection to identify natural political alliances. Tests different resolution parameters for Louvain algorithm. | Generates Figures 5.3-5.4 showing 4-community and 2-community partitions, Table 5.2 (metrics). |

**Key Concepts:**
- Parliamentary voting modeled as fuzzy measure: $\mu(T) = 1$ if $\sum_{j \in T} w_j \geq q$ (majority quota)
- Identification of power blocs: {A, B}, {C, D}, {E, F, G}, {H, I, J}
- Analysis of how resolution parameter affects community granularity

---

### Chapter 6: Visual Interpretation of Models with SHAP
**Thesis Reference:** `thesis-latex/Capitulos/06Capitulo6.tex`

This chapter applies Cooperative Game Theory to Explainable AI using the SHAP library on the NHANES-I survival prediction model (XGBoost).

#### Notebooks in `run/Cap6/`

| Notebook | Description | Thesis Usage |
|----------|-------------|--------------|
| `toy_example_profile.ipynb` | **Illustrative Example** - Implements the toy example from Section 6 with 10 observations and 3 variables (age, sex, blood pressure). Demonstrates computation of relevance ($R_p^i$), direction ($D_p^i$), and magnitude ($M_p^i$) metrics. | Generates Tables 6.1-6.2 and illustrative profile plots. |
| `toy_example_compute.ipynb` | Computes the step-by-step calculations for the toy example, validating formulas. | Validates equations 6.1-6.3 in the thesis. |
| `nhanesi_variable_profile.ipynb` | **Variable Influence Profiles** - Analyzes how SHAP importance varies across percentiles for key NHANES-I variables (age, sex, BMI, blood pressure, etc.). | Generates Figures 6.1-6.6 showing per-percentile relevance bar charts with impact direction coloring. |
| `nhanesi_network.ipynb` | **Network Visualization of SHAP Interactions** - Constructs the $(\Psi, R)$ network from aggregated SHAP values and interaction matrices for 79 variables. | Generates Figure 6.12 showing feature interaction network with filtering. |

**Key Analyses:**
- **Relevance per Percentile:** $R_p^i = \frac{\sum_{j \in P_p} |\phi_i(x_j)|}{\sum_{k=1}^{n} \sum_{j \in P_p} |\phi_k(x_j)|} \times 100$
- **Direction (Risk/Protective):** $D_p^i = \frac{\sum_{j \in P_p} \phi_i(x_j)}{|P_p|}$
- **Magnitude vs. Maximum:** $M_p^i$ comparing percentile impact to maximum observed impact
- **Lorenz Curves and Gini Coefficients** for importance distribution

---

### Chapter 7: Interaction Analysis with KWSHAP
**Thesis Reference:** `thesis-latex/Capitulos/07Capitulo7.tex`

This chapter introduces the KWSHAP algorithm for optimally discretizing continuous variables using Kruskal-Wallis tests on SHAP values.

#### Notebooks in `run/Cap7/`

| Notebook | Description | Thesis Usage |
|----------|-------------|--------------|
| `kw_nhanesi_shap_graph.ipynb` | **KWSHAP on NHANES-I** - Applies the KWSHAP algorithm to discretize `age` into optimal intervals, then builds the interaction network between age intervals and `sex_isFemale`. | Generates Figure 7.1 showing the interaction network between discretized age ranges and sex categories. |
| `kw_i_12_fuzzy_cat_20240106_shap_graph.ipynb` | KWSHAP analysis on fuzzy categorical dataset with specific interval (I-12). | Generates supplementary interaction visualizations. |
| `kw_sh_3_fuzzy_cat_20240106_shap_graph.ipynb` | KWSHAP for category SH-3 in the fuzzy dataset. | Supplementary analysis. |
| `kw_sh_4_fuzzy_cat_20240106_shap_graph.ipynb` | KWSHAP for category SH-4 in the fuzzy dataset. | Supplementary analysis. |

**KWSHAP Algorithm (Algorithm 1 in thesis):**
1. Order data by continuous variable
2. Test sample sizes from 5 to ⌊n/3⌋
3. For each size, create groups and compute Kruskal-Wallis p-value
4. Select $t_{opt}$ that minimizes p-value (maximum statistical significance)

**Network Construction:**
- Nodes: Intervals of continuous variables + categories of discrete variables
- Node size: Mean SHAP value for the group
- Edges: Mean SHAP interaction value between groups
- Color: Red (positive/risk), Blue (negative/protective)

---

### Chapter 8: Aggregation of Explanations via Multi-Measure Fuzzy Systems
**Thesis Reference:** `thesis-latex/Capitulos/08Capitulo8.tex`

This chapter introduces the Multi-Measure Fuzzy System (MMFS) framework for aggregating multiple local SHAP explanations into unified network representations.

#### Notebooks in `run/Cap8/`

| Notebook | Description | Thesis Usage |
|----------|-------------|--------------|
| `global_mean.ipynb` | **Global Mean Aggregation** - Aggregates all 500 NHANES-I patient SHAP values using sum-based normalization. Baseline approach treating all patients equally. | Demonstrates baseline MMFS aggregation (Definition 4-6). |
| `risk_stratified.ipynb` | **Risk-Stratified Aggregation** - Stratifies patients by actual outcome variable (y) into low-risk (66%) and high-risk (34%) groups, building separate networks for each. | Reveals how feature importance differs across risk levels. Figure 8.x. |
| `clustering.ipynb` | **Clustering-Based Aggregation** - Clusters patients by SHAP value patterns using K-Means. Uses Silhouette and Davies-Bouldin scores to determine optimal k. | Discovers patient phenotypes with distinct disease mechanisms. |
| `manual_segmentation.ipynb` | **Manual Segmentation** - Segments cohort based on hypothesis-driven criteria (males over 50 vs. rest). Demonstrates domain-expert-guided aggregation. | Shows how aggregation reveals different patterns in subgroups. |
| `median_iqr.ipynb` | **Robust Aggregation with Median/IQR** - Uses median instead of mean (robust to outliers) with γ-parameterized percentile bounds for uncertainty quantification. Gray nodes indicate mixed-sign effects. | Provides robust summary with uncertainty bounds. |

**MMFS Key Definitions:**
- **Representation Functions** $\mathcal{R}_p$: Reduce fuzzy measure from $2^n$ to $\mathbb{R}^{n^p}$
  - $\mathcal{R}_1 = Sh$: Shapley values (vector)
  - $\mathcal{R}_2 = I$: Interaction Index (matrix)
- **Node Weight Aggregation:** $v_i^* = \mathcal{N}^*_i(v^1, \ldots, v^m)$
- **Edge Weight Aggregation:** $e_{ij}^* = \mathcal{E}^*_{ij}(M^1, \ldots, M^m)$

**Aggregation Strategies Implemented:**
1. **Sum-based:** $\mathcal{N}^*_i = \frac{\sum_t Sh_i(\Delta^t)}{\sum_u \sum_t |Sh_u(\Delta^t)|}$
2. **Risk-stratified:** Separate aggregation per risk stratum
3. **Cluster-based:** K-Means on SHAP patterns
4. **Manual segmentation:** Domain-driven criteria
5. **Robust (Median/IQR):** Uncertainty-aware with tolerance parameter γ

---

## Data Sources

### `data/input/`
Input data for canonical cooperative games:
- `airport.csv`: Aircraft types and runway costs
- `bankruptcy.csv`: Creditor loan amounts
- `parliament.csv`: Political party seat allocations
- `shoes.csv`: Left/right shoe ownership

### `data/exact/`
Pre-computed game-theoretic values (from external exact computation):
- `shapley_*.csv`: Shapley values for each game
- `grabisch_*.csv`: Interaction Index matrices

### `data/shap/`
SHAP analysis data for NHANES-I model:
- `x_values.pkl`: Feature matrix (79 variables, 500 patients)
- `shap_values.npy`: SHAP values array (500 × 79)
- `shap_interaction_values.npy`: SHAP interaction matrix (500 × 79 × 79)
- `nhanesi_*.npy/pkl`: Alternative NHANES-I data files
- `fuzzy_*.npy/pkl`: Fuzzy categorical dataset variants

---

## Dependencies

```
ipykernel              # Jupyter kernel support
ipywidgets             # Interactive widgets
matplotlib             # Plotting
pandas                 # Data manipulation
scipy                  # Scientific computing (Kruskal-Wallis test)
networkx               # Network analysis
autopep8               # Code formatting
pygraphviz             # GraphViz integration for network layout
seaborn                # Statistical visualization
regex                  # Regular expressions
scikit-learn           # Clustering (K-Means) and metrics
shap                   # SHAP library for XAI
cgt_perezsechi         # Core thesis library (from GitHub)
```

---

## Core Library: cgt_perezsechi

The `cgt_perezsechi` package provides essential functionality:

### Visualization (`cgt_perezsechi.visualization`)
- `graph.draw()`: Network visualization with filtered $(\Psi, R)$ representation
- `plot.plot_shap_distribution()`: Per-percentile SHAP relevance plots
- `plot.plot_shap_lorenz_curve()`: Lorenz curves for importance distribution

### Manipulation (`cgt_perezsechi.manipulation`)
- `norm.normalize_psi()`: Normalize relevance vector to [−1, 1]
- `norm.normalize_r()`: Normalize interaction matrix
- `coding.code_interval()`: Discretize continuous variables into intervals

### Exploration (`cgt_perezsechi.exploration`)
- `sampling.sample_size_kruskal_wallis()`: KWSHAP optimal interval finder

### Modeling (`cgt_perezsechi.modeling`)
- `inequality.gini_coefficient()`: Compute Gini index for distributions

Install from source:
```bash
pip install git+https://github.com/perez-sechi/cgt@main
```

---

## Usage Example

```python
import pandas as pd
import numpy as np
from cgt_perezsechi.visualization.graph import draw
from cgt_perezsechi.manipulation.norm import normalize_psi, normalize_r

# Load Shapley values and Interaction Index
psi_1 = pd.read_csv("data/exact/shapley_value_of_airport.csv")
r_1 = pd.read_csv("data/exact/grabisch_of_airport.csv")

# Normalize
psi_2 = normalize_psi(psi_1)
r_2 = normalize_r(r_1)

# Draw filtered network
draw(
    psi=psi_2,
    r=r_2,
    positive_alpha=0.0,      # No positive node filtering
    negative_alpha=0.25,     # Filter nodes below 25% relevance
    positive_beta=0.0,       # No positive edge filtering
    negative_beta=0.0,       # No negative edge filtering
    output_path="results/Cap3/Fig1.jpg"
)
```

---

## License

MIT License

Copyright (c) 2026 Pérez-Sechi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{perezsechi2026thesis,
  author  = {Pérez-Sechi, [Name]},
  title   = {Network Visualization Methods for Cooperative Game Theory 
             and Explainable Artificial Intelligence},
  school  = {Universidad Complutense de Madrid},
  year    = {2026}
}
```

---

## Contact

For questions or issues, please open an issue in the repository or contact the thesis author.
