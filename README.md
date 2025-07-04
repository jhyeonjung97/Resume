# Computational Materials Science Portfolio

This repository contains a curated collection of my computational materials science and chemistry projects, demonstrating expertise in density functional theory (DFT) calculations, machine learning applications, data analysis, and materials informatics.

## 🧪 Research Areas

- **Electrochemical Catalysis**: Oxygen evolution/reduction reactions (OER/ORR), hydrogen evolution, nitrate reduction
- **Materials Discovery**: High-entropy oxides, single-atom catalysts, alloy systems
- **Machine Learning**: Property prediction, feature engineering, model optimization
- **Computational Chemistry**: VASP calculations, atomic structure manipulation, electronic structure analysis

## 📁 Project Structure

### 🔬 VASP Calculations (`VASP_Calculations/`)
Automated DFT calculations using VASP for materials science applications:
- **Bulk_Relaxation.py** / **Bulk_Static.py**: Bulk material structure optimization and static calculations
- **Slab_Relaxation.py** / **Slab_Static.py**: Surface slab structure calculations
- **Slab_Constant_Electrode_Potential.py**: Electrochemical calculations with constant electrode potential method

### 🤖 Machine Learning (`Machine_Learning/`)
Advanced ML models for materials property prediction:
- **Oxide_Formation_Energy_Prediction.py**: Gaussian Process Regression for oxide formation energy prediction (1,220 lines)
- **Oxide_Formation_Energy_Linear_Regression.py**: Linear regression models with feature selection
- **Oxide_Coordination_Classification.py**: Classification of oxide coordination environments
- **Feature_Importance_Forward_Selection.py**: Feature selection algorithms
- **Data_Parsing_*.py**: Data preprocessing scripts for bulk/slab structures and Mendeleev properties
- **Data_Plotter.py**: Visualization tools for ML results

### 📊 Data Analysis & Visualization (`Data_Parsing_and_Plotting/`)
Comprehensive data analysis and scientific visualization:
- **Electrochemical Analysis**:
  - `OER_2D_Volcano.py` / `ORR_2D_Volcano.py`: 2D volcano plots for electrocatalysis
  - `NO3RR_1D_Volcano.py` / `ORR_1D_Volcano.py`: 1D volcano plots
  - `OXR_Overpotential.py`: Overpotential calculations
- **Catalyst Analysis**:
  - `SAC_Formation_Energy.py` / `SAC_Adsorption_Energy.py`: Single-atom catalyst analysis
  - `Hydrogen_Binding_Energy.py`: Hydrogen binding energy calculations
- **Materials Analysis**:
  - `Alloy_Bader_Charge.py` / `Alloy_Lattice_Grid_Energy.py`: Alloy system analysis
  - `High_Entropy_Oxide_Energy.py`: High-entropy oxide calculations
  - `Density_of_States_Plotter.py`: Electronic structure visualization
- **API Tools**: `API_OpenMP_*.py`: OpenMP-based API tools

### ⚛️ Atomic Structure Tools (`ASE/`)
Atomic Simulation Environment (ASE) utilities for structure manipulation:
- **Structure Manipulation**: `Edit_Slab_Cell.py`, `Scale_Cell_Size.py`, `Sort_Atoms.py`
- **Analysis Tools**: `Bader_Charge.py`, `Detect_Atom_Shift.py`, `Split_DOS.py`
- **Specialized Tools**: `High_Entropy_Oxide_Combination.py`, `Hydrogen_Hopping_from_Cation_Cluster.py`

### 🗺️ Pourbaix Diagrams (`Hybrid_Pourbaix_Diagram/`)
Electrochemical stability diagram generation:
- **Pymatgen_Pourbaix_Plotter/**: Pymatgen-based Pourbaix diagram generation
- **Matplotlib_Colormesh/**: Custom matplotlib-based visualization

### 🛠️ General Tools (`General_Tools/`)
Utility scripts for workflow automation:
- **Backup_Data.py**: Data backup automation
- **Gits_Sync.py**: Git repository synchronization
- **Spread_to_Subdir.py**: File organization utilities

## 🚀 Key Features

### Advanced Machine Learning
- **Gaussian Process Regression** with custom kernels and uncertainty quantification
- **Ensemble Methods**: Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Feature Engineering**: Permutation importance, correlation analysis, forward selection
- **Cross-validation** and hyperparameter optimization using Bayesian search

### Electrochemical Analysis
- **Volcano Plot Generation**: 1D and 2D volcano plots for electrocatalysis
- **Overpotential Calculations**: OER, ORR, and other electrochemical reactions
- **Scaling Relationships**: Linear scaling relationships for adsorbate binding energies
- **Constant Electrode Potential**: DFT calculations with electrochemical boundary conditions

### Materials Informatics
- **Property Prediction**: Formation energy, cohesive energy, adsorption energies
- **Structure Analysis**: Coordination environment classification, Bader charge analysis
- **High-Throughput Screening**: Automated workflows for materials discovery

## 💻 Technical Stack

- **Programming Languages**: Python
- **Scientific Computing**: NumPy, SciPy, Pandas, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, Gaussian Processes
- **Materials Science**: ASE, Pymatgen, VASP
- **Visualization**: Matplotlib, Seaborn, Custom plotting utilities
- **Data Processing**: Pandas, NumPy, Custom parsing scripts

## 📈 Research Impact

This portfolio demonstrates expertise in:
- **Computational Materials Science**: DFT calculations, electronic structure analysis
- **Machine Learning for Materials**: Property prediction, feature engineering, model optimization
- **Electrochemical Catalysis**: Understanding reaction mechanisms and catalyst design
- **High-Throughput Screening**: Automated workflows for materials discovery
- **Scientific Visualization**: Publication-quality figures and data presentation

## 🔬 Applications

The tools and methods developed here have applications in:
- **Catalyst Design**: Understanding and optimizing electrocatalytic performance
- **Materials Discovery**: Screening for new materials with desired properties
- **Energy Storage**: Battery materials and electrode design
- **Electrochemical Systems**: Fuel cells, electrolyzers, and energy conversion devices
- **Materials Informatics**: Data-driven approaches to materials science

---

*This portfolio represents a comprehensive collection of computational materials science tools and methodologies, showcasing expertise in both fundamental computational chemistry and modern data science approaches to materials research.*