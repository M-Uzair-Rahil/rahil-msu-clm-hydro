## CHM v0.1.0 – Initial Release

Initial public release of MCH (CLM-Hydr-MSU).

---

## Overview

CHM is a Python package for generating Latin Hypercube Sampling (LHS) parameter ensembles for hydrologic calibration in the Community Land Model (CLM5). Developed at Michigan State University (MSU), It enables reproducible and scalable parameter exploration workflows for integrated crop–hydrology simulations.

---

## Key Features

- Latin Hypercube Sampling (LHS) for hydrologic parameters  
- Automated generation of CLM parameter NetCDF ensembles  
- Supports hydrologic P-type and N-type parameters  
- Optional FMAX perturbation in CLM surfdata  
- Built-in hydrologic parameter definitions (no external CSV required)  
- Workflow-ready outputs for calibration experiments  
- Progress bar and execution summary  

---

## Installation

It is recommended to install CHM in a dedicated conda environment.

### Create environment

conda create -n rmch python=3.10 -y  
conda activate rmch  
export PYTHONNOUSERSITE=1  

### Install RMCH

python -m pip install --upgrade pip  
python -m pip install --no-user git+https://github.com/M-Uzair-Rahil/rahil-msu-clm-hydro.git  

---

## Usage

### Basic example

import rahil

out = rahil.generate_lhs(
    Ninit=70,
    seed=10,
    base_surf_dir="/absolute/path/to/base_surfdata.nc",
    param_nc_dir="/absolute/path/to/base_paramfile.nc",
    output_dir="./output"
)

---

## Function Arguments

- Ninit → Number of LHS samples  
- seed → Random seed for reproducibility  
- base_surf_dir → Path to CLM surfdata NetCDF  
- param_nc_dir → Path to CLM parameter NetCDF  
- output_dir → Output directory  

---

## Example Output Structure

output/
└── pe_hydrology/
    └── iter_0/
        ├── paramfile_combined/
        ├── namelist_txt/
        ├── workflow/
        └── surfdata_ensemble/

---

## Output Description

The package generates:

- CLM parameter NetCDF files (hydro P-type)  
- Hydrology namelist text files (hydro N-type)  
- Optional surfdata ensemble (FMAX perturbation)  

Workflow files:

- joint_param_list.txt  
- main_run.txt  

---

## Notes

- Use absolute paths (e.g., /glade/...) on HPC systems  
- No external CSV file is required  
- This is the initial release  

---

## Author

Mohammad Uzair Rahil  
PhD Student, Civil & Environmental Engineering  
Michigan State University  

---

## Citation

Rahil, M. U. (2026). rahil-msu-clm-hydro (Version 1.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.19575755

---
## Plot Stratification

To visualize the stratification of the Latin Hypercube Sampling (LHS), use the code below.  
Only the `output_dir` needs to be specified.

```python
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
output_dir = "add the same output directory here as before"
# ============================================================
location = "pe_hydrology" # dont change thir or anything else.
iteration = 0

param_file = os.path.join(
    output_dir,
    location,
    f"iter_{iteration}",
    "workflow",
    "pe_hydrology_0.joint_param_list.txt"
)

out_fig = os.path.join(
    output_dir,
    location,
    f"iter_{iteration}",
    "workflow",
    "lhs_stratification.png"
)

if not os.path.exists(param_file):
    raise FileNotFoundError(f"Could not find parameter file:\n{param_file}")

df = pd.read_csv(param_file, sep=None, engine="python")

print("Columns found:")
print(df.columns.tolist())

exclude_cols = {
    "case_name", "case", "sample_id", "run_id", "index", "Unnamed: 0"
}

param_cols = []
for col in df.columns:
    if col in exclude_cols:
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        param_cols.append(col)

if len(param_cols) == 0:
    raise ValueError("No numeric parameter columns found.")

print("\nParameter columns used for plotting:")
print(param_cols)

n_params = len(param_cols)
ncols = 2
nrows = math.ceil(n_params / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
axes = axes.flatten()

for i, col in enumerate(param_cols):
    ax = axes[i]

    vals = df[col].dropna().values
    vals_sorted = sorted(vals)
    n = len(vals_sorted)

    vmin = min(vals_sorted)
    vmax = max(vals_sorted)

    ax.scatter(vals_sorted, [1] * n, s=35)

    for k in range(n + 1):
        x = vmin + (vmax - vmin) * k / n
        ax.axvline(x, linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_title(col, fontsize=11)
    ax.set_yticks([])
    ax.set_ylim(0.85, 1.15)
    ax.set_xlabel("Sample value")
    ax.grid(False)

    ax.text(
        0.01, 0.92,
        f"min={vmin:.4g}\nmax={vmax:.4g}\nN={n}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8")
    )

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("LHS Stratification of Sampled Parameters", fontsize=14, y=0.98)
plt.tight_layout()

plt.savefig(out_fig, dpi=300, bbox_inches="tight")

plt.show()

print(f"\nFigure saved to:\n{out_fig}")
