# WHVoidsCMB

## Description
This repository contains the analysis pipeline to correlate cosmic void catalogs with CMB Lensing maps (Planck PR3/PR4). The voids are identified using the [Sparkling](https://gitlab.com/andresruiz/Sparkling) code from the [Wen & Han](https://iopscience.iop.org/article/10.3847/1538-4365/ad409d/pdf) galaxy cluster catalog.

The main objective is to extract, stack, and model the divergence signal around these cosmic voids, fitting theoretical profiles and contrasting the underlying physics against weak gravitational lensing analysis.

To preserve the signal, this pipeline **does not perform stacking in absolute physical bins**. Instead:
1. It calculates the individual angular size of each void based on its $R_v$ and *redshift*.
2. It projects a CMB patch dynamically adapted to that size.
3. It calculates the radial profile of the convergence $\kappa$ in dimensionless units relative to the void radius: $x = r / R_v$.

## Usage and Data
**Important:** Raw data (FITS maps, original catalogs) and the `Results/` folder (which includes the massive cache of *randoms* and *jackknifes*) **are not under version control** to avoid saturating the repository.

The code assumes that the data is located in an external directory, whose *path* must be configured in the `Run_pipeline.py` file by each user before executing the scripts.

### Execution Structure
* `Run_pipeline.py`: Master orchestration script and hyperparameter configuration.
* `S1_voids.py`: Main module for $\kappa$ signal extraction and *stacking*.
* `Functions_module.py`: Mathematical functions module.
