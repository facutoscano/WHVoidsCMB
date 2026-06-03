# WHVoidsCMB

## Description

This repository contains the analysis pipeline to measure the CMB lensing
convergence signal ($\kappa$) around cosmic voids, using Planck PR3/PR4
lensing maps cross-correlated with void catalogs identified from the
[Wen & Han (2024)](https://iopscience.iop.org/article/10.3847/1538-4365/ad409d/pdf)
galaxy cluster catalog via the
[Sparkling](https://gitlab.com/andresruiz/Sparkling) void finder.

The pipeline stacks CMB lensing patches around each void in a
**physically-scaled coordinate system**: each stamp is dynamically
projected so that its angular size corresponds to a fixed multiple of
$R_v$ at the void's redshift. This avoids diluting the signal by
mixing voids of different angular sizes in a fixed-pixel grid.
Specifically:

1. The angular size of each void is computed from $R_v$ and $z$ using
   the Planck 2018 comoving distance.
2. A gnomonic projection centered on each void is extracted from the
   full-sky HEALPix map, with resolution scaled to $R_v$.
3. Stamps are stacked and the radial convergence profile is computed
   in units of $r / R_v$.

## Features

- **CMB filtering**: optional Gaussian smoothing or Wiener filtering
  ($W_\ell = C_\ell^{\kappa\kappa} / (C_\ell^{\kappa\kappa} + N_\ell^{\kappa\kappa})$)
  applied in harmonic space before stacking, using the official Planck
  noise spectrum (`nlkk.dat`).
- **Multi-seed support**: the pipeline can average results over multiple
  void catalogs generated with different random seeds from Sparkling,
  reducing sensitivity to the void-finding randomness.
- **Flexible void selection**: redshift and radius cuts, completeness
  filtering, and optional line-of-sight density contrast cuts
  (`delta_LOS`) to separate voids embedded in overdense vs. underdense
  environments.
- **Jackknife error estimation**: spatial jackknife using KMeans
  clustering on the sphere, producing a full covariance matrix for the
  radial profile.
- **Null tests**: 
  - Random rotations of the CMB map (only `common_mask` is rotated;
    the survey footprint mask is kept fixed).
  - Random void positions drawn uniformly within the survey footprint,
    preserving the $n(z)$ and $n(R_v)$ distributions.
- **Caching**: null test results are cached to disk and reused across
  runs with the same configuration. `force_rerun=True` bypasses the
  cache.
- **Organized outputs**: each run generates its own folder named after
  all relevant hyperparameters (release, binning, redshift range,
  radius range, $R_v$ scale, delta_LOS cut, and filter mode), ensuring
  reproducibility and easy comparison between configurations.

## Repository Structure

| File | Description |
|------|-------------|
| `Pipeline_voids.py` | Master script: configuration, logging, and pipeline orchestration |
| `S1_voids.py` | Step 1: void selection, binning, stacking, null tests, and result saving |
| `Functions_module.py` | All core functions: filtering, projection, stacking, jackknife, plotting |
| `null_test_plots.py` | Standalone null test visualization utilities |

## Usage and Data

**Raw data are not under version control** (FITS lensing maps, void
catalogs, and the `Results/` cache). Each user must set the paths in
`Pipeline_voids.py` before running.

### Required data files

- `KAPPA_{release}klm_MV.fits` — Planck [PR3](http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.COSMOLOGY_OID=131&COSMOLOGY.FILE_ID=MV.tgz) or [PR4](https://github.com/carronj/planck_PR4_lensing/releases/tag/Data)
  lensing convergence $\kappa$ alm (MV estimator)
- `Common_mask_PR4Lensing_2048.fits` — Planck lensing
  analysis [mask](https://github.com/carronj/planck_PR4_lensing/releases/tag/Data) at nside=2048
- `nlkk_PR3_MV.dat` — Planck noise spectrum for Wiener
  filtering (columns: $\ell$, $N_\ell^{\kappa\kappa}$,
  $C_\ell^{\kappa\kappa} + N_\ell^{\kappa\kappa}$), available from
  the [Planck Legacy Archive](https://pla.esac.esa.int/)
- `WenHan_voids_z0.6/voids_z0.6_{NNN}.dat` — void catalogs
  for each random seed (multi-seed mode) - Under request
- `WenHan_voids.dat` — single void catalog (single-seed mode) - Under request

### Configuration

All hyperparameters are set in the `config` dict in `Pipeline_voids.py`:

```python
config = {
    'release':    'PR4',      # 'PR3' or 'PR4'
    'N_seeds':    10,         # number of Sparkling random seeds (None = single catalog)
    'delta_value': None,      # None: no cut | >0: delta_LOS > value | <0: delta_LOS < -value
    'zmin': 0.05, 'zmax': 0.583,
    'rmin': 35.0, 'rmax': 62.7,   # Mpc/h
    'max_Rvoid':  2.5,        # max stacking radius in units of R_v
    'npix_stamp': 400,        # stamp size in pixels
    'filter_mode': 'wiener',  # 'none', 'gaussian', or 'wiener'
    'binning_mode': 'redshift', # 'redshift' or 'radius'
    'n_bins': 3,
    'exec_mode': 'errors',    # 'errors' (jackknife) or 'no_errors'
    'n_subsamples': 30,       # jackknife spatial regions
    'n_rand_factor': 10,      # randoms per void for null test
    'n_rotations': 10,        # CMB rotations for null test
    'force_rerun': False,     # True to ignore cached null tests
}
```

### Running

```bash
python Pipeline_voids.py
```

Output is saved to `Results/{file_suffix}/`, where `file_suffix`
encodes all relevant parameters for traceability.

## Output files

Each run produces:

| File | Content |
|------|---------|
| `Voids_Lensing_Profiles_{suffix}.pdf` | Stacked $\kappa$ maps and radial profiles with null test bands |
| `JK_Correlation_{suffix}.pdf` | Jackknife profiles and correlation matrices per bin (errors mode only) |
| `Data_FullRun_{suffix}.pkl` | Full results dict (profiles, covariance, maps, null tests, config) |
| `WHVoidsCMB_pipeline_log.txt` | Full run log |
| `WHVoidsCMB_pipeline_config.json` | Config snapshot for reproducibility |

## Dependencies

- `numpy`, `scipy`, `matplotlib`
- `healpy`
- `astropy`
- `pandas`
- `scikit-learn` (KMeans jackknife)
