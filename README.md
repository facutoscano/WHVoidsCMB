# WHVoidsCMB

Stacking measurements of the CMB around cosmic voids. The repository makes
two types of analyses:

1. the **CMB lensing convergence** ($\kappa$) profile of voids, and
2. an **orientation-resolved thermal Sunyaev–Zel'dovich** stacking (tSZ,
   Compton-$y$) of the galaxy clusters that sit in the void walls.

The voids come from the [Sparkling](https://gitlab.com/andresruiz/Sparkling)
finder run on the [Wen & Han (2024)](https://iopscience.iop.org/article/10.3847/1538-4365/ad409d/pdf)
cluster catalogue (WH), and on a BOSS galaxy sample. Both analyses
stack Planck full-sky maps on gnomonic patches centred on each object, with the
patch size scaled to the void radius, and estimate uncertainties with a spatial
jackknife plus null tests.


## Voids and multi-seed
Sparkling is stochastic: it grows voids outward from random seed points, so a
single run is one realization of the catalogue. 
We run it 100 times and keep all 100 seed catalogues. 
There are two ways to use them:

- **Concatenation** — stack every detection from every seed. A void found in
  many seeds simply enters the stack many times, so the more reproducibly
  detected voids get up-weighted.
- **Merge** — collapse the 100 realizations into a single deduplicated
  catalogue with DBSCAN (`void_seed_merge.py`).

Running both (`seed_mode = 'both'`) and comparing the profiles tells you whether
the multiplicity weighting of the concatenation biases the signal, or whether it
just adds noise. The two catalogues are binned with a *single, shared* set of
quantile edges so that bin $i$ covers the same redshift (or radius) range in
either case.

The DBSCAN linking length `merge_eps_mpch` is a physical scale:
it should be a few times the typical distance a void centre wanders between 
seeds (of order a few Mpc/h), and well below the void–void separation so 
distinct neighbours are not fused. 
`void_seed_merge.sanity_report` prints the diagnostics used to set it—median
seeds per void, the fraction of discarded (noise) detections, and a flag for over-merging.
For the current WH and BOSS catalogues, `merge_eps_mpch ≈ 8` with `merge_min_frac ≈ 0.3` 
(a void must appear in at least 30% of the seeds) gives a clean merge.




## The lensing measurement
Every stamp is projected so that its angular size corresponds to a fixed multiple
of $R_v$ at the void's redshift:

1. the angular size of $\texttt{max\_Rvoid} \times R_v$ is computed from the
   Planck 2018 comoving distance at $z$;
2. a gnomonic patch centred on the void is cut from the HEALPix map with the
   resolution set by that angular size;
3. patches are co-added weighting each pixel by how many stamps covered it, and
   the radial profile is measured in units of $r/R_v$.

The convergence map can be Wiener-filtered in harmonic space before stacking,
$W_\ell = C_\ell^{\kappa\kappa} / (C_\ell^{\kappa\kappa} + N_\ell^{\kappa\kappa})$,
using the Planck PR3 noise spectrum (`nlkk`), or Gaussian-smoothed, or left alone.

Uncertainties come from a spatial jackknife: the voids are split into
`n_subsamples` regions with KMeans on the sphere, and the leave-one-out
resampling gives the full covariance of the profile. On top of that, two null
tests calibrate what a non-detection looks like:

- **Rotations** — the $\kappa$ map is rotated by random angles and re-stacked at
  the true void positions.
- **Randoms** — voids are replaced by random positions inside the footprint,
  preserving the $n(z)$ and $n(R_v)$ distributions. Optionally, disks of
  `random_excl_factor` $\times R_v$ around the real voids are removed from the
  random pool so the null cannot accidentally pick up real void signal.

### Caching and run folders
All lensing output lives under `Results/lensing/`. Stacking is expensive, so
every signal and null-test result is cached to `Results/lensing/Cache_Stacks/`
(under a per-catalogue subfolder, so WH and BOSS never collide) and reused unless
`force_rerun = True`. Each run also gets its own folder under `Results/lensing/`
whose name encodes every parameter that matters — catalogue, release, binning,
redshift and radius ranges, $R_v$ scale, the density-contrast cut and the
filter — so different configurations never overwrite each other and are trivial
to compare.




## The tSZ oriented measurement
`ClusterXVoids_tSZ.py` drives a different question: do the clusters that make up
a void wall show an anisotropic hot-gas (Compton-$y$) signal that knows about the
void? The steps are:

1. **Merge** the seed catalogues into one void catalogue (`void_seed_merge.py`).
2. **Associate** clusters to voids (`cluster_void_assoc.py`): a cluster is kept
   if it lies in the shell $[\,f_{\min}, f_{\max}\,]\times R_v$ of a void, using
   comoving transverse and line-of-sight separations. Clusters too close to the
   line of sight (inclination $> 30°$) are dropped, because projection there
   swamps the effect we want. For each kept cluster the position angle toward its
   void is recorded.
3. **Stack** the Compton-$y$ map on each cluster, rotating the stamp so the void
   always points to the same place, and split the ring into octants
   (`oriented_stacking.py`). From the eight octants we build two profiles:
   - a *facing-the-void* dipole (the side pointing at the void minus the opposite
     side), and
   - a *filament* elongation (the direction tangential to the wall minus the
     radial one), which tests whether the cluster is stretched along the wall
     rather than toward the void.

Because the sign and offset that put the void "to the left" of the stamp depend
on healpy's projector convention, the orientation must be calibrated once with
`oriented_stacking.calibrate_orientation()` and the resulting `(pa_sign,
pa_offset)` fed back into the config. Significance is assessed with an empirical
Monte-Carlo null suite: shuffled position angles, random position angles, and
rigid rotations of the catalogue over the real map.

The clusters are always the Wen & Han ones, but the voids they are hung on can be
either catalogue (`void_catalog = 'WH'` or `'BOSS'`); BOSS positions are rotated
from $(\mathrm{RA}, \mathrm{Dec})$ to galactic first. Running both is a check
that the anisotropy survives against a different void sample. Each run writes to
`Results/tSZ/{catalogue}_{release}_{...}/` — the same suffix convention as the
lensing side — holding the association map, the oriented-stack plots and the
`.npz` results. If those `.npz` are already there they are not recomputed unless
`force_rerun = True`.



## Repository structure
| File | Role |
|------|------|
| `Pipeline_voids.py` | Lensing driver: configuration, logging, and choice of backend |
| `S1_voids.py` | Serial step 1 — void selection, binning, stacking, nulls, saving |
| `S1_voids_parallel.py` | Parallel step 1 — same, plus WH/BOSS choice and the DBSCAN merge |
| `Functions_module.py` | Core routines: filtering, projection, stacking, profiles, jackknife, nulls, plots |
| `Parallel_module.py` | Multi-core twin of the per-bin stacking (fork-based process pool) |
| `void_seed_merge.py` | DBSCAN de-duplication of the multi-seed catalogues into one catalogue |
| `null_test_plots.py` | Standalone diagnostic maps (void overlap, rotations, random positions) |
| `ClusterXVoids_tSZ.py` | tSZ driver: merge → cluster–void association → oriented stacking |
| `cluster_void_assoc.py` | Cluster-to-void association in a 2.5D shell, with the LOS-inclination cut |
| `oriented_stacking.py` | Void-oriented octant stacking, jackknife, orientation calibration, null suite |


## Configuration
Everything for the lensing run lives in the `config` dict in
`Pipeline_voids.py`. The parameters that change the result:

```python
config = {
    'release':      'PR4',      # 'PR3' or 'PR4'
    'void_catalog': 'WH',       # 'WH' (galactic l,b) or 'BOSS' (equatorial ra,dec)
    'N_seeds':      10,         # Sparkling seed catalogues to use (None = single catalogue, max = 100)

    'seed_mode':          'both',  # 'concat' | 'merge' | 'both'
    'merge_eps_mpch':     8.0,     # DBSCAN linking length [Mpc/h]
    'merge_min_frac':     0.3,     # min fraction of seeds for a void to survive
    'merge_use_catalog_xyz': False,# recompute comoving xyz from (l,b,z)

    'delta_value':  None,       # None: no cut | >0: delta_23 > value | <0: delta_23 < -value
    'zmin': 0.2, 'zmax': 0.6,
    'rmin': 20.0, 'rmax': 60.0, # Mpc/h

    'max_Rvoid':    2.5,        # max stacking radius in units of R_v
    'npix_stamp':   400,        # stamp side in pixels
    'filter_mode':  'wiener',   # 'none' | 'gaussian' | 'wiener'
    'binning_mode': 'redshift', # 'redshift' or 'radius'
    'n_bins':       1,

    'exec_mode':    'errors',   # 'errors' (jackknife) or 'no_errors'
    'n_subsamples': 30,         # jackknife regions
    'n_rand_factor': 15,        # random positions per void
    'n_rotations':  30,         # map rotations
    'random_pool':  'full',     # 'full' (whole footprint) or 'survey'
    'random_excl_factor': None, # exclude disks of value*Rv around real voids (None disables)

    'exec_backend': 'parallel', # 'serial' -> S1_voids | 'parallel' -> S1_voids_parallel
    'n_workers':    100,        # worker processes (None = all cores)

    'force_rerun':  False,      # ignore caches and recompute
}
```

`merge_*`, `void_catalog` and the BOSS handling only take effect in the parallel
backend. The `rmin_fit_mpc`, `rmax_fit_mpc` and `mcmc_*` entries are placeholders
for the fitting/MCMC steps, which are stubbed out for now.

The tSZ analysis has its own `config` at the bottom of `ClusterXVoids_tSZ.py`
(`void_catalog`, `release`, `force_rerun`, the shell limits `f_min`/`f_max`, the
LOS-inclination cut, the richness scan, the octant orientation
`pa_sign`/`pa_offset`, and the Monte-Carlo null settings). Its `output_folder` is
the `Results/` base; the code builds the `tSZ/{catalogue}_{release}_{...}/`
subfolder itself.


## Running
```bash
python Pipeline_voids.py        # lensing profiles
python ClusterXVoids_tSZ.py     # tSZ oriented stacking
```

Everything lands under `Results/`, split into two trees:

```
Results/
├── lensing/                      # Pipeline_voids.py
│   ├── Cache_Stacks/             # stacking cache (per catalogue)
│   └── {catalogue}_{release}_{...}/   # one folder per run
└── tSZ/                          # ClusterXVoids_tSZ.py
    └── {catalogue}_{release}_{...}/   # one folder per run
```


## Outputs
Lensing, inside each `Results/lensing/{suffix}/` folder:

| File | Content |
|------|---------|
| `Stacked_Maps_NullTests_{suffix}.pdf` | Stacked $\kappa$ maps, radial profiles with null bands, and S/N |
| `JK_Profile_Correlation_{suffix}.pdf` | Jackknife profiles and correlation matrices (errors mode) |
| `Seed_Consistency_{suffix}.pdf` | Per-seed vs. combined profiles (multi-seed runs) |
| `Merge_vs_Concat_{suffix}.pdf` | Merged vs. concatenated profile and their difference (`seed_mode = 'both'`) |
| `Data_FullRun_{suffix}.pkl` | Full results: profiles, covariance, maps, nulls, and the config |

tSZ, inside each `Results/tSZ/{suffix}/` folder (one row per richness/`delta_23` sample):

| File | Content |
|------|---------|
| `assoc_mollview.pdf` | Voids and their associated clusters on the sphere |
| `tSZ_oriented_{sample}.pdf` | Oriented stack, facing-void and filament profiles, null bands |
| `tSZ_oriented_{sample}.npz` | Profiles, covariances and Monte-Carlo null distributions |

`Results/lensing/` also holds `pipeline_run.log` and
`WHVoidsCMB_pipeline_config.json`, which record the run for reproducibility.


## Data
The raw inputs are not under version control (FITS maps, catalogues, and the
`Results/` cache). Set the paths in the relevant `config` before running.


Lensing:
- `CMB/Lensing/KAPPA_{release}klm_MV.fits` — Planck
  [PR3](http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.COSMOLOGY_OID=131&COSMOLOGY.FILE_ID=MV.tgz)
  or [PR4](https://github.com/carronj/planck_PR4_lensing/releases/tag/Data)
  lensing $\kappa$ alm (MV estimator)
- `CMB/Lensing/Common_mask_PR4Lensing_2048.fits` — Planck lensing
  [mask](https://github.com/carronj/planck_PR4_lensing/releases/tag/Data) at nside 2048
- `CMB/Lensing/nlkk_PR3_MV.dat` — Planck lensing noise spectrum for the Wiener
  filter (columns $\ell$, $N_\ell^{\kappa\kappa}$,
  $C_\ell^{\kappa\kappa}+N_\ell^{\kappa\kappa}$)


tSZ:
- `CMB/Compton_y/Compton-SZMap-NILC-ymap_2048_PR4.fits` — Planck NILC Compton-$y$ map
- `CMB/Temperature/Common_mask_Temperature_2048.fits` — temperature common mask


Voids (available on request):
- `CATALOGOS/WenHan_voids_z0.6/voids_z0.6_{NNN}.dat` — 100 WH seed catalogues
- `CATALOGOS/BOSS_voids_merge/voids_boss_{NNN}.dat` — 100 BOSS seed catalogues
- `CATALOGOS/WenHan_voids.dat` — single WH catalogue (used when `N_seeds = None`)
- `CATALOGOS/WenHan_Clusters.dat` — the Wen & Han cluster catalogue


## Dependencies
`numpy`, `scipy`, `matplotlib`, `healpy`, `astropy`, `pandas`, and
`scikit-learn` (KMeans jackknife and the DBSCAN merge).
