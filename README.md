# WHVoidsCMB

Stacked CMB-lensing convergence ($\kappa$) profiles of cosmic voids.

The voids are identified with the [Sparkling](https://gitlab.com/andresruiz/Sparkling)
finder on two tracers: the [Wen & Han (2024)](https://iopscience.iop.org/article/10.3847/1538-4365/ad409d/pdf)
cluster catalogue (WH) and a BOSS galaxy sample. The convergence is taken from
Planck (several reconstructions) and from ACT DR6. The same machinery runs on the
Websky simulation to provide a theory-side comparison with identical selection,
geometry and estimator.

The orientation-resolved tSZ analysis of the clusters in the void walls is kept
outside this repository (`WHVoidsCMB_tSZ/`), for a later paper.


## Void catalogues

**Multi-seed catalogues.** Sparkling grows voids from random seed points, so one run
is one realization of the catalogue. It is run 100 times. The 100 realizations are
either *concatenated* (reproducible voids are effectively up-weighted) or *merged*
into a single catalogue by grouping the void centres in comoving space with
DBSCAN: a void is kept if it appears in at least a fraction `merge_min_frac` of the
seeds within a linking length `merge_eps_mpch` (of order the wandering of a void
centre between seeds, well below the void-void separation). `seed_mode = 'both'`
runs the two and compares them; both are binned with the same edges.

**Selection.** Redshift and radius ranges, Sparkling completeness, and optionally
the density contrast $\delta_{23}$ (`delta_value`: $\delta_{23} > v$ for $v \ge 0$,
$\delta_{23} < v$ for $v < 0$). A void is used only if at least
`min_footprint_coverage` of its disc of radius `max_Rvoid` $\times R_v$ lies inside
the convergence map mask.

**Catalogue choice.** `void_catalog = 'WH'`, `'BOSS'` or `'BOTH'`. With `'BOTH'`
every requested measurement is done for the two catalogues and an extra figure
shows them side by side, one column per catalogue.


## Convergence maps

| Name | Product |
|------|---------|
| `PLANCK_PR4` | PR4 (NPIPE) minimum-variance quadratic estimator |
| `PLANCK_PR3` | Planck 2018 minimum-variance quadratic estimator |
| `PLANCK_inhom` | Planck 2018, inhomogeneous-noise filtering |
| `PLANCK_SZ` | Planck 2018 reconstruction without the tSZ-cluster mask |
| `PLANCK_SZ_deproj` | Planck 2018 reconstruction from tSZ-deprojected CMB maps |
| `PLANCK_CIBcomb` | Planck 2018 lensing tracer combining the quadratic estimator with the GNILC 545 GHz CIB map (`COM_Lensing-CIBcomb`) |
| `ACT` | ACT DR6 baseline reconstruction |

Quadratic-estimator maps are Wiener filtered,
$W_L = C_L^{\kappa\kappa}/(C_L^{\kappa\kappa}+N_L^{\kappa\kappa})$, with the noise
spectrum of each reconstruction (for PR4, an empirical $N_L$ from
`make_wiener_PR4.py`). Products that are already filtered (`PLANCK_CIBcomb` and the
ACT map) are used as delivered. Note that `PLANCK_CIBcomb` is not a CIB-cleaned
reconstruction: it contains the CIB as an additional tracer of the matter field,
so its void profile has a different transfer and content from the others and is
not a test of CIB contamination.

**Groups.** `cmb_map` can be one map or a group:

* `'All_Planck'`: the six Planck products;
* `'All'`: the six Planck products plus ACT;
* `'ACT-PR4'`: PR4 and ACT.

Within a group every map is measured on **the same voids**: the void selection and
the randoms use the intersection of the masks of all the maps in the group, so the
differences between profiles come from the maps only. Note that `'All'` therefore
works on the ACT footprint.


## The measurement

The profile is measured in rings of $r/R_v$ out to `max_Rvoid`, with every void
weighted equally (masked area down-weighted). Two estimators are available
(`profile_method`):

* `'healpix'` (default): the map pixels around each void are assigned to the rings by
  their exact angular distance to the centre. Every map pixel is used, the result
  does not depend on any resampling choice, and adding a void changes the profile only
  by its own contribution.
* `'stamp'`: a patch of the map is projected on the tangent plane, centred on the void
  and scaled to $2\,\texttt{max\_Rvoid}\,R_v$; patches are co-added and the profile is
  read from the stacked image. When the stamp pixel is coarser than the map pixel
  (ACT), the profile picks up extra noise from the sampling of small-scale power.

In both cases the stacked image shown in the figures comes from the stamps.

**Background (randoms).** The profile of random positions is subtracted. The
randoms reproduce the void sample as closely as possible:

* they fall inside the angular footprint of the tracer survey (WH clusters, or
  BOSS) **and** inside the convergence mask (or the common mask of a group);
* each random takes the $(z, R_v)$ of a void of the sample;
* each random passes the same coverage criterion as the voids.

This way the mean of the randoms is the mean of the filtered map over the region
where the voids actually are. Drawing randoms over the whole map mask instead
would subtract the mean of a different region of the sky.

`bkg_mode` sets the background subtraction:

* `'randoms'`: $\kappa_{\rm net} = \kappa - \langle\kappa_{\rm rand}\rangle$;
* `'outer'`: the mean of the profile at large radii (`bkg_outer`, default
  $2$–$2.5\,R_v$) is subtracted, forcing the profile to zero there;
* `'randoms+outer'`: both.

The same operation is applied to every jackknife and random profile, so the
covariances always refer to the net profile.

**Uncertainties.** A spatial jackknife (void sample split into `n_subsamples` sky
regions) gives the covariance of the stacked profile. The scatter of the random
stacks gives the covariance from map noise and cosmic variance at fixed void
positions; it is the one used in the $\chi^2$ comparisons, with the Hartlap
correction and an empirical PTE calibrated on the random stacks themselves.

**Null tests.**

* *Randoms*: the stacks at random positions themselves (expected zero after
  subtraction of their mean).
* *Rotations*: the void catalogue is rigidly shifted in longitude over the map
  (full-sky maps only).


## Websky

`Websky.py` runs the same measurement on voids found in Websky and on the Websky
convergence, using the same cuts and settings as the data. The Websky voids are
cut to the WH or BOSS survey footprint (`void_catalogs = 'WH'`, `'BOSS'` or `'BOTH'`),
and the same footprint defines the randoms. Cases (`surveys`):

* `FULL`: noiseless $\kappa$, full sky, all the Websky voids;
* `PR4`: noiseless $\kappa$ on the PR4 mask, and $\kappa$ + PR4 noise + the PR4
  Wiener filter (the direct counterpart of the data);
* `ACT`: noiseless $\kappa$ on the ACT footprint, and a lensing-free ACT-like map
  (Websky foregrounds through the ACT reconstruction and filter) that measures the
  foreground bias around voids;
* `ACT-PR4`: the PR4-like map on the ACT footprint.

All runs on the ACT footprint use the ACT $\times$ PR4 coverage, as the `'ACT-PR4'`
data mode, so they share the voids. When matching data runs exist, the Websky
profiles are compared with them (profiles and $\chi^2$ with the random
covariances). There is no lensing prediction through the ACT filter and noise:
the ACT footprint is compared through the PR4-like map.

Websky uses its own cosmology (flat, $h=0.68$, $\Omega_m=0.31$) for every distance
in the pipeline.


## Running

```bash
python Pipeline_voids.py    # data: lensing profiles (config at the top of the file)
python Websky.py            # simulation, same settings as Pipeline_voids.py
```

Auxiliary scripts:

| Script | Purpose |
|--------|---------|
| `make_wiener_PR4.py` | empirical PR4 noise spectrum used by the Wiener filter |
| `diagnostic_plots.py` | the selected sample (n(z), n(R)), sky coverage, random region, filter and map |
| `null_test_plots.py` | stamps overlap, rotated catalogues and random realizations on the sky |
| `check_map_consistency.py` | relative transfer of the Planck products (cross-correlation with WH clusters) and stack of map differences on the voids |
| `websky_maps_plot.py` | Websky voids and maps on the sky |

Every result is stored under `Results/`: one folder per run, whose name records
the catalogue, map, selection, filter, coverage and background choice; each
contains the figures and a `Data_FullRun_*.pkl` with all profiles, covariances,
random and rotation stacks, and the configuration. Comparison figures (map groups,
catalogue columns, ACT vs PR4, Websky vs data) are written next to the run folders.


## Inputs

Not under version control; set the paths in the configs.

* Planck lensing products (PR3, PR4, inhomogeneous filtering, SZ, SZ-deprojected,
  CIB-combined), their masks and noise spectra;
* ACT DR6 convergence map and mask;
* Sparkling void catalogues (100 seeds) for WH and BOSS, and the WH cluster
  catalogue (survey footprint);
* Websky convergence, voids and the ACT-like foreground map.


## Dependencies

`numpy`, `scipy`, `matplotlib`, `pandas`, `healpy`, `astropy`, `scikit-learn`
(`camb` for `make_wiener_PR4.py`).
