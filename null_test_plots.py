"""
null_test_plots.py
==================
Visual check of the two null tests on the selection of Pipeline_voids.py:
  1. rings of max_Rvoid*R_v around the voids (overlap of the stamps);
  2. rotated catalogues (longitude shifts) over the coverage region;
  3. three random realizations as drawn by the pipeline (same footprint and
     coverage criterion as the voids).
    python null_test_plots.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp

import Functions_module as fm
import S1_voids_parallel as s1
from Pipeline_voids import config as PIPELINE_CONFIG

config = dict(PIPELINE_CONFIG)
config.update({'cmb_map': 'PLANCK_PR4', 'void_catalog': 'WH', 'coverage_map': None, 'nside_plot': 512})


def main(cfg=config):
    out_dir = os.path.join(cfg['output_folder'], 'Plots')
    os.makedirs(out_dir, exist_ok=True)
    df_ = cfg['data_folder']
    cat = s1._catalog_spec(cfg['void_catalog'], df_)
    ns_cov = cfg['coverage_nside']
    cov_frac, labels = s1._coverage_region(s1._coverage_keys(cfg, cfg['cmb_map']), df_, ns_cov)
    region = np.where(cov_frac >= 0.5, 1.0, 0.0)
    F = fm.survey_footprint(cat.get('footprint'), df_, ns_cov,
                            os.path.join(cfg['output_folder'], 'lensing', 'Cache_Stacks', 'footprints'),
                            label=cat['label']) if cfg.get('random_pool', 'footprint') == 'footprint' else None
    if F is not None:
        region = region * F
    S = s1.load_void_samples(cfg, cat, cov_frac)
    import pandas as pd
    v = S['merged'] if S['merged'] is not None else (S['single'] if S['single'] is not None
                                                      else pd.concat(S['per_seed'].values()))
    l, b, z, rv = v['l'].values, v['b'].values, v['z'].values, v['R_void'].values
    tag = f"{cat['label']}_{'+'.join(labels)}_N{len(v)}"
    nsp = cfg['nside_plot']

    rings = np.zeros(hp.nside2npix(nsp))
    for i in range(len(v)):
        th = fm.get_angularsize_comoving(z[i], cfg['max_Rvoid'] * rv[i])
        vec = hp.ang2vec(l[i], b[i], lonlat=True)
        out = hp.query_disc(nsp, vec, np.radians(th))
        inn = hp.query_disc(nsp, vec, np.radians(max(0.001, th - 0.15)))
        rings[np.setdiff1d(out, inn)] += 1
    rings[rings == 0] = hp.UNSEEN
    hp.mollview(rings, title=f"r = {cfg['max_Rvoid']} Rv  ({tag})", cmap='turbo', min=1)
    plt.savefig(os.path.join(out_dir, f'Null_1_rings_{tag}.png'), dpi=200, bbox_inches='tight'); plt.close('all')

    covp = hp.ud_grade(np.where(cov_frac >= 0.5, 1.0, 0.0), nsp)
    plt.figure(figsize=(12, 8))
    for i, ang in enumerate([0, 90, 180, 270]):
        hp.mollview(covp, sub=(2, 2, i + 1), title=f'longitude shift {ang} deg', cbar=False, cmap='Blues')
        hp.projscatter((l - ang) % 360, b, lonlat=True, s=0.5, color='red', alpha=0.5)
    plt.savefig(os.path.join(out_dir, f'Null_2_rotations_{tag}.png'), dpi=200, bbox_inches='tight'); plt.close('all')

    regp = hp.ud_grade(region, nsp)
    plt.figure(figsize=(12, 8))
    hp.mollview(regp, sub=(2, 2, 1), title='random-centre region', cbar=False, cmap='Greys')
    hp.projscatter(l, b, lonlat=True, s=0.5, color='red', alpha=0.6)
    for i in range(3):
        rng = np.random.default_rng(1000 + i)
        perm = rng.permutation(len(z))
        rl, rb, _, _, nsw = fm.matched_randoms(z[perm], rv[perm], region, cov_frac, cfg['max_Rvoid'],
                                               cfg.get('min_footprint_coverage', 0.0), rng)
        hp.mollview(regp, sub=(2, 2, i + 2), title=f'random realization {i + 1} (swaps={nsw})',
                    cbar=False, cmap='Greys')
        hp.projscatter(rl, rb, lonlat=True, s=0.5, color='blue', alpha=0.6)
    plt.savefig(os.path.join(out_dir, f'Null_3_randoms_{tag}.png'), dpi=200, bbox_inches='tight'); plt.close('all')
    print(f'[null plots] saved in {out_dir}')


if __name__ == '__main__':
    main()
