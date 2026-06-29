"""
S1_voidcluster_tSZ.py
=====================
Driver del análisis tSZ de cúmulos en paredes de void:

  1. Fusiona las N semillas en un catálogo único de voids        (void_seed_merge)
  2. Asocia cúmulos en cáscara 2.5D [f_min,f_max]*Rv + PA al void (cluster_void_assoc)
     con corte de inclinación LOS ("fases de luna": sólo separación transversal)
  3-4. Por submuestra (escaneo de riqueza o split por delta_23): perfiles orientados
       en OCTANTES -> diff_void (mirar al void) + diff_fil (elongación del filamento)
       + suite de nulls (barajado/aleatorio/mapa rotado) con p-value MC (oriented_stacking)

Muestras (build_samples): si 'richness_scan' está seteado, escanea lambda500;
si no, si 'delta23_split' no es None, separa open/closed; si no, 'all'.

ANTES de creerle a cualquier 'diff':
  - oriented_stacking.calibrate_orientation() -> pegar (pa_sign, pa_offset)
  - verificar paridad (~0) y los p-value de los nulls
"""

import os
import numpy as np
import healpy as hp
import pandas as pd

import void_seed_merge as vsm
import cluster_void_assoc as cva
import oriented_stacking as ost


def _file_suffix(config, lam):
    """
    Identificador de la corrida (selección + geometría). Las muestras se guardan
    como tSZ_oriented_{name}_{suffix}.{npz,pdf}; si el .npz existe, no se recalcula.
    Campos: zmin_zmax_rmin_rmax_lambdamin_ztype_maxlosincldeg_delta23split_sizempch.
    """
    def f(x):
        return 'None' if x is None else f'{x:g}'
    return (f"z{f(config['zmin'])}-{f(config['zmax'])}"
            f"_r{f(config['rmin'])}-{f(config['rmax'])}"
            f"_lam{f(lam)}"
            f"_{config.get('z_type') or 'any'}"
            f"_los{f(config.get('max_los_incl_deg'))}"
            f"_d23{f(config.get('delta23_split'))}"
            f"_sz{f(config['size_mpch'])}")


def build_samples(selected, config):
    scan = config.get('richness_scan')
    if scan:
        return {f'lam_ge{thr:g}': selected[selected['lambda500'] >= thr] for thr in scan}
    dsplit = config.get('delta23_split')
    if dsplit is not None and 'delta_23' in selected.columns:
        thr = float(dsplit)
        return {f'open_d23lt{thr:g}':   selected[selected['delta_23'] < thr],
                f'closed_d23ge{thr:g}': selected[selected['delta_23'] >= thr]}
    return {'all': selected}


def _save_suite(out, name, suffix, suite):
    d = {}
    sector_keys = ('full', 'C', 'D', 'along', 'perp', 'up', 'down',
                   'diff_void', 'diff_fil', 'diff_parity')
    for blk in ('signal', 'parity'):
        s = suite[blk]
        d[f'{blk}_r'] = s['r']
        for k in sector_keys:
            d[f'{blk}_{k}'] = s[k]
            d[f'{blk}_err_{k}'] = s[f'err_{k}']
        d[f'{blk}_cov_void'] = s['cov_void']
        d[f'{blk}_cov_fil'] = s['cov_fil']
    for mode, blk in suite['nulls'].items():
        for which in ('void', 'fil'):
            s = blk[which]
            tag = f'null_{mode}_{which}'
            d[f'{tag}_diff_mean'] = s['diff_mean']
            d[f'{tag}_diff_std'] = s['diff_std']
            d[f'{tag}_T_null'] = s['T_null']
            d[f'{tag}_T_signal'] = np.array(s['T_signal'])
            d[f'{tag}_p_value'] = np.array(s['p_value'])
    np.savez(os.path.join(out, f"tSZ_oriented_{name}_{suffix}.npz"), **d)


def _print_summary(name, suite):
    sig, par = suite['signal'], suite['parity']
    print(f"[summary:{name}] N={suite['n']}  pico full={np.nanmax(sig['full']):.2e}")
    for which, lab in (('void', 'mirar void'), ('fil', 'filamento')):
        ps = "  ".join(f"p({mode})={blk[which]['p_value']:.3f}"
                       for mode, blk in suite['nulls'].items())
        T = next(iter(suite['nulls'].values()))[which]['T_signal']
        print(f"           {lab:9s}: T={T:.1f}  {ps}")
    for which in ('void', 'fil'):
        ratio = np.nanmax(np.abs(par[f'diff_{which}']) / par[f'err_diff_{which}'])
        print(f"           paridad {which}: max|diff|/err = {ratio:.1f}  (debe ~<2-3)")


def run(config):
    df = config['data_folder']
    out = config['output_folder']
    os.makedirs(out, exist_ok=True)

    # ---- 1. catálogo único de voids -----------------------------------
    print("\n>>> 1. Fusión de semillas")
    tmpl = f"{df}CATALOGOS/WenHan_voids_z0.6/voids_z0.6_{{:03d}}.dat"

    def base_filter(d):
        return d[(d['z'] >= config['zmin']) & (d['z'] < config['zmax']) &
                 (d['R_void'] >= config['rmin']) & (d['R_void'] <= config['rmax']) &
                 (d['completeness'] > 1.9)]

    seeds = vsm.load_seed_catalogs(tmpl, config['N_seeds'], base_filter=base_filter)
    voids, _ = vsm.merge_seeds(seeds, eps_mpch=config['merge_eps_mpch'],
                               min_frac=config['merge_min_frac'],
                               n_seeds=config['N_seeds'], use_catalog_xyz=False)
    vsm.sanity_report(voids)

    # ---- 2. asociación cúmulo-void ------------------------------------
    print("\n>>> 2. Asociación cúmulo-void")
    col_names = ['ID', 'Name', 'RAdeg', 'DEdeg', 'zCl', 'f_zCl', 'zmag', 'W1mag',
                 'log(M*)', 'r500', 'lambda500', 'M500', 'Ngal', 'Gamma',
                 'e_Gamma', 'Source', 'Cat']
    clu = pd.read_csv(f"{df}CATALOGOS/WenHan_Clusters.dat", sep=r'\s+',
                      skiprows=56, names=col_names, header=None)
    # pre-corte de riqueza: el mínimo del escaneo (para asociar una sola vez)
    scan = config.get('richness_scan')
    lam_pre = min(scan) if scan else config['lambda_min']
    clu = clu[(clu['zCl'] >= config['zmin']) & (clu['zCl'] < config['zmax']) &
              (clu['lambda500'] >= lam_pre)].copy()
    if config.get('z_type') == 'spec':
        clu = clu[clu['f_zCl'] == 1]
    clu = cva.radec_to_galactic(clu).reset_index(drop=True)

    pairs = cva.associate(clu, voids, f_min=config['f_min'], f_max=config['f_max'],
                          clu_z='zCl', resolve=config['resolve'],
                          rpar_max_frac=config.get('rpar_max_frac'),
                          max_los_incl_deg=config.get('max_los_incl_deg'))
    selected = cva.select_clusters(clu, pairs)
    cva.plot_association_mollview(voids, selected,
                                  os.path.join(out, "assoc_mollview.pdf"))

    # ---- muestras + caché por file_suffix -----------------------------
    suffix = _file_suffix(config, lam_pre)
    print(f"[driver] file_suffix = {suffix}")
    samples = build_samples(selected, config)
    print("[driver] muestras: " + ", ".join(f"{k}(N={len(v)})" for k, v in samples.items()))
    pending = {}
    for name, s in samples.items():
        npz_path = os.path.join(out, f"tSZ_oriented_{name}_{suffix}.npz")
        if os.path.exists(npz_path):
            print(f"[driver] '{name}' ya calculado para este suffix; salteo.")
        elif len(s) < config['n_subsamples']:
            print(f"[driver] muestra '{name}' N={len(s)} < n_subsamples; salteo.")
        else:
            pending[name] = s
    if not pending:
        print("[driver] nada pendiente para este file_suffix.\n\n=== LISTO ===")
        return

    # ---- mapa Compton-Y + máscara -------------------------------------
    print("\n>>> Lectura mapa Compton-Y")
    ymap = hp.read_map(f"{df}CMB/Compton_y/Compton-SZMap-NILC-ymap_2048_PR4.fits")
    ymask = hp.read_map(f"{df}CMB/Temperature/Common_mask_Temperature_2048.fits")
    ymask = np.where(ymask >= 0.9, 1.0, 0.0)

    size_mpch = config['size_mpch']
    npix = config['npix_stamp']
    bins = np.linspace(0, size_mpch / 2, config['n_radial_bins'] + 1)
    ps, po = config['pa_sign'], config['pa_offset']
    rng = np.random.default_rng(123)

    print("\n>>> 3-4. Stacking orientado + suite de nulls (MC)")
    for name, s in pending.items():
        suite = ost.run_suite(s['l'].values, s['b'].values, s['zCl'].values,
                              s['PA_void_deg'].values, ymap, ymask, size_mpch,
                              npix, bins, pa_sign=ps, pa_offset=po,
                              n_subsamples=config['n_subsamples'],
                              n_null_real=config['n_null_real'], rng=rng, label=name)
        ost.plot_suite(suite, os.path.join(out, f"tSZ_oriented_{name}_{suffix}.pdf"),
                       size_mpch, label=name)
        _save_suite(out, name, suffix, suite)
        _print_summary(name, suite)

    print("\n=== LISTO ===")


if __name__ == "__main__":
    config = {
        'data_folder': '/home/ftoscano/Doctorado/Data/',
        'output_folder': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/VoidCluster_tSZ/',
        'zmin': 0.1, 'zmax': 0.65, 'rmin': 35.0, 'rmax': 62.7,
        'N_seeds': 100, 'lambda_min': 45, 'z_type': 'spec',
        # dedup
        'merge_eps_mpch': 5.0, 'merge_min_frac': 0.2,
        # asociación
        'f_min': 0.9, 'f_max': 1.6, 'resolve': 'nearest_shell',
        'rpar_max_frac': None,
        'max_los_incl_deg': 30.0,
        # stacking
        'size_mpch': 24.0, 'npix_stamp': 200, 'n_radial_bins': 5, 'n_subsamples': 25,
        # significancia: realizaciones MC del null (>=100 para p~0.01 de resolución)
        'n_null_real': 100,
        # muestras: escaneo de riqueza (precede a delta23). None -> usa delta23/all.
        'richness_scan': [25, 35, 45, 60],
        'delta23_split': None,        # None -> 'all'; float -> open/closed (si scan=None)
        # orientación (de calibrate_orientation())
        'pa_sign': -1.0, 'pa_offset': 90.0,
    }
    run(config)
