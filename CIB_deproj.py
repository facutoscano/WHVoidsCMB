"""
check_CIB_deproj.py
===================
¿Por qué el stack de voids con la reconstrucción CIB-deproyectada da ~la mitad que
PR3/PR4 (en WH y en BOSS)? Tres hipótesis, que este script separa:

  (1) RUIDO: el mapa CIB-deproj es más ruidoso y la caída es una fluctuación.
  (2) NORMALIZACIÓN: el klm CIB-deproj tiene otra normalización (señal global ~0.5x).
  (3) CONTAMINACIÓN: los mapas estándar llevan un sesgo de CIB alrededor de los voids
      que el CIB-deproj no tiene.

TEST 0  ¿Los mapas de datos llevan el corte de L de sus espectros?
        Fracción de potencia en L fuera de [8,400] (Planck) y [40,763] (ACT). Si es
        apreciable, el corte NO está en el mapa (sólo en el likelihood del espectro).

TEST A  Normalización global con un trazador independiente: densidad de cúmulos WH (g).
        Para cada mapa X:  A_X - 1 = C_L(d x g) / C_L(ref x g),  con d = kappa_X - kappa_ref.
        Usar la DIFERENCIA cancela el ruido común a ambas reconstrucciones (mismos datos CMB);
        el error de Knox usa C_L(d x d), o sea el ruido propio de la diferencia.
        A_X ~ 1  ->  descarta (2).

TEST B  Específico de voids: mapa diferencia FILTRADO (cada mapa con su propio Wiener,
        como en S1),  D = W_X*kappa_X - W_ref*kappa_ref,  apilado con S1 en los MISMOS
        voids que la corrida de datos. Randoms sobre D = ruido exacto de la diferencia.
        chi2 de D contra 0 (cov de randoms):
          compatible con 0           -> (1) fluctuación
          significativo y A_X ~ 1    -> (3) contaminación alrededor de voids

Uso (desde la carpeta de la pipeline, toma la selección de voids de Pipeline_voids.config):
    python check_CIB_deproj.py
"""

import os
import sys
import numpy as np
import pandas as pd
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u

import Functions_module as fm
import S1_voids_parallel as s1
import Pipeline_voids as PV          # sólo para copiar su config (no corre nada al importar)

config = {
    'out_folder': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/checks/CIB/',
    'ref':        'PLANCK_PR3',                       # referencia (mismo pipeline Planck 2018)
    'maps_A':     ['PLANCK_CIB', 'PLANCK_SZ', 'PLANCK_SZ_DEPROJ', 'PLANCK_INHOM', 'PLANCK_PR4'],
    'map_B':      'PLANCK_CIB',                       # el que se diferencia y apila en el test B

    # ---- test 0 ----
    'run_test0': True,
    'act_map':   'CMB/ACT/kappa_act_dr6_baseline_ns512_WF.fits',   # relativo a data_folder

    # ---- test A ----
    'run_testA':  True,
    'clusters':   'CATALOGOS/WenHan_Clusters.dat',    # relativo a data_folder
    'clu_zmin': 0.1, 'clu_zmax': 0.6,
    'nside_A':    512,
    'lmax_A':     600,
    'L_bins_A':   [8, 40, 100, 200, 400],

    # ---- test B ----
    'run_testB':  True,
    'nside_B':    2048,
    # selección de voids: la de Pipeline_voids.config (se pisan sólo estas claves)
    'pipeline_overrides': {'void_catalog': 'WH', 'random_pool': 'full'},
}


# ----------------------------------------------------------------------
def _read_klm(path):
    alm = hp.fitsfunc.read_alm(path, hdu=1, return_mmax=False)
    return alm, hp.Alm.getlmax(len(alm))


def _trunc(alm, lmax_in, lmax_out):
    return hp.resize_alm(alm, lmax_in, lmax_in, lmax_out, lmax_out) if lmax_in > lmax_out else alm


def _mask(path, nside):
    m = hp.read_map(path, dtype=np.float64)
    if hp.get_nside(m) != nside:
        m = hp.ud_grade(m, nside_out=nside)
    return (m >= 0.9).astype(float)


def _frac_power(cl, lo, hi):
    L = np.arange(len(cl))
    w = (2 * L + 1) * cl
    tot = np.sum(w[2:])
    return np.sum(w[(L >= 2) & (L < lo)]) / tot, np.sum(w[L > hi]) / tot


# ----------------------------------------------------------------------
def test0(cfg, df):
    print("\n=== TEST 0: ¿el corte de L de los espectros está en los mapas? ===")
    for key, (lo, hi) in (('PLANCK_PR4', (8, 400)), ('PLANCK_PR3', (8, 400))):
        sp = s1._map_spec(key, df)
        alm, lmax = _read_klm(sp['klm'])
        cl = hp.alm2cl(alm)
        flo, fhi = _frac_power(cl, lo, hi)
        print(f"[0] {key:11s} klm (lmax={lmax}, SIN Wiener): potencia (2L+1)C_L en L<{lo}: {flo:.3f} | "
              f"L>{hi}: {fhi:.3f}")
        W = fm.apply_wiener_filter(alm.copy(), sp['nlkk'], lmax=sp['nside'])[1]
        clw = cl[:len(W)] * W ** 2
        flo, fhi = _frac_power(clw, lo, hi)
        print(f"[0] {key:11s} klm CON Wiener (como en S1):                 L<{lo}: {flo:.3f} | L>{hi}: {fhi:.3f}")
    act = os.path.join(df, cfg['act_map'])
    if os.path.exists(act):
        m = hp.read_map(act, dtype=np.float64)
        msk = _mask(s1._map_spec('ACT', df)['mask'], hp.get_nside(m))
        cl = hp.anafast(m * msk, lmax=3 * hp.get_nside(m) - 1) / np.mean(msk ** 2)
        flo, fhi = _frac_power(cl, 40, 763)
        print(f"[0] ACT WF map (nside {hp.get_nside(m)}):                         L<40: {flo:.3f} | L>763: {fhi:.3f}")
    print("[0] Fracciones apreciables => el corte NO está en el mapa: aplicarlo sólo a Websky lo haría "
          "MENOS parecido a los datos. Para cortar, cortar en los dos.")


# ----------------------------------------------------------------------
def _cluster_map(cfg, df, nside):
    d = pd.read_csv(os.path.join(df, cfg['clusters']), sep=r'\s+', skiprows=56, header=None,
                    usecols=[2, 3, 4])
    d = d[(d[4] >= cfg['clu_zmin']) & (d[4] < cfg['clu_zmax'])]
    c = SkyCoord(ra=d[2].values * u.degree, dec=d[3].values * u.degree, frame='icrs').galactic
    l, b = c.l.degree, c.b.degree
    n = np.zeros(hp.nside2npix(nside))
    np.add.at(n, hp.ang2pix(nside, l, b, lonlat=True), 1.0)
    foot = fm.footprint_mask(l, b, output_nside=nside, footprint_nside=64)
    print(f"[A] {len(l)} cúmulos WH en z=[{cfg['clu_zmin']},{cfg['clu_zmax']}); footprint fsky={foot.mean():.3f}")
    return n, foot


def testA(cfg, df):
    print("\n=== TEST A: normalización global vía cruce con cúmulos WH ===")
    ns, lmax = cfg['nside_A'], cfg['lmax_A']
    keys = [cfg['ref']] + [k for k in cfg['maps_A'] if k != cfg['ref']]
    specs = {k: s1._map_spec(k, df) for k in keys}
    n, foot = _cluster_map(cfg, df, ns)
    mask = foot.copy()
    for k in keys:
        mask *= _mask(specs[k]['mask'], ns)
    fsky, w2 = mask.mean(), np.mean(mask ** 2)
    nbar = n[mask > 0].mean()
    g = np.where(mask > 0, n / nbar - 1.0, 0.0)
    print(f"[A] máscara común (todos los mapas x footprint WH): fsky={fsky:.3f}")

    kmaps = {}
    for k in keys:
        alm, l0 = _read_klm(specs[k]['klm'])
        kmaps[k] = hp.alm2map(_trunc(alm, l0, lmax), ns, lmax=lmax)
    L = np.arange(lmax + 1)
    cgg = hp.anafast(g * mask, lmax=lmax) / w2
    ref = kmaps[cfg['ref']]
    c_rg = hp.anafast(ref * mask, g * mask, lmax=lmax) / w2
    edges = cfg['L_bins_A']

    rows = []
    for k in keys[1:]:
        dmap = kmaps[k] - ref
        c_dg = hp.anafast(dmap * mask, g * mask, lmax=lmax) / w2
        c_dd = hp.anafast(dmap * mask, lmax=lmax) / w2
        line = f"[A] {k:17s}"
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (L >= lo) & (L < hi)
            dg, rg = np.mean(c_dg[s]), np.mean(c_rg[s])
            var_L = (c_dd[s] * cgg[s] + c_dg[s] ** 2) / ((2 * L[s] + 1) * fsky)   # Knox por L
            sig = np.sqrt(np.sum(var_L)) / np.sum(s)                             # error de la media del bin
            A, eA = 1 + dg / rg, sig / abs(rg)
            line += f" | L[{lo},{hi}): A={A:5.2f}±{eA:4.2f}"
            rows.append((k, lo, hi, A, eA))
        print(line)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, k in enumerate(keys[1:]):
        r = [x for x in rows if x[0] == k]
        Lc = np.array([0.5 * (x[1] + x[2]) for x in r]) * (1 + 0.03 * (i - 2))
        ax.errorbar(Lc, [x[3] for x in r], yerr=[x[4] for x in r], fmt='o-', capsize=3, label=k)
    ax.axhline(1, color='k', ls='--'); ax.axhline(0.5, color='gray', ls=':')
    ax.set_xscale('log'); ax.set_xlabel('L'); ax.set_ylabel(rf"$A_X = C_L^{{\kappa_X g}}/C_L^{{\kappa_{{ref}} g}}$")
    ax.set_title(f"Normalización relativa a {cfg['ref']} (cruce con cúmulos WH)"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = os.path.join(cfg['out_folder'], 'testA_amplitud_vs_cumulos.pdf')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"[A] figura: {out}")
    print("[A] A~1 en 40<L<400 => normalización OK (descarta hipótesis 2). A~0.5 => normalización.")


# ----------------------------------------------------------------------
def _filtered_map(spec, nside):
    alm, l0 = _read_klm(spec['klm'])
    if spec.get('apply_wiener') and spec.get('nlkk'):
        alm, _ = fm.apply_wiener_filter(alm, spec['nlkk'], lmax=nside)
    return hp.alm2map(alm, nside=nside)


def testB(cfg, df):
    print("\n=== TEST B: stack del mapa diferencia filtrado en los voids ===")
    ns = cfg['nside_B']
    X, R = cfg['map_B'], cfg['ref']
    sx, sr = s1._map_spec(X, df), s1._map_spec(R, df)
    if sx['nlkk'] != sr['nlkk']:
        print(f"[B] nota: {X} y {R} usan distinto nlkk -> se filtra cada uno con el suyo antes de restar")
    label = f"DIFF_{sx['label'].replace('PLANCK_', '')}_minus_{sr['label'].replace('PLANCK_', '')}"
    mdir = os.path.join(cfg['out_folder'], 'maps'); os.makedirs(mdir, exist_ok=True)
    dpath, mpath = os.path.join(mdir, f"{label}_ns{ns}.fits"), os.path.join(mdir, f"{label}_mask_ns{ns}.fits")
    if not os.path.exists(dpath):
        D = _filtered_map(sx, ns) - _filtered_map(sr, ns)
        hp.write_map(dpath, D.astype(np.float32), overwrite=True, dtype=np.float32)
        M = _mask(sx['mask'], ns) * _mask(sr['mask'], ns)
        hp.write_map(mpath, M.astype(np.float32), overwrite=True, dtype=np.float32)
        print(f"[B] mapa diferencia -> {dpath}")

    orig = s1._map_spec
    spec = {'label': label, 'kind': 'map', 'frame': 'galactic', 'nside': ns, 'full_sky': True,
            'apply_wiener': False, 'klm': dpath, 'mask': mpath, 'nlkk': None}
    s1._map_spec = lambda k, d, a=0.0: dict(spec) if str(k).upper() == label.upper() else orig(k, d, a)
    c = dict(PV.config)
    c.update(cfg['pipeline_overrides'])
    c.update(cmb_map=label, coverage_map=None, output_folder=os.path.join(cfg['out_folder'], ''))
    res = s1._run_single_map(c)
    s1._map_spec = orig
    if res is None:
        print("[B] (!) S1 no devolvió resultado"); return

    import Websky as W                     # reutiliza chi2 con cov de randoms y su figura
    ch = W.chi2_test({'res': res})
    print(f"[B] {label} contra 0: {ch['msg']}")
    W.plot_chi2(ch, os.path.join(cfg['out_folder'], f"testB_chi2_{label}.pdf"),
                f"{label} apilado en voids, contra 0 | cov de randoms")
    r = np.asarray(res['r_frac'])
    net, sig = W._net(res)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(r, -sig * 1e3, sig * 1e3, color='gray', alpha=0.3, label=r'$1\sigma$ randoms')
    ax.errorbar(r, net * 1e3, yerr=sig * 1e3, fmt='o-', color='xkcd:crimson', capsize=3,
                label=f"{X} − {R}  (N={res.get('n_voids', '?')})")
    ax.axhline(0, color='k', ls=':'); ax.axvline(1, color='gray', ls='--')
    ax.set_xlabel(r'$r/R_v$'); ax.set_ylabel(r'$\Delta\kappa_{\rm net}\;[10^{-3}]$'); ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    out = os.path.join(cfg['out_folder'], f"testB_profile_{label}.pdf")
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"[B] figura: {out}")
    print("[B] PTE alto => la caída del CIB-deproj es ruido (hipótesis 1). PTE bajo => diferencia real: "
          "mirar test A para distinguir normalización (2) de contaminación (3).")


def main(cfg=config):
    os.makedirs(cfg['out_folder'], exist_ok=True)
    df = PV.config['data_folder']
    if cfg['run_test0']:
        test0(cfg, df)
    if cfg['run_testA']:
        testA(cfg, df)
    if cfg['run_testB']:
        testB(cfg, df)


if __name__ == '__main__':
    main()
