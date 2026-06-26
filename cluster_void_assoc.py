"""
cluster_void_assoc.py
=====================
Asociación de cúmulos a voids: selecciona cúmulos en una cáscara
[f_min, f_max] * R_void alrededor de cada void, en geometría "2.5D"
(separación transversal comóvil r_perp + separación LOS comóvil r_par), resuelve
multiplicidad (un cúmulo -> un void) y calcula el ángulo de posición (PA) del
void respecto del cúmulo para la orientación posterior.

Convenciones
------------
- Todo en comóvil Mpc/h (consistente con R_void de Sparkling).
- r_par  = chi(z_cl) - chi(z_void)            [LOS, con signo, aprox. ángulo chico]
- s (3D) = |x_cl - x_void| en comóvil
- r_perp = sqrt(s^2 - r_par^2)
- PA: bearing esférico cúmulo->void (E de N). SU SIGNO/OFFSET FRENTE AL PROYECTOR
  GNOMÓNICO DEBE CALIBRARSE (ver oriented_stacking.calibrate_orientation).
  Acá sólo lo calculamos geométricamente; la calibración vive en el stacking.

Nota física
-----------
Como los voids se identifican A PARTIR de estos cúmulos, esta selección es
parcialmente circular (pared del void = donde se acumulan cúmulos). No sesga la
medición de presión, pero condiciona la interpretación y exige un control físico
aparte del null instrumental.
"""

import numpy as np
import pandas as pd
import healpy as hp
from scipy.spatial import cKDTree
from astropy.cosmology import Planck18
from astropy.coordinates import SkyCoord
from astropy import units as u


def radec_to_galactic(df, ra_col='RAdeg', dec_col='DEdeg',
                      out_l='l', out_b='b'):
    """Agrega columnas galácticas l,b al catálogo de cúmulos (icrs->galactic)."""
    c = SkyCoord(ra=df[ra_col].values * u.deg, dec=df[dec_col].values * u.deg,
                 frame='icrs').galactic
    df = df.copy()
    df[out_l] = c.l.degree
    df[out_b] = c.b.degree
    return df


def _comoving_cart(l_deg, b_deg, z, h=None):
    if h is None:
        h = Planck18.h
    chi = Planck18.comoving_distance(z).value * h
    lon, lat = np.radians(l_deg), np.radians(b_deg)
    X = np.column_stack([chi * np.cos(lat) * np.cos(lon),
                         chi * np.cos(lat) * np.sin(lon),
                         chi * np.sin(lat)])
    return X, chi


def bearing_deg(l_c, b_c, l_v, b_v):
    """
    PA del void visto desde el cúmulo, medido E de N (grados). Geométrico:
    el signo/offset efectivo frente al stamp se fija en la calibración.
    """
    lc, bc = np.radians(l_c), np.radians(b_c)
    lv, bv = np.radians(l_v), np.radians(b_v)
    dl = lv - lc
    y = np.sin(dl) * np.cos(bv)
    x = np.cos(bc) * np.sin(bv) - np.sin(bc) * np.cos(bv) * np.cos(dl)
    return np.degrees(np.arctan2(y, x))


def associate(clusters, voids, f_min=0.9, f_max=1.6, h=None,
              clu_l='l', clu_b='b', clu_z='z',
              rperp_max_frac=None, rpar_max_frac=None,
              max_los_incl_deg=None, resolve='nearest_shell'):
    """
    Parameters
    ----------
    clusters : DataFrame con columnas galácticas (clu_l, clu_b) y redshift (clu_z).
    voids    : DataFrame con columnas l, b, z, R_void (y opcional void_id, delta_23).
    f_min, f_max   : límites de la cáscara en unidades de R_void (selección sobre s/Rv).
    rperp_max_frac : (opcional) corte adicional |r_perp| <= rperp_max_frac*Rv.
    rpar_max_frac  : (opcional) corte adicional |r_par|  <= rpar_max_frac*Rv.
    max_los_incl_deg : (opcional) "fases de luna". incl = arcsin(|r_par|/s) es el
                     ángulo del vector void->cúmulo sobre el plano del cielo.
                     Se conservan sólo los cúmulos con incl <= max_los_incl_deg
                     (separación transversal -> PA bien definido). 30 deg <-> |r_par|/s<=0.5.
    resolve  : 'nearest_shell' (|s/Rv-1| mínimo) o 'nearest' (s mínimo).

    Returns
    -------
    pairs : DataFrame, un cúmulo por fila (tras resolver multiplicidad), con
            cluster_idx, void_id, s_over_Rv, r_perp, r_par, los_incl_deg, R_void,
            void_(l,b,z), PA_void_deg, [delta_23 si está en voids].
    """
    if h is None:
        h = Planck18.h

    Xc, chic = _comoving_cart(clusters[clu_l].values, clusters[clu_b].values,
                              clusters[clu_z].values, h)
    Xv, chiv = _comoving_cart(voids['l'].values, voids['b'].values,
                              voids['z'].values, h)
    Rv = voids['R_void'].values
    vid = voids['void_id'].values if 'void_id' in voids.columns else np.arange(len(voids))
    has_d23 = 'delta_23' in voids.columns
    d23 = voids['delta_23'].values if has_d23 else None

    tree = cKDTree(Xc)
    rows = []
    for j in range(len(voids)):
        idx = tree.query_ball_point(Xv[j], r=f_max * Rv[j])
        if not idx:
            continue
        idx = np.asarray(idx)
        d3 = np.linalg.norm(Xc[idx] - Xv[j], axis=1)
        r_par = chic[idx] - chiv[j]
        r_perp = np.sqrt(np.clip(d3 ** 2 - r_par ** 2, 0, None))
        srv = d3 / Rv[j]
        incl = np.degrees(np.arcsin(np.clip(np.abs(r_par) / np.maximum(d3, 1e-9), 0, 1)))

        sel = (srv >= f_min) & (srv <= f_max)
        if rperp_max_frac is not None:
            sel &= (r_perp <= rperp_max_frac * Rv[j])
        if rpar_max_frac is not None:
            sel &= (np.abs(r_par) <= rpar_max_frac * Rv[j])
        if max_los_incl_deg is not None:
            sel &= (incl <= max_los_incl_deg)
        if not np.any(sel):
            continue

        ci = idx[sel]
        pa = bearing_deg(clusters[clu_l].values[ci], clusters[clu_b].values[ci],
                         voids['l'].values[j], voids['b'].values[j])
        for k in range(len(ci)):
            row = {
                'cluster_idx': int(ci[k]),
                'void_id': int(vid[j]),
                's_over_Rv': float(srv[sel][k]),
                'r_perp': float(r_perp[sel][k]),
                'r_par': float(r_par[sel][k]),
                'los_incl_deg': float(incl[sel][k]),
                'R_void': float(Rv[j]),
                'void_l': float(voids['l'].values[j]),
                'void_b': float(voids['b'].values[j]),
                'void_z': float(voids['z'].values[j]),
                'PA_void_deg': float(pa[k]),
            }
            if has_d23:
                row['delta_23'] = float(d23[j])
            rows.append(row)

    pairs = pd.DataFrame(rows)
    n_raw = len(pairs)
    incl_txt = f", incl<={max_los_incl_deg}deg" if max_los_incl_deg is not None else ""
    print(f"[assoc] {n_raw} pares (cúmulo,void) en cáscara [{f_min},{f_max}]·Rv{incl_txt}.")
    if n_raw == 0:
        return pairs

    n_multi = int((pairs['cluster_idx'].value_counts() > 1).sum())
    if resolve == 'nearest_shell':
        pairs['rank'] = np.abs(pairs['s_over_Rv'] - 1.0)
    elif resolve == 'nearest':
        pairs['rank'] = pairs['s_over_Rv']
    else:
        raise ValueError(f"resolve desconocido: {resolve}")
    pairs = (pairs.sort_values('rank')
                  .drop_duplicates('cluster_idx', keep='first')
                  .drop(columns='rank')
                  .reset_index(drop=True))
    print(f"[assoc] {n_multi} cúmulos asociados a >1 void; "
          f"tras resolver ('{resolve}'): {len(pairs)} cúmulos únicos.")
    return pairs


def select_clusters(clusters, pairs):
    """Devuelve el sub-DataFrame de cúmulos seleccionados, con PA y void pegados."""
    sub = clusters.iloc[pairs['cluster_idx'].values].copy().reset_index(drop=True)
    carry = ['void_id', 's_over_Rv', 'r_perp', 'r_par', 'los_incl_deg', 'R_void',
             'void_l', 'void_b', 'void_z', 'PA_void_deg']
    if 'delta_23' in pairs.columns:
        carry.append('delta_23')
    for c in carry:
        sub[c] = pairs[c].values
    return sub


def plot_association_mollview(voids, selected, outpath=None,
                              clu_l='l', clu_b='b', title="Voids y cúmulos asociados"):
    """
    Inspección visual: voids (rojo, tamaño ~R_void) y cúmulos seleccionados (azul).
    No dibuja líneas de unión (engorroso en mollview); para auditar pares puntuales
    usá un void_id concreto.
    """
    import matplotlib.pyplot as plt
    npix = hp.nside2npix(64)
    hp.mollview(np.zeros(npix), title=title, cbar=False, cmap='Greys',
                min=-1, max=1)
    hp.projscatter(voids['l'].values, voids['b'].values, lonlat=True,
                   s=np.clip(voids['R_void'].values / 5, 2, 40),
                   facecolors='none', edgecolors='red', linewidths=0.6,
                   alpha=0.7, label='voids')
    hp.projscatter(selected[clu_l].values, selected[clu_b].values, lonlat=True,
                   s=2, color='blue', alpha=0.5, label='cúmulos asoc.')
    hp.graticule()
    plt.legend(loc='lower right', fontsize=8)
    if outpath:
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
        print(f"[assoc] mollview guardado en {outpath}")
        plt.close()
