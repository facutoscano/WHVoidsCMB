"""
oriented_stacking.py
====================
Stacking gnomónico ORIENTADO según la dirección al void, perfiles radiales
partidos en OCTANTES (un único binneado de 45 deg) y errores jackknife sobre
cada grupo y sobre las diferencias. Incluye calibración de la convención de
orientación y suite de nulls (PA barajado/aleatorio + mapa rotado).

Escala
------
Comóvil Mpc/h, consistente con R_void. `size_mpch` = ancho TOTAL del stamp en
Mpc/h. (El ángulo proyectado es el mismo que con Mpc propios; lo que cambia es la
etiqueta del eje, y la elegimos comóvil Mpc/h para comparar con R_void.)

Convención de octantes
----------------------
Tras calibrar, el void queda a la IZQUIERDA del stamp. Con los 8 octantes:
  - diff_void = (cara al void: NW+W+SW) - (opuesta: NE+E+SE)   [mirar al void]
  - diff_fil  = (a lo largo del filamento: N+S) - (perp/eje void: W+E)
  - diff_parity = (arriba: NE+N+NW) - (abajo: SW+S+SE)   [debe ~0; sistemáticos]
Ver cabecera de radial_profile_regions para los bordes exactos (22.5+45k deg).

ADVERTENCIA
-----------
El signo/offset que alinea el void a la izquierda DEPENDE de la convención del
proyector gnomónico de healpy y del frame de (l,b). NO confiar en el cálculo
analítico: correr calibrate_orientation() UNA vez y usar el (pa_sign, pa_offset)
que devuelve. Además correr SIEMPRE el null de orientación aleatoria.
"""

import numpy as np
import healpy as hp
from astropy.cosmology import Planck18
from sklearn.cluster import KMeans


# ----------------------------------------------------------------------
# geometría
# ----------------------------------------------------------------------
def angsize_comoving_deg(z, size_mpch, h=None):
    """Tamaño angular (deg) de `size_mpch` (Mpc/h comóvil) a redshift z."""
    if h is None:
        h = Planck18.h
    d_c = Planck18.comoving_distance(z).value * h
    return np.degrees(size_mpch / d_c)


# ----------------------------------------------------------------------
# stacking orientado (devuelve SUM y COUNT para poder hacer JK por sustracción)
# ----------------------------------------------------------------------
def stack_oriented(lon, lat, z, pa_deg, cmb_map, mask, size_mpch, npix,
                   pa_offset=0.0, pa_sign=1.0, randomize_pa=False, rng=None):
    """
    Devuelve (SUM, COUNT), arrays npix x npix.
    El stamp se rota en su plano (3er ángulo de Euler del proyector) para fijar la
    dirección al void:  psi = pa_sign * pa_deg + pa_offset  [deg].
    randomize_pa=True -> psi aleatorio uniforme (null de orientación).
    """
    lon = np.atleast_1d(lon); lat = np.atleast_1d(lat)
    z = np.atleast_1d(z); pa_deg = np.atleast_1d(pa_deg)
    if rng is None:
        rng = np.random.default_rng(0)
    nside = hp.npix2nside(len(cmb_map))
    v2p = lambda x, y, zz: hp.vec2pix(nside, x, y, zz)
    SUM = np.zeros((npix, npix))
    COUNT = np.zeros((npix, npix))
    for i in range(len(lon)):
        box_deg = angsize_comoving_deg(z[i], size_mpch)
        reso = box_deg * 60.0 / npix
        psi = rng.uniform(0, 360) if randomize_pa else (pa_sign * pa_deg[i] + pa_offset)
        proj = hp.projector.GnomonicProj(rot=[lon[i], lat[i], psi],
                                         xsize=npix, ysize=npix, reso=reso)
        stamp = proj.projmap(cmb_map, vec2pix_func=v2p)
        smask = proj.projmap(mask, vec2pix_func=v2p)
        good = (smask > 0.9) & (~np.isnan(stamp))
        SUM[good] += stamp[good]
        COUNT[good] += 1
    return SUM, COUNT


# ----------------------------------------------------------------------
# perfiles radiales por OCTANTES (un único binneado de 45 deg)
# ----------------------------------------------------------------------
# Convención (void calibrado a la IZQUIERDA del stamp):
#   phi = atan2(yy, xx) en deg, 0..360. Derecha(+x)=0, arriba(+y)=90,
#   izquierda(-x, void)=180, abajo(-y)=270.
#
#   Octantes (centro, ancho 45): E=0, NE=45, N=90, NW=135, W=180(void),
#   SW=225, S=270, SE=315. Bordes en 22.5 + 45*k.
#
# Grupos derivados de los 8 octantes:
#   C  (cara al void)  = NW+W+SW   (izquierda, phi in [112.5,247.5))
#   D  (opuesta)       = NE+E+SE   (derecha,   phi in [-67.5, 67.5))
#   along (filamento)  = N+S       (vertical,  tangencial a la pared)
#   perp  (eje al void)= W+E       (horizontal, perpendicular al filamento)
#   up / down          = chequeo de simetría arriba/abajo (null ~0)
#
# Medidas:
#   diff_void   = C - D       (mirar al void vs opuesta)
#   diff_fil    = along - perp (a lo largo del filamento vs perpendicular)
#   diff_parity = up - down    (debe ~0; sistemáticos de orientación)
# ----------------------------------------------------------------------
def _octant_geom(npix, size_mpch):
    c = npix // 2
    yy, xx = np.ogrid[-c:npix - c, -c:npix - c]
    r = np.sqrt(xx * xx + yy * yy) * (size_mpch / npix)     # Mpc/h comóvil
    phi = np.degrees(np.arctan2(yy * np.ones_like(xx),
                                xx * np.ones_like(yy))) % 360.0
    return r, phi


def _region_masks(phi):
    """Máscaras booleanas de los grupos de octantes (ver cabecera)."""
    C = (phi >= 112.5) & (phi < 247.5)                       # NW+W+SW (cara)
    D = (phi < 67.5) | (phi >= 292.5)                        # NE+E+SE (opuesta)
    along = (((phi >= 67.5) & (phi < 112.5)) |               # N
             ((phi >= 247.5) & (phi < 292.5)))               # S
    perp = ((phi < 22.5) | (phi >= 337.5) |                  # E
            ((phi >= 157.5) & (phi < 202.5)))                # W
    up = (phi >= 22.5) & (phi < 157.5)                       # NE+N+NW
    down = (phi >= 202.5) & (phi < 337.5)                    # SW+S+SE
    return {'C': C, 'D': D, 'along': along, 'perp': perp, 'up': up, 'down': down}


# claves de perfil que produce radial_profile_regions (orden estable)
REGION_KEYS = ('full', 'C', 'D', 'along', 'perp', 'up', 'down',
               'diff_void', 'diff_fil', 'diff_parity')


def radial_profile_regions(SUM, COUNT, size_mpch, bins):
    """
    Perfil pesado por COUNT en cada anillo, para cada grupo de octantes.
    Returns: dict con 'r' + las claves de REGION_KEYS (cada una array nbins).
    """
    npix = SUM.shape[0]
    r, phi = _octant_geom(npix, size_mpch)
    regions = _region_masks(phi)

    def _prof(extra_mask):
        out = []
        for i in range(len(bins) - 1):
            ring = (r >= bins[i]) & (r < bins[i + 1]) & extra_mask
            den = np.nansum(COUNT[ring])
            out.append(np.nansum(SUM[ring]) / den if den > 0 else np.nan)
        return np.array(out)

    P = {name: _prof(m) for name, m in regions.items()}
    P['full'] = _prof(np.ones_like(r, dtype=bool))
    P['diff_void'] = P['C'] - P['D']
    P['diff_fil'] = P['along'] - P['perp']
    P['diff_parity'] = P['up'] - P['down']
    P['r'] = 0.5 * (bins[:-1] + bins[1:])
    return P


# ----------------------------------------------------------------------
# errores jackknife (KMeans en la esfera) para todos los grupos + diffs
# ----------------------------------------------------------------------
def sectors_with_errors(lon, lat, z, pa_deg, cmb_map, mask, size_mpch, npix, bins,
                        n_subsamples=20, pa_offset=0.0, pa_sign=1.0,
                        randomize_pa=False, rng=None):
    lon = np.asarray(lon); lat = np.asarray(lat)
    z = np.asarray(z); pa_deg = np.asarray(pa_deg)
    n = len(lon)

    lonr, latr = np.radians(lon), np.radians(lat)
    XYZ = np.column_stack([np.cos(latr) * np.cos(lonr),
                           np.cos(latr) * np.sin(lonr),
                           np.sin(latr)])
    labels = KMeans(n_clusters=n_subsamples, random_state=42,
                    n_init=10).fit_predict(XYZ)

    sums, counts = [], []
    for k in range(n_subsamples):
        m = labels == k
        if m.sum() == 0:
            sums.append(np.zeros((npix, npix)))
            counts.append(np.zeros((npix, npix)))
            continue
        s, c = stack_oriented(lon[m], lat[m], z[m], pa_deg[m], cmb_map, mask,
                              size_mpch, npix, pa_offset, pa_sign,
                              randomize_pa, rng)
        sums.append(s); counts.append(c)
    sums, counts = np.array(sums), np.array(counts)
    SUM, COUNT = sums.sum(0), counts.sum(0)

    P = radial_profile_regions(SUM, COUNT, size_mpch, bins)

    jk = {key: [] for key in REGION_KEYS}
    for k in range(n_subsamples):
        Pk = radial_profile_regions(SUM - sums[k], COUNT - counts[k],
                                    size_mpch, bins)
        for key in REGION_KEYS:
            jk[key].append(Pk[key])

    def _cov(jklist, best):
        a = np.array(jklist)
        delta = np.nan_to_num(a - best)
        return (n_subsamples - 1) / n_subsamples * (delta.T @ delta)

    res = {'r': P['r'], 'n': n,
           'map': np.divide(SUM, COUNT, out=np.full_like(SUM, np.nan),
                            where=COUNT > 0)}
    for key in REGION_KEYS:
        res[key] = P[key]
        res['err_' + key] = np.sqrt(np.diag(_cov(jk[key], P[key])))
    res['cov_void'] = _cov(jk['diff_void'], P['diff_void'])
    res['cov_fil'] = _cov(jk['diff_fil'], P['diff_fil'])
    return res


# ----------------------------------------------------------------------
# CALIBRACIÓN de la convención de orientación  (CORRER UNA VEZ)
# ----------------------------------------------------------------------
def _bearing(l_c, b_c, l_v, b_v):
    lc, bc = np.radians(l_c), np.radians(b_c)
    lv, bv = np.radians(l_v), np.radians(b_v)
    dl = lv - lc
    y = np.sin(dl) * np.cos(bv)
    x = np.cos(bc) * np.sin(bv) - np.sin(bc) * np.cos(bv) * np.cos(dl)
    return np.degrees(np.arctan2(y, x))


def _offset_point(l_c, b_c, sep_deg, beta_deg):
    """Punto a distancia angular sep_deg y bearing beta_deg (E de N) desde (l_c,b_c)."""
    b1, l1 = np.radians(b_c), np.radians(l_c)
    d, beta = np.radians(sep_deg), np.radians(beta_deg)
    b2 = np.arcsin(np.sin(b1) * np.cos(d) + np.cos(b1) * np.sin(d) * np.cos(beta))
    l2 = l1 + np.arctan2(np.sin(beta) * np.sin(d) * np.cos(b1),
                         np.cos(d) - np.sin(b1) * np.sin(b2))
    return np.degrees(l2) % 360.0, np.degrees(b2)


def calibrate_orientation(nside=512, npix=200, z_test=0.3, box_deg=6.0,
                          sep_deg=1.0, offset_step=5.0, h=None, verbose=True):
    """
    Calibra (pa_sign, pa_offset) tal que la dirección al void apunte EXACTAMENTE
    a la izquierda (9 en punto, -x), no sólo "a la mitad izquierda".

    Por qué el ángulo exacto importa: el split D/C parte el stamp por la vertical,
    así que (L-R) mide la PROYECCIÓN del dipolo sobre el eje horizontal. Si el
    void apunta, p.ej., a 10:30 (60deg de error), la señal se diluye por cos(60)=0.5.

    Robustez del signo: se prueban 8 bearings; con el signo equivocado la
    orientación se scramblea (psi ~ 2·PA) y el error angular explota. El criterio
    minimiza el PEOR |ángulo_centroide - 180deg| sobre los 8 bearings.
    """
    if h is None:
        h = Planck18.h
    l_c, b_c = 40.0, 25.0                       # lejos de polos y del wrap en l=0
    betas = np.arange(0.0, 360.0, 45.0)
    lv, bv = _offset_point(l_c, b_c, sep_deg, betas)
    pa = _bearing(np.full(betas.shape, l_c), np.full(betas.shape, b_c), lv, bv)

    size_mpch = np.radians(box_deg) * Planck18.comoving_distance(z_test).value * h
    mask = np.ones(hp.nside2npix(nside))

    hot = []
    for k in range(len(betas)):
        m = np.zeros(hp.nside2npix(nside))
        m[hp.query_disc(nside, hp.ang2vec(lv[k], bv[k], lonlat=True),
                        np.radians(0.3))] = 1.0
        hot.append(m)

    c = npix // 2
    yy, xx = np.mgrid[-c:npix - c, -c:npix - c]   # xx=columna (izq<0), yy=fila

    def _centroid_angle(S):
        tot = np.nansum(S)
        xb = np.nansum(xx * S) / tot
        yb = np.nansum(yy * S) / tot
        return np.degrees(np.arctan2(yb, xb))     # 180 = apuntando a la izquierda (-x)

    def _ang_dist(a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)

    best, table = None, []
    for sign in (1.0, -1.0):
        for off in np.arange(0.0, 360.0, offset_step):
            err = np.empty(len(betas))
            for k in range(len(betas)):
                S, _ = stack_oriented([l_c], [b_c], [z_test], [pa[k]],
                                      hot[k], mask, size_mpch, npix,
                                      pa_offset=off, pa_sign=sign)
                err[k] = _ang_dist(_centroid_angle(S), 180.0)
            score = err.max()                     # el peor bearing (queremos ~0)
            table.append((sign, off, score))
            if best is None or score < best[2]:
                best = (sign, off, score)

    if verbose:
        print("[calib] criterio = peor |ángulo_centroide - 180deg| (queremos ~0):")
        for sign in (1.0, -1.0):
            b_ = min((t for t in table if t[0] == sign), key=lambda t: t[2])
            print(f"   sign={sign:+.0f}: offset*={b_[1]:6.1f}  max_err={b_[2]:5.1f} deg")
        print(f"[calib] RECOMENDADO: pa_sign={best[0]:+.0f}, pa_offset={best[1]:.1f}  "
              f"(max_err={best[2]:.1f} deg)")
        if best[2] > 10.0:
            print("[calib] (!) error angular alto: orientación inconsistente; "
                  "revisar convención del proyector/bearing antes de seguir.")
        else:
            print("[calib] OK: void apuntando a la izquierda de forma consistente.")
    return best[0], best[1]


# ----------------------------------------------------------------------
# suite completo (señal + nulls) para una muestra
# ----------------------------------------------------------------------
# Modos de null (rompen el vínculo físico cúmulo->void de distinta forma):
#   'shuffled' : permuta PA entre cúmulos -> conserva la distribución real de PA.
#   'random'   : PA uniforme -> piso de ruido del estimador.
#   'rotated'  : rotación rígida del catálogo sobre el mapa real (mantiene la
#                geometría interna y el PA, pero lo manda a un parche no
#                correlacionado) -> piso de SISTEMÁTICOS del mapa/máscara.
NULL_MODES = ('shuffled', 'random', 'rotated')


def _null_realizations(lon, lat, z, pa, cmb_map, mask, size_mpch, npix, bins,
                       pa_sign, pa_offset, n_real, mode, rng):
    """
    n_real realizaciones del null `mode` (UN stack cada una, sin JK).
    Returns: (diffs_void, diffs_fil), cada uno array (n_real, nbins).
    """
    nb = len(bins) - 1
    dv = np.empty((n_real, nb))
    df = np.empty((n_real, nb))
    lon = np.asarray(lon); lat = np.asarray(lat)
    for i in range(n_real):
        if mode == 'shuffled':
            S, C = stack_oriented(lon, lat, z, rng.permutation(pa), cmb_map, mask,
                                  size_mpch, npix, pa_sign=pa_sign, pa_offset=pa_offset)
        elif mode == 'random':
            S, C = stack_oriented(lon, lat, z, pa, cmb_map, mask, size_mpch, npix,
                                  randomize_pa=True, rng=rng)
        elif mode == 'rotated':
            rot = hp.Rotator(rot=(rng.uniform(0, 360), rng.uniform(-90, 90),
                                  rng.uniform(0, 360)), deg=True)
            rlon, rlat = rot(lon, lat, lonlat=True)
            S, C = stack_oriented(rlon, rlat, z, pa, cmb_map, mask, size_mpch, npix,
                                  pa_sign=pa_sign, pa_offset=pa_offset)
        else:
            raise ValueError(f"modo de null desconocido: {mode}")
        P = radial_profile_regions(S, C, size_mpch, bins)
        dv[i] = P['diff_void']
        df[i] = P['diff_fil']
    return dv, df


def _pvalue_from_null(diffs, cov_signal, diff_signal):
    """
    p-value empírico del estadístico T = d^T C^-1 d con la distribución `diffs`
    del null (cada fila una realización). C = covarianza JK de la señal.
    p = (#{T_null >= T_señal} + 1)/(n_real + 1).
    """
    good = np.isfinite(diff_signal) & np.isfinite(np.diag(cov_signal))
    Cinv = np.linalg.pinv(cov_signal[np.ix_(good, good)])

    def _T(d):
        dd = np.nan_to_num(d)[good]
        return float(dd @ Cinv @ dd)

    T_sig = _T(diff_signal)
    T_null = np.array([_T(d) for d in diffs])
    n_real = len(diffs)
    p = (np.sum(T_null >= T_sig) + 1) / (n_real + 1)
    return {'n_real': n_real, 'T_signal': T_sig, 'T_null': T_null, 'p_value': p,
            'diff_mean': np.nanmean(diffs, 0), 'diff_std': np.nanstd(diffs, 0)}


def run_suite(lon, lat, z, pa, cmb_map, mask, size_mpch, npix, bins,
              pa_sign, pa_offset, n_subsamples=25, n_null_real=100, rng=None,
              label="", null_modes=NULL_MODES):
    """
    Para UNA muestra (binneado de octantes):
      signal : orientado al void (sectors_with_errors + covarianza JK de cada diff).
      parity : SIN orientar (pa_sign=0) -> null instrumental (determinista).
      nulls  : por cada modo de NULL_MODES, distribución MC de T (n_null_real
               realizaciones) -> p-value empírico + banda, TANTO para diff_void
               como para diff_fil.

    Estructura de salida:
      out['signal'], out['parity'] : dicts de sectors_with_errors.
      out['nulls'][mode]['void'|'fil'] : dict de _pvalue_from_null.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    lon = np.asarray(lon); lat = np.asarray(lat)
    z = np.asarray(z); pa = np.asarray(pa)
    print(f"[suite] '{label}': N={len(lon)}")

    common = dict(cmb_map=cmb_map, mask=mask, size_mpch=size_mpch, npix=npix, bins=bins)
    out = {'label': label, 'n': len(lon)}
    out['signal'] = sectors_with_errors(lon, lat, z, pa, pa_sign=pa_sign,
                                        pa_offset=pa_offset, n_subsamples=n_subsamples,
                                        **common)
    out['parity'] = sectors_with_errors(lon, lat, z, pa, pa_sign=0.0, pa_offset=0.0,
                                        n_subsamples=n_subsamples, **common)

    cov_v, d_v = out['signal']['cov_void'], out['signal']['diff_void']
    cov_f, d_f = out['signal']['cov_fil'], out['signal']['diff_fil']
    out['nulls'] = {}
    for mode in null_modes:
        dv, df = _null_realizations(lon, lat, z, pa, pa_sign=pa_sign,
                                    pa_offset=pa_offset, n_real=n_null_real,
                                    mode=mode, rng=rng, **common)
        nv = _pvalue_from_null(dv, cov_v, d_v)
        nf = _pvalue_from_null(df, cov_f, d_f)
        out['nulls'][mode] = {'void': nv, 'fil': nf}
        print(f"[null:{mode:8s}] n_real={n_null_real}  "
              f"void: T={nv['T_signal']:.1f} p={nv['p_value']:.3f}  | "
              f"fil: T={nf['T_signal']:.1f} p={nf['p_value']:.3f}")
    return out


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
_NULL_COLORS = {'shuffled': 'xkcd:purple', 'random': 'xkcd:slate blue',
                'rotated': 'xkcd:green'}
_NULL_LABEL = {'shuffled': 'barajado', 'random': 'aleatorio', 'rotated': 'mapa rotado'}


def _draw_octants(ax, half):
    """Bordes de octantes (22.5+45k deg) sobre el panel del mapa orientado."""
    for ang in np.arange(22.5, 360, 45.0):
        a = np.radians(ang)
        ax.plot([0, half * np.cos(a)], [0, half * np.sin(a)],
                color='w', ls=':', lw=0.6, alpha=0.6)


def _plot_diff_panel(ax, r, diff, err, parity, nulls, which, ylabel):
    """Panel de una diff (void o fil) con banda de paridad + nulls + p-values."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ax.axhline(0, color='k', ls=':', alpha=0.6)
    ax.fill_between(r, parity[f'diff_{which}'] - parity[f'err_diff_{which}'],
                    parity[f'diff_{which}'] + parity[f'err_diff_{which}'],
                    color='xkcd:grey', alpha=0.22)
    handles = [plt.Line2D([], [], color='xkcd:crimson', marker='o', label='dato'),
               Patch(color='xkcd:grey', alpha=0.22, label='paridad (sin orientar)')]
    for mode, nd in nulls.items():
        n = nd[which]
        col = _NULL_COLORS.get(mode, 'xkcd:grey')
        ax.fill_between(r, n['diff_mean'] - n['diff_std'],
                        n['diff_mean'] + n['diff_std'], color=col, alpha=0.20)
        handles.append(Patch(color=col, alpha=0.20,
                             label=f"null {_NULL_LABEL.get(mode, mode)} (p={n['p_value']:.3f})"))
    ax.errorbar(r, diff, yerr=err, fmt='o-', color='xkcd:crimson',
                capsize=3, lw=1.8, zorder=5)
    ax.set_xlabel("r [Mpc/h]"); ax.set_ylabel(ylabel)
    ax.legend(handles=handles, fontsize=7); ax.grid(alpha=0.25)


def plot_suite(suite, outpath, size_mpch, label=""):
    """
    Figura 2x3:
      (0,0) mapa orientado con bordes de octantes.
      (0,1) perfiles cara (C) vs opuesta (D) + isótropo.
      (0,2) diff_void = C - D, con paridad + nulls + p-values.
      (1,0) chequeo de simetría arriba/abajo (diff_parity, debe ~0).
      (1,1) perfiles a lo largo del filamento (N+S) vs perpendicular (W+E).
      (1,2) diff_fil = along - perp, con paridad + nulls + p-values.
    """
    import matplotlib.pyplot as plt
    sig = suite['signal']
    par = suite['parity']
    nulls = suite['nulls']
    r = sig['r']
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) mapa
    ext = [-size_mpch / 2, size_mpch / 2, -size_mpch / 2, size_mpch / 2]
    im = ax[0, 0].imshow(sig['map'], origin='lower', cmap='viridis', extent=ext)
    _draw_octants(ax[0, 0], size_mpch / 2)
    ax[0, 0].set_title(f"Stack orientado (void=izq)\n{label}  N={suite['n']}")
    ax[0, 0].set_xlabel("Mpc/h"); ax[0, 0].set_ylabel("Mpc/h")
    plt.colorbar(im, ax=ax[0, 0], fraction=0.046)

    # (0,1) cara vs opuesta
    ax[0, 1].errorbar(r, sig['C'], yerr=sig['err_C'], fmt='o-', color='xkcd:teal',
                      capsize=2, label='cara al void (C=NW+W+SW)')
    ax[0, 1].errorbar(r, sig['D'], yerr=sig['err_D'], fmt='s-', color='xkcd:orange',
                      capsize=2, label='opuesta (D=NE+E+SE)')
    ax[0, 1].plot(r, sig['full'], 'k:', alpha=0.6, label='isótropo')
    ax[0, 1].axhline(0, color='k', ls=':', alpha=0.5)
    ax[0, 1].set_xlabel("r [Mpc/h]"); ax[0, 1].set_ylabel("señal")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.25)

    # (0,2) diff_void
    _plot_diff_panel(ax[0, 2], r, sig['diff_void'], sig['err_diff_void'], par, nulls,
                     'void', "(cara al void) - (opuesta)")
    ax[0, 2].set_title("Mirar al void")

    # (1,0) simetría arriba/abajo (null interno)
    ax[1, 0].axhline(0, color='k', ls=':', alpha=0.6)
    ax[1, 0].errorbar(r, sig['diff_parity'], yerr=sig['err_diff_parity'], fmt='o-',
                      color='xkcd:dark grey', capsize=2)
    ax[1, 0].set_title("Simetría arriba/abajo (debe ~0)")
    ax[1, 0].set_xlabel("r [Mpc/h]"); ax[1, 0].set_ylabel("up - down")
    ax[1, 0].grid(alpha=0.25)

    # (1,1) along vs perp
    ax[1, 1].errorbar(r, sig['along'], yerr=sig['err_along'], fmt='o-',
                      color='xkcd:purple', capsize=2, label='a lo largo del filamento (N+S)')
    ax[1, 1].errorbar(r, sig['perp'], yerr=sig['err_perp'], fmt='s-',
                      color='xkcd:goldenrod', capsize=2, label='perpendicular / eje void (W+E)')
    ax[1, 1].plot(r, sig['full'], 'k:', alpha=0.6, label='isótropo')
    ax[1, 1].axhline(0, color='k', ls=':', alpha=0.5)
    ax[1, 1].set_xlabel("r [Mpc/h]"); ax[1, 1].set_ylabel("señal")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.25)

    # (1,2) diff_fil
    _plot_diff_panel(ax[1, 2], r, sig['diff_fil'], sig['err_diff_fil'], par, nulls,
                     'fil', "(a lo largo filamento) - (perp)")
    ax[1, 2].set_title("Elongación del filamento")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"[plot] suite guardada en {outpath}")
    plt.close()
