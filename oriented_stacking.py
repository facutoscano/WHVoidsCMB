"""
oriented_stacking.py
====================
Stacking gnomónico ORIENTADO según la dirección al void, perfiles radiales
partidos en mitades (cara al void "C" vs cara opuesta "D") y errores jackknife
sobre L, R y la diferencia L-R. Incluye calibración de la convención de
orientación y null de orientación aleatoria.

Escala
------
Comóvil Mpc/h, consistente con R_void. `size_mpch` = ancho TOTAL del stamp en
Mpc/h. (El ángulo proyectado es el mismo que con Mpc propios; lo que cambia es la
etiqueta del eje, y la elegimos comóvil Mpc/h para comparar con R_void.)

Convención de mitades
---------------------
Tras calibrar, el void queda a la IZQUIERDA del stamp. Entonces:
  - mitad IZQUIERDA  -> 'left'  -> CARA AL VOID   (media luna "C")
  - mitad DERECHA    -> 'right' -> CARA OPUESTA   (media luna "D")
La diferencia se define diff = left - right = (cara al void) - (opuesta).

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
# perfiles radiales: completo + mitades
# ----------------------------------------------------------------------
def _geom(npix, size_mpch):
    c = npix // 2
    yy, xx = np.ogrid[-c:npix - c, -c:npix - c]
    r = np.sqrt(xx * xx + yy * yy) * (size_mpch / npix)   # Mpc/h comóvil
    left = (xx < 0)            # columna central (xx==0) excluida de ambas
    right = (xx > 0)
    return r, left, right


def radial_profile_halves(SUM, COUNT, size_mpch, bins):
    """
    Perfil pesado por COUNT en cada anillo, separando mitades.
    Returns: r_centers, full, left, right, diff(=left-right)
    """
    npix = SUM.shape[0]
    r, left, right = _geom(npix, size_mpch)

    def _prof(extra_mask):
        out = []
        for i in range(len(bins) - 1):
            ring = (r >= bins[i]) & (r < bins[i + 1]) & extra_mask
            den = np.nansum(COUNT[ring])
            out.append(np.nansum(SUM[ring]) / den if den > 0 else np.nan)
        return np.array(out)

    full = _prof(np.ones_like(left, dtype=bool))
    L = _prof(left)
    R = _prof(right)
    rc = 0.5 * (bins[:-1] + bins[1:])
    return rc, full, L, R, (L - R)


# ----------------------------------------------------------------------
# errores jackknife (KMeans en la esfera) para L, R, full y diff
# ----------------------------------------------------------------------
def halves_with_errors(lon, lat, z, pa_deg, cmb_map, mask, size_mpch, npix, bins,
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

    rc, full, L, R, D = radial_profile_halves(SUM, COUNT, size_mpch, bins)

    jkF, jkL, jkR, jkD = [], [], [], []
    for k in range(n_subsamples):
        _, f, l, r, d = radial_profile_halves(SUM - sums[k], COUNT - counts[k],
                                              size_mpch, bins)
        jkF.append(f); jkL.append(l); jkR.append(r); jkD.append(d)

    def _cov(jk, best):
        jk = np.array(jk)
        delta = np.nan_to_num(jk - best)
        return (n_subsamples - 1) / n_subsamples * (delta.T @ delta)

    covF, covL, covR, covD = (_cov(jkF, full), _cov(jkL, L),
                              _cov(jkR, R), _cov(jkD, D))
    return {
        'r': rc, 'full': full, 'left': L, 'right': R, 'diff': D,
        'err_full': np.sqrt(np.diag(covF)),
        'err_left': np.sqrt(np.diag(covL)),
        'err_right': np.sqrt(np.diag(covR)),
        'err_diff': np.sqrt(np.diag(covD)),
        'cov_diff': covD, 'n': n, 'map': np.divide(SUM, COUNT,
                                                   out=np.full_like(SUM, np.nan),
                                                   where=COUNT > 0),
    }


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
# plotting
# ----------------------------------------------------------------------
def plot_halves(res, outpath, null_res=None, size_mpch=None, label=""):
    """
    res, null_res : dicts de halves_with_errors (null con randomize_pa=True).
    Panel 1: mapa orientado. Panel 2: left/right/full. Panel 3: diff +- err, con
    banda del null si se provee.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    if size_mpch is not None and res.get('map') is not None:
        ext = [-size_mpch / 2, size_mpch / 2, -size_mpch / 2, size_mpch / 2]
        im = ax[0].imshow(res['map'], origin='lower', cmap='viridis', extent=ext)
        ax[0].axvline(0, color='w', ls='--', lw=0.8)
        ax[0].set_title(f"Stack orientado (void a la izq.)\n{label}")
        ax[0].set_xlabel("Mpc/h"); ax[0].set_ylabel("Mpc/h")
        plt.colorbar(im, ax=ax[0], fraction=0.046)

    r = res['r']
    ax[1].errorbar(r, res['left'], yerr=res['err_left'], fmt='o-',
                   color='xkcd:teal', capsize=2, label='izq = cara al void (C)')
    ax[1].errorbar(r, res['right'], yerr=res['err_right'], fmt='s-',
                   color='xkcd:orange', capsize=2, label='der = opuesta (D)')
    ax[1].plot(r, res['full'], 'k:', alpha=0.6, label='isótropo')
    ax[1].axhline(0, color='k', ls=':', alpha=0.5)
    ax[1].set_xlabel("r [Mpc/h]"); ax[1].set_ylabel("señal")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    ax[2].errorbar(r, res['diff'], yerr=res['err_diff'], fmt='o-',
                   color='xkcd:crimson', capsize=2, label='L - R (dato)')
    if null_res is not None:
        ax[2].fill_between(r, null_res['diff'] - null_res['err_diff'],
                           null_res['diff'] + null_res['err_diff'],
                           color='gray', alpha=0.35, label='null (PA aleatorio)')
    ax[2].axhline(0, color='k', ls=':', alpha=0.6)
    ax[2].set_xlabel("r [Mpc/h]"); ax[2].set_ylabel("(cara al void) - (opuesta)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"[plot] guardado en {outpath}")
    plt.close()


# ----------------------------------------------------------------------
# suite completo (señal + nulls) para una muestra
# ----------------------------------------------------------------------
def null_pvalue(lon, lat, z, pa, cmb_map, mask, size_mpch, npix, bins,
                cov_signal, diff_signal, pa_sign, pa_offset,
                n_real=100, mode='shuffled', rng=None):
    """
    Distribución MC del estadístico T = d^T C^-1 d bajo la hipótesis nula,
    repitiendo n_real realizaciones (UN stack cada una, sin JK). C = covarianza
    JK de la señal. p-value empírico = (#{T_null >= T_señal} + 1)/(n_real + 1).

      mode='shuffled' : permuta PA entre cúmulos -> rompe el vínculo físico pero
                        conserva la distribución real de PA (null fuerte).
      mode='random'   : PA uniforme -> piso de ruido del estimador.

    Esto reemplaza al p~chi2 gaussiano (que asume la cov JK perfecta) por un
    p-value calibrado por simulación, y de paso da una banda (media +- std de la
    'diff' sobre realizaciones) mejor que la de un solo draw.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    good = np.isfinite(diff_signal) & np.isfinite(np.diag(cov_signal))
    Cinv = np.linalg.pinv(cov_signal[np.ix_(good, good)])

    def _T(d):
        dd = np.nan_to_num(d)[good]
        return float(dd @ Cinv @ dd)

    T_sig = _T(diff_signal)
    T_null = np.empty(n_real)
    diffs = np.empty((n_real, len(bins) - 1))
    for i in range(n_real):
        if mode == 'shuffled':
            S, C = stack_oriented(lon, lat, z, rng.permutation(pa), cmb_map, mask,
                                  size_mpch, npix, pa_sign=pa_sign, pa_offset=pa_offset)
        else:
            S, C = stack_oriented(lon, lat, z, pa, cmb_map, mask, size_mpch, npix,
                                  randomize_pa=True, rng=rng)
        _, _, _, _, d = radial_profile_halves(S, C, size_mpch, bins)
        diffs[i] = d
        T_null[i] = _T(d)
    p = (np.sum(T_null >= T_sig) + 1) / (n_real + 1)
    print(f"[null:{mode:8s}] n_real={n_real}  T_signal={T_sig:.1f}  p-value={p:.3f}")
    return {'mode': mode, 'n_real': n_real, 'T_signal': T_sig, 'T_null': T_null,
            'p_value': p, 'diff_mean': np.nanmean(diffs, 0),
            'diff_std': np.nanstd(diffs, 0)}


def run_suite(lon, lat, z, pa, cmb_map, mask, size_mpch, npix, bins,
              pa_sign, pa_offset, n_subsamples=25, n_null_real=100, rng=None, label=""):
    """
    Para UNA muestra:
      signal : orientado al void (halves_with_errors + covarianza JK).
      parity : SIN orientar (pa_sign=0) -> null instrumental de L-R (1 realización
               JK; es determinista, no Monte Carlo).
      null_shuffled / null_random : distribución MC de T con n_null_real
               realizaciones -> p-value empírico + banda (media +- std).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    lon = np.asarray(lon); lat = np.asarray(lat)
    z = np.asarray(z); pa = np.asarray(pa)
    print(f"[suite] '{label}': N={len(lon)}")

    common = dict(cmb_map=cmb_map, mask=mask, size_mpch=size_mpch, npix=npix, bins=bins)
    out = {'label': label, 'n': len(lon)}
    out['signal'] = halves_with_errors(lon, lat, z, pa, pa_sign=pa_sign,
                                       pa_offset=pa_offset, n_subsamples=n_subsamples,
                                       **common)
    out['parity'] = halves_with_errors(lon, lat, z, pa, pa_sign=0.0, pa_offset=0.0,
                                       n_subsamples=n_subsamples, **common)
    cov, dsig = out['signal']['cov_diff'], out['signal']['diff']
    out['null_shuffled'] = null_pvalue(lon, lat, z, pa, cov_signal=cov, diff_signal=dsig,
                                       pa_sign=pa_sign, pa_offset=pa_offset,
                                       n_real=n_null_real, mode='shuffled', rng=rng, **common)
    out['null_random'] = null_pvalue(lon, lat, z, pa, cov_signal=cov, diff_signal=dsig,
                                     pa_sign=pa_sign, pa_offset=pa_offset,
                                     n_real=n_null_real, mode='random', rng=rng, **common)
    return out


def plot_suite(suite, outpath, size_mpch, label=""):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    sig = suite['signal']
    r = sig['r']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ext = [-size_mpch / 2, size_mpch / 2, -size_mpch / 2, size_mpch / 2]
    im = ax[0].imshow(sig['map'], origin='lower', cmap='viridis', extent=ext)
    ax[0].axvline(0, color='w', ls='--', lw=0.8)
    ax[0].set_title(f"Stack orientado (void=izq)\n{label}  N={suite['n']}")
    ax[0].set_xlabel("Mpc/h"); ax[0].set_ylabel("Mpc/h")
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    ax[1].errorbar(r, sig['left'], yerr=sig['err_left'], fmt='o-', color='xkcd:teal',
                   capsize=2, label='izq = cara al void (C)')
    ax[1].errorbar(r, sig['right'], yerr=sig['err_right'], fmt='s-', color='xkcd:orange',
                   capsize=2, label='der = opuesta (D)')
    ax[1].plot(r, sig['full'], 'k:', alpha=0.6, label='isótropo (sin restar mean-field)')
    ax[1].axhline(0, color='k', ls=':', alpha=0.5)
    ax[1].set_xlabel("r [Mpc/h]"); ax[1].set_ylabel("señal")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    ax[2].axhline(0, color='k', ls=':', alpha=0.6)
    par = suite['parity']
    ax[2].fill_between(r, par['diff'] - par['err_diff'], par['diff'] + par['err_diff'],
                       color='xkcd:grey', alpha=0.22)
    for key, col in [('null_random', 'xkcd:slate blue'), ('null_shuffled', 'xkcd:purple')]:
        nd = suite[key]
        ax[2].fill_between(r, nd['diff_mean'] - nd['diff_std'],
                           nd['diff_mean'] + nd['diff_std'], color=col, alpha=0.22)
    ax[2].errorbar(r, sig['diff'], yerr=sig['err_diff'], fmt='o-', color='xkcd:crimson',
                   capsize=3, lw=1.8, zorder=5)
    p_rn = suite['null_random']['p_value']
    p_sh = suite['null_shuffled']['p_value']
    handles = [plt.Line2D([], [], color='xkcd:crimson', marker='o', label='L-R (dato)'),
               Patch(color='xkcd:grey', alpha=0.22, label='paridad'),
               Patch(color='xkcd:slate blue', alpha=0.22, label=f'null aleat. (p={p_rn:.3f})'),
               Patch(color='xkcd:purple', alpha=0.22, label=f'null barajado (p={p_sh:.3f})')]
    ax[2].legend(handles=handles, fontsize=8)
    ax[2].set_xlabel("r [Mpc/h]"); ax[2].set_ylabel("(cara al void) - (opuesta)")
    ax[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"[plot] suite guardada en {outpath}")
    plt.close()
