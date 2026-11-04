"""
make_wiener_PR4.py
==================
Empirical PR4 noise spectrum for the Wiener filter:
    N_L = C_L^{data} - C_L^{kk, fid}
with C_L^{data} the pseudo-C_L of the PR4 MV convergence on the lensing mask
(corrected by the mean of mask^2) and C_L^{kk, fid} from CAMB (Planck 2018 best fit).
Saved as (L, N_L, C_L^fid + N_L), the format read by apply_wiener_filter.

The map is synthesised from the alm without pixel window, so its pseudo-C_L must
NOT be divided by the pixel window.
"""
#%% imports
import healpy as hp
import numpy as np
import camb
from scipy.ndimage import uniform_filter1d

LENS = '/home/ftoscano/Doctorado/Data/CMB/PLANCK/Lensing/'
nside = 2048
lmax = 2048

#%% data
cmb_alm = hp.read_alm(LENS + 'KAPPA_PR4klm_MV.fits')
kappa_map = hp.alm2map(cmb_alm, nside)              # no pixel window applied
common_mask = hp.read_map(LENS + 'Common_mask_PR4Lensing_2048.fits')
w2 = np.mean(common_mask ** 2)
cl_tot = hp.anafast(kappa_map * common_mask, lmax=lmax) / w2

#%% fiducial
pars = camb.set_params(H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544, ns=0.9649, As=2.1e-9, mnu=0.06)
pars.set_for_lmax(lmax, lens_potential_accuracy=2)
results = camb.get_results(pars)
clpp = results.get_lens_potential_cls(lmax=lmax)[:, 0]     # [L(L+1)]^2 C_L^phiphi / 2pi
cl_kk_fid = (np.pi / 2.0) * clpp                            # C_L^kk = [L(L+1)]^2 C_L^phiphi / 4

#%% noise
N_smooth = np.clip(uniform_filter1d(cl_tot - cl_kk_fid, size=50, mode='nearest'), 0.0, None)
denom = cl_kk_fid + N_smooth

#%% save
L = np.arange(cl_kk_fid.size)
outpath = LENS + 'nlkk_PR4_MV.dat'
np.savetxt(outpath, np.column_stack([L, N_smooth, denom]),
           header='L\tN_L^kk\t(C_L^kk_fid + N_L^kk)\t[PR4 empirical, no pixel-window division]')
print(f'Saved: {outpath}')
