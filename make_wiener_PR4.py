#%% imports
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import camb
from scipy.ndimage import uniform_filter1d


#%% CMB loading
nside = 2048
lmax = 2048

cmb_alm = hp.read_alm('/home/ftoscano/Doctorado/Data/CMB/Lensing/KAPPA_PR4klm_MV.fits')
kappa_map = hp.alm2map(cmb_alm, nside)
common_mask = hp.read_map('/home/ftoscano/Doctorado/Data/CMB/Lensing/Common_mask_PR4Lensing_2048.fits')

w2 = np.mean(common_mask**2)
pw = hp.pixwin(nside, lmax=lmax)

cl_tot = hp.anafast(kappa_map * common_mask , lmax= lmax) / (w2 * pw**2)

#%% Fiducial CMB
pars = camb.set_params(H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544, ns=0.9649, As=2.1e-9, mnu=0.06)
pars.set_for_lmax(lmax, lens_potential_accuracy=2)
results = camb.get_results(pars)

clpp = results.get_lens_potential_cls(lmax=lmax)[:,0]
cl_kk_fid = (np.pi / 2.0) * clpp

#%% Noise
N_mean = cl_tot - cl_kk_fid
N_smooth = uniform_filter1d(N_mean, size=50, mode='nearest')
N_smooth = np.clip(N_smooth, 0.0, None)

denom = cl_kk_fid + N_smooth
with np.errstate(divide='ignore', invalid='ignore'):
	W = np.where(denom > 0, cl_kk_fid / denom, 0.0)
W = np.clip(W, 0.0, 1.0)
W[:2] = 0.0

#%% Saving
L = np.arange(cl_kk_fid.size)
out = np.column_stack([L, N_smooth, denom])
outpath = '/home/ftoscano/Doctorado/Data/CMB/Lensing/nlkk_PR4_MV.dat'
np.savetxt(outpath, out, header = 'L	N_L^kk	(C_L^kk_fid + N_L^kk)	[PR4 empirical]')
print(f'Guardado: {outpath}')

