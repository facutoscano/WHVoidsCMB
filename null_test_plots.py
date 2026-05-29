#%% IMPORTS
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
from astropy.cosmology import Planck18
import Functions_module as fm

#%% Configuration
config = {
    'data_folder': '/home/ftoscano/Doctorado/Data/',
    'output_folder': '/home/ftoscano/Doctorado/Proyectos/WHVoidsCMB/Results/',
    'release': 'PR4',
    'zmin': 0.2, 'zmax': 0.65,
    'rmin': 35.0, 'rmax': 62.7,
    'max_Rvoid': 2.5,
    'nside': 2048,
    'nside_plot': 2048 
}

def main():
    print("Diagnostic plots")
    data_folder = config['data_folder']
    output_folder = config['output_folder']
    nside = config['nside']
    nside_plot = config['nside_plot']
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #%% Loading data and masks
    print("Loading data and masks...")
    col_names = ['R_void', 'l', 'b', 'z']
    voids_data = pd.read_csv(f'{data_folder}CATALOGOS/WenHan_voids.dat', sep='\s+', names=col_names, header=None)
    
    mask_cat = (voids_data['z'] >= config['zmin']) & (voids_data['z'] < config['zmax']) & \
               (voids_data['R_void'] >= config['rmin']) & (voids_data['R_void'] < config['rmax'])
    cat_final = voids_data[mask_cat].copy()
    
    l_v = cat_final['l'].values
    b_v = cat_final['b'].values
    z_v = cat_final['z'].values
    rv_v = cat_final['R_void'].values
    n_voids = len(cat_final)
    print(f"Number of voids: {n_voids}")

    common_mask_file = f'{data_folder}CMB/Lensing/Common_mask_Lensing_2048.fits'
    common_mask = hp.read_map(common_mask_file)
    survey_mask = fm.footprint_mask(l_v, b_v, output_nside=nside, footprint_nside=32)
    effective_mask = common_mask * survey_mask

    eff_mask_plot = hp.ud_grade(effective_mask, nside_out=nside_plot)
    
    #%% PLOT 1
    print("Generating Plot 1: Angular Overlap...")
    map_rings = np.zeros(hp.nside2npix(nside_plot))
    
    for i in range(n_voids):
        box_size_mpch = config['max_Rvoid'] * rv_v[i]
        theta_deg = fm.get_angularsize_comoving(z_v[i], box_size_mpch)
        
        vec = hp.ang2vec(l_v[i], b_v[i], lonlat=True)
        theta_in_deg = max(0.001, theta_deg - 0.1) 
        
        pix_out = hp.query_disc(nside_plot, vec, radius=np.radians(theta_deg))
        pix_in = hp.query_disc(nside_plot, vec, radius=np.radians(theta_in_deg))
        
        boundary = np.setdiff1d(pix_out, pix_in)
        map_rings[boundary] += 1

    map_rings[map_rings == 0] = hp.UNSEEN

    plt.figure(figsize=(10, 6))
    hp.mollview(map_rings, title=f"r = {config['max_Rvoid']} R_v", 
                cmap='turbo', return_projected_map=False, min=1)
    plt.savefig(os.path.join(output_folder, "Diag_1_Void_Overlap_Rings.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    #%% PLOT 2
    print("Generating Plot 2: Rotation Test...")
    rot_angles = [90, 180, 270, 360]
    
    fig = plt.figure(figsize=(12, 8))
    for i, ang in enumerate(rot_angles):
        rot_mask = fm.rotate_map(eff_mask_plot, rot_angles=[ang, 0, 0])
        
        hp.mollview(rot_mask, sub=(2, 2, i+1), 
                    title=f"Rotation in Longitude: {ang}º", 
                    cbar=False, cmap='Blues')
        
        hp.projscatter(l_v, b_v, lonlat=True, s=0.5, color='red', alpha=0.5)

    plt.savefig(os.path.join(output_folder, "Diag_2_Rotations_Test.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    #%% PLOT 3
    print("Generating Plot 3: Random Point Distribution...")
    fig = plt.figure(figsize=(12, 8))
    
    
    hp.mollview(eff_mask_plot, sub=(2, 2, 1), title="Footprint Mask", cbar=False, cmap='Greys')
    
    for i in range(3):
        rand_l, rand_b = fm.generate_random(effective_mask, n_voids, nside)
        
        hp.mollview(eff_mask_plot, sub=(2, 2, i+2), title=f"Random realization {i+1}", cbar=False, cmap='Greys')
        hp.projscatter(rand_l, rand_b, lonlat=True, s=0.5, color='blue', alpha=0.5)

    plt.savefig(os.path.join(output_folder, "Diag_3_Random_Distributions.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Done! All the plots are saved in {output_folder}")

if __name__ == "__main__":
    main()
