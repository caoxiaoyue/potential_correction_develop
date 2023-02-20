#%%
import numpy as np
import autolens as al 
import pickle
import pixelized_mass
import pixelized_source
import os 
from os import path 
import json
current_dir, current_file_name = os.path.split(os.path.abspath(__file__))

pixel_scale = 0.05
sub_size = 4
annular_width = 0.3
inner_mask_radius = 1.2*(1-annular_width)
outter_mask_radius = 1.2*(1+annular_width)


dataset_path = f'{current_dir}/dataset/sie_sis_sersic'
imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
)

with open(path.join(dataset_path, "positions.json"), 'r') as f:
    positions = json.load(f)


mask = al.Mask2D.circular_annular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    sub_size=sub_size,
    inner_radius=inner_mask_radius,
    outer_radius=outter_mask_radius,
)
masked_imaging = imaging.apply_mask(mask=mask)

with open(f'{current_dir}/psi2d_macro.pkl','rb') as f:
    psi_2d_macro = pickle.load(f)


#%%
import iterative_solve
potential_corrector = iterative_solve.IterativePotentialCorrect(
    masked_imaging,
    shape_2d_dpsi=(100, 100), #the shape of lens potential correction grid
    shape_2d_src=(50, 50), #the shape of source grid
)
potential_corrector.initialize_iteration(
    psi_2d_start=psi_2d_macro, 
    niter=50, #number of iteration
    lam_s_start=10, # the regualrization strength for the source; we manually increase this value to make the source smoother;
    # ensuring the`potential correction` algorithm can converge.
    lam_dpsi_start=1e9, # the regualrization strength for the lens potential
    lam_dpsi_type='2nd', #the regularization type for the lens potential
    psi_anchor_points=np.array([(-1.0,-1.0),(-1.0,1.0),(0.0,1.0)]), #the anchor points, which is used to remove the mass-sheet degeneracy; see sec-2.3 of our team document
    subhalo_fiducial_point=(1.25, 0.0), #the input true location of the subhalo
    save_fits=True,
)
potential_corrector.run_iter_solve()  #run the potential correction; the results of each iteration are saved in the result folder

#%%
"""
In the result folder, there are lots of jpg file, named as 0.jpg, 1.jpg, ... etc.
0.jpg represents the result of iteration-0, which is just the macro model (without any potential correction)
In image 2.jpg, the meaning of different panels is:
row-1, col-1: the signal-to-noise ratio map of the data
row-1, col-2: the data image
row-1, col-3: the model image of current iteration
row-2, col-1: the normalized residual map of currrent iteration. i.e, (data_map - model_map)/noise_map
row-2, col-2: the source reconstruction of current iteration
row-2, col-3: the potential correction of current iteration
row-3, col-1: the corresponding convergence correction of current iteration
row-3, col-2: the cumulative potential correction, by summing the potential correction of this and all previous iteration
row-3, col-3: the cumulative convergence correction, calculated from the row-2, col-3 panel.
"""