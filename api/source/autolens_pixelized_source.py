#%%
import numpy as np
import autolens as al 
import pickle
import sys
sys.path.append('/home/cao/data_disk/autolens_xycao/potential_correction')
import os 
from os import path
import time
from scipy.optimize import dual_annealing
current_dir, current_file_name = os.path.split(os.path.abspath(__file__))

with open(f'{current_dir}/masked_imaging.pkl','rb') as f:
    masked_imaging = pickle.load(f) #this is a autolens masked imaging object, see code below


lens_galaxy = al.Galaxy(
    redshift=0.2,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.2,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

def optimize_func_log(x):
    pixelization_shape_2d = (50, 50)
    pixelization = al.pix.DelaunayMagnification(shape=pixelization_shape_2d)

    source_galaxy = al.Galaxy(
        redshift=0.6,
        pixelization=pixelization,
        regularization=al.reg.Constant(coefficient=10**(x[0])),
    )
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    fit = al.FitImaging(
        dataset=masked_imaging,
        tracer=tracer,
        settings_inversion=al.SettingsInversion(use_w_tilde=False),
        settings_pixelization=al.SettingsPixelization(use_border=True),
    )
    fit_log_evidence = fit.log_evidence

    return -1.0*fit_log_evidence

# t0 = time.time()
# ret = dual_annealing(optimize_func_log, bounds=[[-5,4],])
# t1 = time.time()
# print(10**(ret.x))
# print(ret.fun)
# print(f'total time elapse: {t1-t0}')

#above regularization strength fitting result is:
# [2.33469951]
# -4945.056446449687
# total time elapse: 319.6242527961731

#%%
pixelization_shape_2d = (50, 50)
pixelization = al.pix.DelaunayMagnification(shape=pixelization_shape_2d)
source_galaxy = al.Galaxy(
    redshift=0.6,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=2.33469951),
)
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
fitter = al.FitImaging(
    dataset=masked_imaging,
    tracer=tracer,
    settings_inversion=al.SettingsInversion(use_w_tilde=False),
    settings_pixelization=al.SettingsPixelization(use_border=True),
)

from matplotlib import  pyplot as plt
from plot import pixelized_source as ps_plot
fig, axes = plt.subplots(figsize=(12,12), nrows=2, ncols=2)
ps_plot.visualize_unmasked_1d_image(masked_imaging.data, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[0,0])
ps_plot.visualize_unmasked_1d_image(fitter.inversion.mapped_reconstructed_data, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[0,1])
ps_plot.visualize_unmasked_1d_image(fitter.normalized_residual_map, masked_imaging.mask, dpix=masked_imaging.pixel_scales[0], ax=axes[1,0])
ps_plot.visualize_source(fitter.inversion.mapper_list[0].source_pixelization_grid, fitter.inversion.reconstruction, ax=axes[1,1])
fig.savefig('autolens_pix_src.jpg')
