import numpy as np
from matplotlib import pyplot as plt
from os import path
import autolens as al
import autolens.plot as aplt
import sys
sys.path.append('/home/cao/data_disk/autolens_xycao/potential_correction')
import time
t0 = time.time()
grid = al.Grid2D.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
    sub_size=2,  
)

lens_galaxy = al.Galaxy(
    redshift=0.2,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.2,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    subhalo=al.mp.SphIsothermal(
        centre=(1.15, 0.0),
        einstein_radius=0.01,
    )
)

source_galaxy = al.Galaxy(
    redshift=0.6,
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, angle=60.0),
        intensity=0.8,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer_with_sub = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

lens_galaxy = al.Galaxy(
    redshift=0.2,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.2,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    # subhalo=al.mp.SphIsothermal(
    #     centre=(1.15, 0.0),
    #     einstein_radius=0.05,
    # )
)
tracer_no_sub = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])


image_with_sub = tracer_with_sub.image_2d_from(grid).binned.native #suppose this is the real data
image_no_sub = tracer_no_sub.image_2d_from(grid).binned.native #suppose this is the best-fit macro model (although it isn't, since source model can absorb signals from the subhalo)
residual_true = image_with_sub - image_no_sub #this residual is induced by the subhalo

t1 = time.time()
print('time elapse:', t1-t0)

t0 = time.time()
psi_with_sub = tracer_with_sub.potential_2d_from(grid).binned.native #lens potential with the subhalo
psi_no_sub = tracer_no_sub.potential_2d_from(grid).binned.native #lens potential without the subhalo
dpsi_2d = psi_with_sub - psi_no_sub
t1 = time.time()
print('time elapse:', t1-t0)


plt.figure(figsize=(10,10))
plt.subplot(221)
plt.imshow(image_with_sub, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(222)
plt.imshow(image_no_sub, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(223)
plt.imshow(residual_true, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(224)
plt.imshow(dpsi_2d, cmap='jet')
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('residual.png')


#%%
import grid_util
mask_data = al.Mask2D.circular_annular(
    shape_native=(200, 200), pixel_scales=0.05, inner_radius=1.2*(1-0.4), outer_radius=1.2*(1+0.4)
)
grid_obj = grid_util.SparseDpsiGrid(mask_data, 0.05, (200, 200))
grid_dpsi = al.Grid2D.uniform(shape_native=grid_obj.shape_2d_dpsi, pixel_scales=grid_obj.dpix_dpsi, sub_size=1)
dpsi_sparse_2d = tracer_with_sub.potential_2d_from(grid_dpsi).binned.native - tracer_no_sub.potential_2d_from(grid_dpsi).binned.native
dpsi_sparse_1d  = dpsi_sparse_2d[~(grid_obj.mask_dpsi)] #this is just the $\delta \psi$ term in eq.7 in our document

#%%
import pickle
import gzip
with gzip.open('pt_data_1.pklz','wb') as f:
    obj = {
        'image_with_sub': image_with_sub,
        'image_no_sub': image_no_sub,
        'psi_with_sub': psi_with_sub,
        'psi_no_sub': psi_no_sub,
        'pixel_scales': 0.05,
        'tracer_with_sub': tracer_with_sub,
        'tracer_no_sub': tracer_no_sub,
        'grid': grid,
        'dpsi_sparse_1d': dpsi_sparse_1d,
        'grid_obj': grid_obj,
    }
    pickle.dump(obj,f)



