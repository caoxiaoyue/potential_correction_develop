# %%
import numpy as np
from matplotlib import pyplot as plt
from os import path
import autolens as al
import autolens.plot as aplt
import sys
sys.path.append('/home/cao/data_disk/autolens_xycao/potential_correction')
import pickle
import gzip
import grid_util
import potential_correction_util as pcu

with gzip.open('pt_data_1.pklz','rb') as f:
    obj = pickle.load(f)
    image_with_sub = obj['image_with_sub']
    image_no_sub = obj['image_no_sub']
    psi_with_sub = obj['psi_with_sub']
    psi_no_sub = obj['psi_no_sub']
    pixel_scales = obj['pixel_scales']
    dpsi_sparse_1d = obj['dpsi_sparse_1d']
    grid_obj = obj['grid_obj']
    tracer_with_sub = obj['tracer_with_sub']
    tracer_no_sub = obj['tracer_no_sub']

#-------Dpsi matrix
dpsi_gradient_matrix = pcu.dpsi_gradient_operator_from(grid_obj.Hx_dpsi, grid_obj.Hy_dpsi) 
#shape: [2Np, Np]

#-------src gradient matrix
src_grid_vec_numpy = np.vstack([grid_obj.ygrid_data_1d, grid_obj.xgrid_data_1d]).T
src_grid_vec_al = al.Grid2DIrregular(src_grid_vec_numpy)
alpha_src_yx = tracer_no_sub.deflections_yx_2d_from(src_grid_vec_al)
source_points = src_grid_vec_al - alpha_src_yx
source_values = tracer_with_sub.galaxies[1].image_2d_from(grid=source_points)

dpsi_grid_vec_numpy = np.vstack([grid_obj.ygrid_dpsi_1d, grid_obj.xgrid_dpsi_1d]).T
dpsi_grid_vec_al = al.Grid2DIrregular(dpsi_grid_vec_numpy)
alpha_dpsi_yx = tracer_no_sub.deflections_yx_2d_from(dpsi_grid_vec_al)
src_plane_dpsi_yx = dpsi_grid_vec_al - alpha_dpsi_yx
"""# %%
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(dpsi_grid_vec_al[:,1], dpsi_grid_vec_al[:,0], s=1.0)
plt.axis('square')
plt.subplot(122)
plt.scatter(src_plane_dpsi_yx[:,1], src_plane_dpsi_yx[:,0], s=1.0)
plt.axis('square')
plt.show()
"""
source_gradient = pcu.source_gradient_from(
    source_points, 
    source_values, 
    src_plane_dpsi_yx, 
    cross_size=1e-3,
)
source_gradient_matrix = pcu.source_gradient_matrix_from(source_gradient) #shape: [Np, 2Np]

#------------conformation matrix, see the C_f matrix (eq.7) in our document
Cf_matrix = np.copy(grid_obj.map_matrix)

pt_image_correction = -1.0*np.matmul(
    Cf_matrix,
    np.matmul(
        source_gradient_matrix,
        np.matmul(dpsi_gradient_matrix, dpsi_sparse_1d),
    )
)

# %%
pt_image_correction_2d = np.zeros_like(grid_obj.xgrid_data)
pt_image_correction_2d[~grid_obj.mask_data] = pt_image_correction
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(pt_image_correction_2d, cmap='jet', extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.subplot(122)
plt.imshow(image_with_sub-image_no_sub, cmap='jet', extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.tight_layout()
plt.show()

# %%
diff = pt_image_correction_2d-(image_with_sub-image_no_sub)
rel_diff = diff/(image_with_sub-image_no_sub)
plt.figure(figsize=(5,5))
plt.imshow(diff, cmap='jet', extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('square')
plt.tight_layout()
plt.show()

# %%
with open('linear_image_correction.pkl','wb') as f:
    obj = {
        'dpsi_sparse_1d': np.asarray(dpsi_sparse_1d),
        'pt_image_correction_true': np.asarray(pt_image_correction),
    }
    pickle.dump(obj,f)

# %%
