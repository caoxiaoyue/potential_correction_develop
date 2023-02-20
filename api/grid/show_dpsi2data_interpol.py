import grid_util
import numpy as np
from matplotlib import pyplot as plt
import autolens as al 
import os

current_dir, current_file_name = os.path.split(os.path.abspath(__file__))

#test data-dpsi pairing
grid_data = al.Grid2D.uniform(shape_native=(100,100), pixel_scales=0.1, sub_size=1)
xgrid_data = grid_data.native[:,:,1]
ygrid_data = grid_data.native[:,:,0]
rgrid = np.sqrt(xgrid_data**2 + ygrid_data**2)
annular_mask = (rgrid>4.0) #np.logical_or(rgrid<1.0, rgrid>4.0)
grid_obj = grid_util.SparseDpsiGrid(annular_mask, 0.1, shape_2d_dpsi=(50,50))
grid_obj.show_grid()
def test_func(xgrid, ygrid):
    return 2*xgrid + 3*ygrid
data_image2d_true = test_func(grid_obj.xgrid_data, grid_obj.ygrid_data)
dpsi_image2d_true = test_func(grid_obj.xgrid_dpsi, grid_obj.ygrid_dpsi)
data_image1d_true = test_func(grid_obj.xgrid_data_1d, grid_obj.ygrid_data_1d)
dpsi_image1d_true = test_func(grid_obj.xgrid_dpsi_1d, grid_obj.ygrid_dpsi_1d)
data_image2d_recover = np.zeros_like(data_image2d_true)
data_image2d_recover.reshape(-1)[grid_obj.indices_1d_data] = data_image1d_true[:] #should not use flatten() here!!!
dpsi_image2d_recover = np.zeros_like(dpsi_image2d_true)
dpsi_image2d_recover.reshape(-1)[grid_obj.indices_1d_dpsi] = dpsi_image1d_true[:]
data_image1d_map = np.matmul(grid_obj.map_matrix, dpsi_image1d_true) 
data_image2d_map = np.zeros_like(data_image2d_true)
data_image2d_map.reshape(-1)[grid_obj.indices_1d_data] = data_image1d_map[:]
plt.figure(figsize=(10,15))
plt.subplot(321)
plt.imshow(data_image2d_true, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(322)
plt.imshow(dpsi_image2d_true, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(323)
plt.imshow(data_image2d_recover, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(324)
plt.imshow(dpsi_image2d_recover, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(325)
plt.imshow(data_image2d_map, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.subplot(326)
plt.imshow(data_image2d_map-data_image2d_recover, extent=grid_obj.image_bound)
plt.colorbar(fraction=0.046, pad=0.04)
plt.savefig(f'{current_dir}/png/itp_image.png')
plt.close()