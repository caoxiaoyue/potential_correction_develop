import grid_util
import numpy as np
from matplotlib import pyplot as plt
import autolens as al 
import os 

current_dir, current_file_name = os.path.split(os.path.abspath(__file__))

#An regular grid test
grid_data = al.Grid2D.uniform(shape_native=(20,20), pixel_scales=0.1, sub_size=1)
xgrid_data = grid_data.native[:,:,1]
ygrid_data = grid_data.native[:,:,0]
rgrid = np.sqrt(xgrid_data**2 + ygrid_data**2)
annular_mask = np.logical_or(rgrid<0.3, rgrid>0.7)
grid_obj = grid_util.SparseDpsiGrid(annular_mask, 0.1, shape_2d_dpsi=(10,10))
grid_obj.show_grid(output_file=f'{current_dir}/png/regular_annular_mask.png')