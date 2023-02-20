#%%dpix=0.054th
import autolens as al
import numpy as np
import grid_util

grid_data = al.Grid2D.uniform(shape_native=(20,20), pixel_scales=0.5, sub_size=1)
xgrid_data = grid_data.native[:,:,1]
ygrid_data = grid_data.native[:,:,0]
rgrid = np.sqrt(xgrid_data**2 + ygrid_data**2)
annular_mask = np.logical_or(rgrid<1.5, rgrid>4.0)
grid_obj = grid_util.SparseDpsiGrid(annular_mask, 0.5, shape_2d_dpsi=(10,10))
grid_obj.show_grid()

# %%
grid_obj.get_diff_4th_reg_operator_dpsi()
Hy_dpsi_4th_reg, Hx_dpsi_4th_reg = grid_obj.Hy_dpsi_4th_reg, grid_obj.Hx_dpsi_4th_reg
np.savetxt('./Hy_dpsi_4th_reg.txt', Hy_dpsi_4th_reg, fmt='%.0f')
np.savetxt('./Hx_dpsi_4th_reg.txt', Hx_dpsi_4th_reg, fmt='%.0f')


Hy_dpsi_2nd_reg, Hx_dpsi_2nd_reg = grid_obj.Hy_dpsi_2nd_reg, grid_obj.Hx_dpsi_2nd_reg
np.savetxt('./Hy_dpsi_2nd_reg.txt', Hy_dpsi_2nd_reg, fmt='%.0f')
np.savetxt('./Hx_dpsi_2nd_reg.txt', Hx_dpsi_2nd_reg, fmt='%.0f')

# %%
