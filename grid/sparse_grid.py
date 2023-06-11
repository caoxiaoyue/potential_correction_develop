import numpy as np
import autolens as al 
from grid.util import *

class SparseDpsiGrid(object):
    def __init__(self, mask, dpix_data, shape_2d_dpsi=(30,30)):
        """
        This class represent the potential correction (Dpsi) grid,
        usually sparser than the native ccd image grid (or data grid).

        Parameters
        ----------
        mask: a bool array represents the data mask, which typically marks an annular-like region.
        dpix_data: the pixel size in arcsec for the native ccd image data.
        dpsi_shape_2d: the shape of the sparser potential correction grid before the mask
        """
        self.mask_data = mask
        self.mask_data = iter_clean_mask(self.mask_data)

        self.dpix_data = dpix_data
        self.shape_2d_dpsi = shape_2d_dpsi

        grid_data = al.Grid2D.uniform(shape_native=mask.shape, pixel_scales=dpix_data, sub_size=1)
        self.xgrid_data = np.array(grid_data.native[:,:,1])
        self.ygrid_data = np.array(grid_data.native[:,:,0])

        xmin, xmax = self.xgrid_data.min()-0.5*self.dpix_data, self.xgrid_data.max()+0.5*self.dpix_data
        ymin, ymax = self.ygrid_data.min()-0.5*self.dpix_data, self.ygrid_data.max()+0.5*self.dpix_data
        self.data_bound = [xmin, xmax, ymin, ymax]

        self.dpix_dpsi = float((xmax-xmin)/shape_2d_dpsi[0])
        grid_dpsi = al.Grid2D.uniform(shape_native=shape_2d_dpsi, pixel_scales=self.dpix_dpsi, sub_size=1)
        self.xgrid_dpsi = np.array(grid_dpsi.native[:,:,1])
        self.ygrid_dpsi = np.array(grid_dpsi.native[:,:,0])
        xmin, xmax = self.xgrid_dpsi.min()-0.5*self.dpix_dpsi, self.xgrid_dpsi.max()+0.5*self.dpix_dpsi
        ymin, ymax = self.ygrid_dpsi.min()-0.5*self.dpix_dpsi, self.ygrid_dpsi.max()+0.5*self.dpix_dpsi
        self.dpsi_bound = [xmin, xmax, ymin, ymax]

        self.grid_1d_from_mask()

        self.get_sparse_box_center()
        self.pair_data_dpsi_pixel()
        self.get_dpsi2data_mapping()

        self.get_gradient_operator_data()
        self.get_gradient_operator_dpsi()
        self.get_hamiltonian_operator_data()
        self.get_hamiltonian_operator_dpsi()

        # self.get_diff_4th_reg_operator_dpsi()
        # self.get_diff_2nd_reg_operator_dpsi()
        # self.get_gauss_reg_operator_dpsi()
        # self.get_exp_reg_operator_dpsi()


    def mask_dpsi_from_data(self):
        self.mask_dpsi = np.ones(self.shape_2d_dpsi).astype('bool')

        for i in range(self.shape_2d_dpsi[0]):
            for j in range(self.shape_2d_dpsi[1]):
                dist = (self.xgrid_data-self.xgrid_dpsi[i,j])**2 + (self.ygrid_data-self.ygrid_dpsi[i,j])**2
                dist = np.sqrt(dist)
                if_array_eq = np.isclose(dist, dist.min(), rtol=1e-05, atol=1e-08, equal_nan=False)
                min_indices = np.where(if_array_eq)
                if np.any((~self.mask_data)[min_indices]):
                    self.mask_dpsi[i,j] = False
        
        self.mask_dpsi = iter_clean_mask(self.mask_dpsi) #clean the dpsi_mask, also remove the exposed pixels


    def grid_1d_from_mask(self):
        """
        Get the 1d data/dpsi grid via the mask (self.xgrid_data_1d and self.xgrid_dpsi_1d)
        Also save the corresponding 1d-indices `indices_1d_data` and `indices_1d_dpsi`
        for example,
        self.data_xgrid_1d = self.data_xgrid.flatten()[self.indices_1d_data]
        """
        self.indices_1d_data = np.where((~self.mask_data).flatten())[0]
        self.xgrid_data_1d = self.xgrid_data.flatten()[self.indices_1d_data]
        self.ygrid_data_1d = self.ygrid_data.flatten()[self.indices_1d_data]

        self.mask_dpsi_from_data()
        self.indices_1d_dpsi = np.where((~self.mask_dpsi).flatten())[0]
        self.xgrid_dpsi_1d = self.xgrid_dpsi.flatten()[self.indices_1d_dpsi]
        self.ygrid_dpsi_1d = self.ygrid_dpsi.flatten()[self.indices_1d_dpsi]           


    def show_grid(self, output_file='grid.png'):
        plt.figure(figsize=(5,5))
        plt.plot(self.xgrid_data.flatten(), self.ygrid_data.flatten(), '*', color='black')
        plt.plot(self.xgrid_dpsi.flatten(), self.ygrid_dpsi.flatten(), '*', color='red')
        plt.plot(self.xgrid_data_1d, self.ygrid_data_1d, 'o', color='black')
        plt.plot(self.xgrid_dpsi_1d, self.ygrid_dpsi_1d, 'o', color='red')
        plt.plot(self.sparse_box_xcenter.flatten(), self.sparse_box_ycenter.flatten(), '+', color='blue')
        plt.plot(self.sparse_box_xcenter_1d, self.sparse_box_ycenter_1d, '+', color='red')
        plt.savefig(output_file)


    def get_sparse_box_center(self):
        n1, n2 = self.shape_2d_dpsi
        n1-=1
        n2-=1
        sparse_box_center = al.Grid2D.uniform(shape_native=(n2,n2), pixel_scales=self.dpix_dpsi, sub_size=1)
        self.sparse_box_xcenter = np.array(sparse_box_center.native[:,:,1]) #2d sparse box center x-grid
        self.sparse_box_ycenter = np.array(sparse_box_center.native[:,:,0]) #2d sparse box center y-grid

        self.mask_sparse_box = np.ones((n1, n2)).astype('bool')
        for i in range(n1):
            for j in range(n2):
                if (~self.mask_dpsi[i,j]) and (~self.mask_dpsi[i+1,j]) and (~self.mask_dpsi[i,j+1]) and ((~self.mask_dpsi[i+1,j+1])):
                    self.mask_sparse_box[i,j] = False

        self.indices_1d_sparse_box = np.where((~self.mask_sparse_box).flatten())[0]
        self.sparse_box_xcenter_1d = self.sparse_box_xcenter.flatten()[self.indices_1d_sparse_box]
        self.sparse_box_ycenter_1d = self.sparse_box_ycenter.flatten()[self.indices_1d_sparse_box] 


    def pair_data_dpsi_pixel(self):
        """
        pair the data grid to dpsi grid.
	    self.data_dpsi_pair_info: shape [n_unmasked_data_pixels, 2, 4], save the information how to interpolate `image` defined on 
        the coarser `dpsi grid` to finner `data grid`. 
        For exmaple:
	    self.data_dpsi_pair_info[0, 0, :]: the 1d indices of the paried (nearest) dpsi box for the first unmaksed data pixels.
	    self.data_dpsi_pair_info[0, 1, :]: the interpolation weight for each corner point of the box for the first unmaksed data pixels.
        """
        self.data_dpsi_pair_info = np.zeros((len(self.indices_1d_data), 2, 4))

        for count, item in enumerate(self.indices_1d_data):
            this_x_data = self.xgrid_data.flatten()[item]
            this_y_data = self.ygrid_data.flatten()[item]
            dist = np.sqrt((self.sparse_box_xcenter_1d-this_x_data)**2 + (self.sparse_box_ycenter_1d-this_y_data)**2)
            id_nearest_sparse_box = self.indices_1d_sparse_box[np.argmin(dist)] #this data pixel pairs with sparse box with 1d-index of `id_nearest_sparse_box`

            i,j = np.unravel_index(id_nearest_sparse_box, shape=self.mask_sparse_box.shape) #2d indices of nearest sparse box center
            #sparse_box_corners_x: [top-left,top-right, bottom-left, bottom-right] corner x-positions 
            sparse_box_corners_x = [self.xgrid_dpsi[i,j], self.xgrid_dpsi[i,j+1], self.xgrid_dpsi[i+1,j], self.xgrid_dpsi[i+1,j+1]] 
            sparse_box_corners_y = [self.ygrid_dpsi[i,j], self.ygrid_dpsi[i,j+1], self.ygrid_dpsi[i+1,j], self.ygrid_dpsi[i+1,j+1]]
            weight_sparse_box_corners = linear_weight_from_box(sparse_box_corners_x, sparse_box_corners_y, position=(this_y_data, this_x_data))

            indices_tmp = [j+self.shape_2d_dpsi[0]*i, j+1+self.shape_2d_dpsi[0]*i, j+self.shape_2d_dpsi[0]*(i+1), j+1+self.shape_2d_dpsi[0]*(i+1)]
            indices_of_indices_1d_dpsi = [np.where(self.indices_1d_dpsi == item)[0][0] for item in indices_tmp]
            indices_of_indices_1d_dpsi = np.array(indices_of_indices_1d_dpsi, dtype='int64')

            self.data_dpsi_pair_info[count, 0, :] = indices_of_indices_1d_dpsi[:]
            self.data_dpsi_pair_info[count, 1, :] = weight_sparse_box_corners[:]            
            

    def get_dpsi2data_mapping(self):
        """
        This function mapping a unmasked vector defined on coarser dpsi grid (shape: [n_unmasked_dpsi_pixels,]), 
        to a new unmasked vector defined on finner data grid (shape: [n_unmasked_data_pixels,]).

        return a matrix, with a shape of [n_unmasked_data_pixels, n_unmasked_dpsi_pixels]
        """
        self.interp_matrix = np.zeros((len(self.indices_1d_data), len(self.indices_1d_dpsi)))

        for id_data in range(len(self.indices_1d_data)):
            box_indices = (self.data_dpsi_pair_info[id_data, 0, :]).astype('int64')
            box_weights = (self.data_dpsi_pair_info[id_data, 1, :])
            self.interp_matrix[id_data, box_indices] = box_weights[:]


    def get_gradient_operator_data(self):
        self.Hy_data, self.Hx_data = diff_1st_operator_from_mask(self.mask_data, self.dpix_data)


    def get_gradient_operator_dpsi(self):
        self.Hy_dpsi, self.Hx_dpsi = diff_1st_operator_from_mask(self.mask_dpsi, self.dpix_dpsi)


    def get_diff_4th_reg_operator_dpsi(self):
        self.Hy_dpsi_4th_reg, self.Hx_dpsi_4th_reg = diff_4th_reg_nopad_operator_from_mask(self.mask_dpsi) #for reg matrix, set dpix to 1


    def get_diff_2nd_reg_operator_dpsi(self):
        self.Hy_dpsi_2nd_reg, self.Hx_dpsi_2nd_reg = diff_2nd_reg_nopad_operator_from_mask_koopman(self.mask_dpsi) #for reg matrix, set dpix to 1, to be consistent to suyu06 eq.A.3
        

    def get_gauss_reg_operator_dpsi(self, scale=1.0):
        C_mat = gauss_cov_matrix_from_mask(self.mask_dpsi, self.dpix_dpsi, scale)
        # C_mat = gauss_cov_matrix_from_mask(self.mask_dpsi, 1.0, 1.0)
        self.gauss_reg_dpsi = np.linalg.inv(C_mat)
        # assert np.allclose(np.matmul(C_mat, self.gauss_reg_dpsi), np.identity(len(C_mat)))


    def get_exp_reg_operator_dpsi(self, scale=1.0):
        C_mat = exp_cov_matrix_from_mask(self.mask_dpsi, self.dpix_dpsi, scale)
        # C_mat = exp_cov_matrix_from_mask(self.mask_dpsi, 1.0, 1.0)
        self.exp_reg_dpsi = np.linalg.inv(C_mat)
        # assert np.allclose(np.matmul(C_mat, self.exp_reg_dpsi), np.identity(len(C_mat)))
        

    def get_hamiltonian_operator_data(self):
        self.Hyy_data, self.Hxx_data = diff_2nd_operator_from_mask(self.mask_data, self.dpix_data)
        self.hamiltonian_data = self.Hxx_data + self.Hyy_data


    def get_hamiltonian_operator_dpsi(self):
        self.Hyy_dpsi, self.Hxx_dpsi = diff_2nd_operator_from_mask(self.mask_dpsi, self.dpix_dpsi)
        self.hamiltonian_dpsi = self.Hxx_dpsi + self.Hyy_dpsi

