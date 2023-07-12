import autolens as al
import numpy as np
from grid.sparse_grid import SparseDpsiGrid
import pixelized_mass
import pixelized_source
import potential_correction_util as pcu
import scipy.linalg as linalg
from scipy.spatial import Delaunay
from potential_correction_util import LinearNDInterpolatorExt
import copy
from plot import iter_solve as it_plot
import os
import logging
from astropy.io import fits
from scipy.optimize import differential_evolution
import time
import warnings


class IterativePotentialCorrect(object):
    def __init__(self, masked_imaging, shape_2d_dpsi=None, shape_2d_src=(50,50), sub_size=4):
        """
        shape_2d_dpsi: the shape of potential correction grid, if not set, this will be set to the lens image shape
        shape_2d_src: the number of grid used for source reconstruction (defined on image-plane)
        """
        self.masked_imaging = masked_imaging #include grid, mask, image, noise, psf etc

        self.image_data = self.masked_imaging.image.native #native image resolution, not the oversanpling one
        self.image_noise = self.masked_imaging.noise_map.native
        self.psf_kernel =  self.masked_imaging.psf.native
        image_mask = self.masked_imaging.mask 
        dpix_data = self.masked_imaging.pixel_scales[0]

        self.sub_size = sub_size

        if shape_2d_dpsi is None:
            shape_2d_dpsi = self.image_data.shape
        self.grid_obj = SparseDpsiGrid(image_mask, dpix_data, shape_2d_dpsi=shape_2d_dpsi) #Note, mask_data has not been cleaned

        image_mask = al.Mask2D(mask=self.grid_obj.mask_data, pixel_scales=(dpix_data, dpix_data))
        self.masked_imaging = self.masked_imaging.apply_mask(mask=image_mask) #since mask are cleaned, re-apply it to autolens imaging object
        self.masked_imaging = self.masked_imaging.apply_settings(
            settings=al.SettingsImaging(sub_size=self.sub_size, sub_size_inversion=self.sub_size)
        )

        self.shape_2d_src = shape_2d_src


    def initialize_iteration(
        self, 
        psi_2d_start=None, 
        niter=100, 
        s_reg_type='gradient',
        dpsi_reg_type='4th',
        psi_anchor_points=None,
        subhalo_fiducial_point=None,
        output_dir='./result'
    ):
        self._niter = niter
        self._s_reg_type = s_reg_type
        self._dpsi_reg_type = dpsi_reg_type
        self._psi_anchor_points = psi_anchor_points
        self._psi_2d_start = psi_2d_start
        self._psi_2d_start[self.masked_imaging.mask] = 0.0 #set the lens potential of masked pixels to 0
        self._output_dir = output_dir

        #do iteration-0, the macro model
        self.count_iter = 0 #count the iteration number
        self.pix_mass_this_iter = self.pixelized_mass_from(self._psi_2d_start) #initialize with lens potential given by macro model
        self.pix_mass_prev_iter = copy.deepcopy(self.pix_mass_this_iter) 
        #3---------pix src obj is mainly used for evalulating lens mapping matrix given a lens mass model
        self.pix_src_obj = pixelized_source.PixelizedSource(
            self.masked_imaging, 
            pixelization_shape_2d=self.shape_2d_src,
            reg_type=self._s_reg_type,
        )
        self.pix_src_obj.find_best_regularization(self.pix_mass_this_iter, log10_lam_range=[-4, 4], log10_scale_range=[-3, 3])
        self.lam_s_this_iter = self.pix_src_obj.mp_lam
        self.scale_s_this_iter = self.pix_src_obj.mp_scale
        #to begin the potential correction algorithm, we need a initial guess of source light
        self.pix_src_obj.source_inversion(
            self.pix_mass_this_iter, 
            lam_s=self.lam_s_this_iter,
            scale_s=self.scale_s_this_iter,
        )
        #Note: self.s_points_this_iter are given in autolens [(y1,x1),(y2,x2),...] order
        self._s_values_start = self.pix_src_obj.src_recontruct[:] #the intensity values of current best-fit pixelized source model
        self._s_points_start = np.copy(self.pix_src_obj.relocated_pixelization_grid) #the location of pixelized source grids (on source-plane).
        self._residual_start = self.image_data[~self.grid_obj.mask_data] - self.pix_src_obj.mapped_reconstructed_image
        self._lam_s_start = self.lam_s_this_iter
        self._scale_s_start = self.scale_s_this_iter
        self.s_values_this_iter = np.copy(self._s_values_start) #src intentity of current iteration
        self.s_points_this_iter = np.copy(self._s_points_start)
        self.residual_this_iter = np.copy(self._residual_start) #NOTE, this is the residual of pure source inversion given current best-fit mass model

        #Init other auxiliary info
        self.lam_dpsi_this_iter = None #init the regularization parameters for the potential correction
        self.scale_dpsi_this_iter = None
        self._psi_anchor_values = self.pix_mass_this_iter.eval_psi_at(self._psi_anchor_points)
        self._subhalo_fiducial_point = subhalo_fiducial_point
        self.pix_src_obj.inverse_covariance_matrix()
        self._inv_cov_matrix =  np.copy(self.pix_src_obj.inv_cov_mat) #inverse covariance matrix
        self._cov_matrix = np.linalg.inv(self._inv_cov_matrix) #covariance matrix
        self._ns = len(self.s_values_this_iter) #number source grids
        self._np = len(self.grid_obj.xgrid_dpsi_1d) #number dpsi grids
        self._d_1d = self.image_data[~self.grid_obj.mask_data] #1d unmasked image data
        self._n_1d = self.image_noise[~self.grid_obj.mask_data] #1d unmasked noise
        self._dof = len(self._d_1d) - self._np - self._ns #number degrees of freedom
        self._B_matrix = np.copy(self.pix_src_obj.psf_blur_matrix) #psf bluring matrix, see eq.7 in our document
        self._Cf_matrix = np.copy(
            self.grid_obj.interp_matrix
        ) #see the $C_f$ matrix in our document (eq.7), which interpolate data defined on coarser dpsi grid to native image grid
        self._Dpsi_matrix = pcu.fine_dpsi_gradient_operator_from(
            self._Cf_matrix,
            self.grid_obj.Hx_dpsi, 
            self.grid_obj.Hy_dpsi
        ) #the potential correction gradient operator, see the eq.8 in our document
        self._dpsi_grid_points = np.vstack([self.grid_obj.ygrid_dpsi_1d, self.grid_obj.xgrid_dpsi_1d]).T #points of sparse potential correction grid
        self._data_grid_points = np.vstack([self.grid_obj.ygrid_data_1d, self.grid_obj.xgrid_data_1d]).T

        #a list which save the potential correction map
        self._dpsi_map_coarse = [np.zeros_like(self.grid_obj.xgrid_dpsi)] #the potential correction map of iteration-0 is 0

        self._diag_info = [self.ret_diag_info()]

        #visualize iteration-0
        self.visualize_iteration(basedir=self._output_dir, iter_num=self.count_iter)

        #assign info of this iteration to the previous one
        self.update_iterations()


    def pixelized_mass_from(self, psi_2d):
        pix_mass_obj = pixelized_mass.PixelizedMass(
            xgrid=self.grid_obj.xgrid_data, 
            ygrid=self.grid_obj.ygrid_data, 
            psi_map=psi_2d, 
            mask=self.grid_obj.mask_data, 
            Hx=self.grid_obj.Hx_data, 
            Hy=self.grid_obj.Hy_data,
            Hxx=self.grid_obj.Hxx_data, 
            Hyy=self.grid_obj.Hyy_data,
        ) 
        return pix_mass_obj


    def update_iterations(self):
        self.count_iter += 1
        #this iteration becomes previous iteration
        self.pix_mass_prev_iter = copy.copy(self.pix_mass_this_iter)
        self.s_values_prev_iter = np.copy(self.s_values_this_iter)
        self.s_points_prev_iter = np.copy(self.s_points_this_iter)
        self.lam_s_prev_iter = self.lam_s_this_iter
        self.scale_s_prev_iter = self.scale_s_this_iter
        self.lam_dpsi_prev_iter = self.lam_dpsi_this_iter
        self.scale_dpsi_prev_iter = self.scale_dpsi_this_iter

        #erase information of this iteration 
        self.lam_s_this_iter = None
        self.scale_s_this_iter = None
        self.lam_dpsi_this_iter = None
        self.scale_dpsi_this_iter = None
        self.pix_mass_this_iter = None
        self.s_values_this_iter = None
        self.s_points_this_iter = None


    def Ds_mat_from(self, pix_mass_obj, source_points, source_values):
        alpha_data_yx = pix_mass_obj.eval_alpha_yx_at(self._data_grid_points) #use previously found pix_mass_object to ray-tracing
        alpha_data_yx = np.asarray(alpha_data_yx).T
        src_plane_data_yx = self._data_grid_points - alpha_data_yx #the location of dpsi grid on the source-plane
        source_gradient = pcu.source_gradient_from(
            source_points, #previously found best-fit src pixlization grids
            source_values, #previously found best-fit src reconstruction
            src_plane_data_yx, 
            cross_size=0.01, #TODO, better way to calculate the gradient? cross-size?
        )
        return pcu.source_gradient_matrix_from(source_gradient)  


    def M_mat_from(self, pix_mass_obj, s_points, s_values): 
        #get the linear map matrix, based on previous best fit mass model `pix_mass_obj` and src model `s_points and s_values`
        self.pix_src_obj.build_lens_mapping(pix_mass_obj) 
        L_matrix = np.copy(self.pix_src_obj.mapping_matrix) #rely on pix_src_obj.build_lens_mapping

        Ds_matrix = self.Ds_mat_from(
            pix_mass_obj, 
            s_points, 
            s_values,
        )
        DSD_matrix = -1.0*np.matmul(
            Ds_matrix,
            self._Dpsi_matrix,
        )
        Lc_matrix = np.hstack([L_matrix, DSD_matrix]) #see eq.14 in our document
        Mc_matrix = np.matmul(self._B_matrix, Lc_matrix)

        return Mc_matrix

    def src_reg_mat_from(self, lam_s, scale_s):
        self.pix_src_obj.build_reg_matrix(lam_s=lam_s, scale_s=scale_s) 
        return np.copy(self.pix_src_obj.regularization_matrix)
    

    def dpsi_reg_mat_from(self, lam_dpsi, scale_dpsi):
        if self._dpsi_reg_type == '2nd':
            if not hasattr(self, '_HTH_dpsi_cache'):
                self.grid_obj.get_diff_2nd_reg_operator_dpsi()
                self._HTH_dpsi_cache = np.matmul(self.grid_obj.Hx_dpsi_2nd_reg.T, self.grid_obj.Hx_dpsi_2nd_reg) + \
                    np.matmul(self.grid_obj.Hy_dpsi_2nd_reg.T, self.grid_obj.Hy_dpsi_2nd_reg)
            return np.copy(lam_dpsi * self._HTH_dpsi_cache) #Note, not lam_dpsi**2
        elif self._dpsi_reg_type == '4th':
            if not hasattr(self, '_HTH_dpsi_cache'):
                self.grid_obj.get_diff_4th_reg_operator_dpsi()
                self._HTH_dpsi_cache = np.matmul(self.grid_obj.Hx_dpsi_4th_reg.T, self.grid_obj.Hx_dpsi_4th_reg) + \
                    np.matmul(self.grid_obj.Hy_dpsi_4th_reg.T, self.grid_obj.Hy_dpsi_4th_reg)
            return np.copy(lam_dpsi * self._HTH_dpsi_cache) #Note, not lam_dpsi**2
        elif (self._dpsi_reg_type == 'vkl_exp') or (self._dpsi_reg_type == 'vkl_gauss'):
            RTR_mat = al.util.regularization.regularization_matrix_vkl_from(
                scale_dpsi, 
                lam_dpsi, 
                self._dpsi_grid_points, 
                self._dpsi_reg_type,
            )
            return RTR_mat
        else:
            raise Exception(f"Not supported regularization type: {self._dpsi_reg_type}")


    def reg_mat_from(self, lam_s, scale_s, lam_dpsi, scale_dpsi):
        #see eq.21 in our document, the regularization matrix for both source and lens potential corrections.
        RTR_matrix = np.zeros((self._ns+self._np, self._ns+self._np), dtype='float')

        #src reg matrix depend on the lens mass model (via the `mapper`)
        RTR_matrix[0:self._ns, 0:self._ns] = self.src_reg_mat_from(lam_s, scale_s)
        #dpsi reg matrix
        RTR_matrix[self._ns:, self._ns:] = self.dpsi_reg_mat_from(lam_dpsi, scale_dpsi)

        return RTR_matrix    


    def data_vector_from(self, M_mat):
        #see the right hand side of eq.20 in our document
        data_vector = np.matmul(
            np.matmul(M_mat.T, self._inv_cov_matrix),
            self._d_1d,
        )
        return data_vector


    def linear_inversion(self, lam_s, scale_s, lam_dpsi, scale_dpsi, cal_M_mat=True):
        if cal_M_mat:
            self.M_mat = self.M_mat_from(self.pix_mass_prev_iter, self.s_points_prev_iter, self.s_values_prev_iter)
        #the construction of RTR_mat implicitly use the mass mapping info of M_mat (or self.pix_mass_prev_iter)
        self.reg_mat = self.reg_mat_from(lam_s, scale_s, lam_dpsi, scale_dpsi) #src reg mat rely on the mass mapping when calculating self.M_mat

        data_vec = self.data_vector_from(self.M_mat)

        self.curve_mat = np.matmul(
            np.matmul(self.M_mat.T, self._inv_cov_matrix),
            self.M_mat,
        )
        self.curve_reg_mat = self.curve_mat + self.reg_mat

        #solve for the next source and potential corrections
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.r_vec = linalg.solve(self.curve_reg_mat, data_vec)


    def info_from_inversion(self):
        #extract source
        self.s_values_this_iter = self.r_vec[0:self._ns]
        self.s_points_this_iter = np.copy(self.pix_src_obj.relocated_pixelization_grid) #this src pixelization given by prev best mass model

        #extract potential correction
        dpsi_2d = np.zeros_like(self._psi_2d_start, dtype='float')
        dpsi_2d[~self.grid_obj.mask_data] = np.matmul(
            self._Cf_matrix, 
            self.r_vec[self._ns:]
        )
        #update lens potential with potential correction at this iteration
        psi_2d_this_iter = self.pix_mass_prev_iter.psi_map + dpsi_2d #the new 2d lens potential map

        #rescale the current lens potential, to avoid various degeneracy problems. (see sec.2.3 in our document);
        psi_2d_this_iter, factor = self.rescale_lens_potential(psi_2d_this_iter)
        #save the coarse potential correction map
        dpsi_map_coarse = np.zeros_like(self.grid_obj.xgrid_dpsi)
        dpsi_map_coarse[~self.grid_obj.mask_dpsi] = self.r_vec[self._ns:]
        dpsi_map_coarse = dpsi_map_coarse + factor[0]*self.grid_obj.ygrid_dpsi + factor[1]*self.grid_obj.xgrid_dpsi + factor[2]
        self._dpsi_map_coarse.append(dpsi_map_coarse)
        #get pixelized mass object of this iteration
        self.pix_mass_this_iter = self.pixelized_mass_from(psi_2d_this_iter)


    def evidence(self):
        #noise normalization term
        self.noise_term= float(np.sum(np.log(2 * np.pi * self.masked_imaging.noise_map ** 2.0))) * (-0.5)

        #log det cuverd reg term
        sign, logval = np.linalg.slogdet(self.curve_reg_mat)
        if sign != 1:
            raise Exception(f"The curve reg matrix is not positive definite.")
        self.log_det_curve_reg_term = logval * (-0.5)

        #log det reg matrix term
        sign, logval = np.linalg.slogdet(self.reg_mat)
        if sign != 1:
            raise Exception(f"The regularization matrix is not positive definite.")
        self.log_det_reg_term = logval * 0.5

        #reg r term
        reg_r_term = np.matmul(
            self.r_vec.T, 
            np.matmul(self.reg_mat, self.r_vec),
        )
        self.reg_qf_term = reg_r_term * (-0.5)

        #chi2 term
        mapped_reconstructed_image_1d = np.matmul(self.M_mat, self.r_vec)
        residual_1d = (mapped_reconstructed_image_1d - self._d_1d)
        norm_residual_1d = residual_1d / self._n_1d
        self.chi2_term = np.sum(norm_residual_1d**2) * (-0.5)

        #evidence
        self.evidence_term = self.noise_term + self.log_det_curve_reg_term + self.log_det_reg_term + self.reg_qf_term + self.chi2_term
        return self.evidence_term 
        

    def reg_merit(self, X):
        self.linear_inversion(lam_s=10**(X[0]), scale_s=10**(X[1]), lam_dpsi=10**(X[2]), scale_dpsi=10**(X[3]), cal_M_mat=False)
        # self.info_from_inversion()
        return -1.0*self.evidence()


    def callback_reg_merit(self, xk, convergence):
        self.history_params.append(xk)
        self.history_merit.append(self.reg_merit(xk))


    def find_best_regularization(
        self, 
        log_lam_s_range=[-8, 8], 
        log_scale_s_range=[-3, 3],
        log_lam_psi_range=[-8, 8],
        log_scale_psi_range=[-3, 3],
        ):
        '''
        return the best-fit regularization strength given a ``fixed'' mass model (pixelized_mass_obj)
        log10_lam_range: set the log range of regularization strength. default: 10^-5 to 10^4 
        '''
        self.history_params = []
        self.history_merit = []
        t0 = time.time()
        self.best_fit_reg_info = differential_evolution(
            self.reg_merit, 
            bounds=[log_lam_s_range, log_scale_s_range, log_lam_psi_range, log_scale_psi_range],
            callback=self.callback_reg_merit,
        )
        t1 = time.time()
        print(f'time elapse of it-{self.count_iter} is: {t1-t0}')
        self.mp_lam_s = 10**(self.best_fit_reg_info['x'][0]) #this regularization strength maximize the posterior
        self.mp_scale_s = 10**(self.best_fit_reg_info['x'][1]) #this regularization strength maximize the posterior
        self.mp_lam_dpsi = 10**(self.best_fit_reg_info['x'][2]) #this regularization strength maximize the posterior
        self.mp_scale_dpsi = 10**(self.best_fit_reg_info['x'][3]) #this regularization strength maximize the posterior
        self.mp_ev = -1.0*self.best_fit_reg_info['fun'] #this is the corrpesonding evidence values


    def run_this_iter(
        self,
        log_lam_s_range=[-5, 8], 
        log_scale_s_range=[-4, 4],
        log_lam_psi_range=[-5, 15],
        log_scale_psi_range=[-4, 4],
        ):
        #find the best reg scale and strength by maximizing the evidence
        # only works for exp or gaussian form for both src and dpsi, need refactor; TODO
        self.M_mat = self.M_mat_from(self.pix_mass_prev_iter, self.s_points_prev_iter, self.s_values_prev_iter)
        self.find_best_regularization(
                log_lam_s_range, 
                log_scale_s_range,
                log_lam_psi_range,
                log_scale_psi_range,
        )
        self.lam_s_this_iter = self.mp_lam_s
        self.scale_s_this_iter = self.mp_scale_s
        self.lam_dpsi_this_iter = self.mp_lam_dpsi
        self.scale_dpsi_this_iter = self.mp_scale_dpsi
        self.linear_inversion(
            self.lam_s_this_iter, 
            self.scale_s_this_iter, 
            self.lam_dpsi_this_iter, 
            self.scale_dpsi_this_iter,
            cal_M_mat=False,
        )
        self.info_from_inversion()
        print(f'evidence value of this iteration-{self.count_iter}: {self.evidence()}')

        # NOTE, the current best source regularization coupled with the previous best-fit mass model
        # calculate the image residual of a pure source inversion
        self.pix_src_obj.source_inversion( #can move out of here to improve speed
            self.pix_mass_prev_iter, 
            lam_s=self.lam_s_this_iter,
            scale_s=self.scale_s_this_iter,
        )
        self.residual_this_iter = copy.deepcopy(self.pix_src_obj.residual_map)

        #get diagnostics info for this iteration
        self._diag_info.append(self.ret_diag_info())

        #do visualization
        self.visualize_iteration(basedir=self._output_dir, iter_num=self.count_iter)

        if self.has_converged():
            return True
        else:
            # if not converge, keep updating 
            self.update_iterations() #current iteration info become previous one
            return False 


    def has_converged(self):
        relative_change = (self._diag_info[-2]['image_rchi2'] - self._diag_info[-1]['image_rchi2'])/self._diag_info[-2]['image_rchi2']
        print('previous VS current image_rchi2:', self._diag_info[-2]['image_rchi2'], self._diag_info[-1]['image_rchi2'], relative_change)

        if (abs(relative_change) < 1e-8) and (self.count_iter>1):
            return True
        else:
            return False 
        # return False


    def run_iter_solve(
        self, 
        log_lam_s_range=[-5, 8], 
        log_scale_s_range=[-4, 4],
        log_lam_psi_range=[-5, 15],
        log_scale_psi_range=[-4, 4],
        ):
        for ii in range(1, self._niter):
            condition = self.run_this_iter(
                log_lam_s_range, 
                log_scale_s_range,
                log_lam_psi_range,
                log_scale_psi_range,
            )
            if condition:
                print('------','code converge')
                break
            else:
                print('------', ii, self.count_iter)  

        
    def visualize_iteration(self, basedir='./result', iter_num=0):
        abs_path = os.path.abspath(basedir)  #get absolute path
        if not os.path.exists(basedir):  #check if path exist
            os.makedirs(abs_path) #create new directory recursively

        it_plot.visualize_correction(
            self, 
            basedir=basedir, 
            iter_num=iter_num,
        )


    def ret_diag_info(self):
        info_dict = {}
        image_chi2 = (self.residual_this_iter/self._n_1d)**2
        info_dict['image_chi2'] = float(np.sum(image_chi2))
        info_dict['image_rchi2'] = float(info_dict['image_chi2']/self._dof)

        try:
            this_reg_mat = self.reg_mat[0:self._ns, 0:self._ns]
            s_quadratic = np.matmul(
                self.s_values_this_iter.T, 
                np.matmul(
                    this_reg_mat, 
                    self.s_values_this_iter
                )
            )
        except Exception as e:
            print(e)
            s_quadratic = 0
        finally:
            info_dict['s_quadratic'] = float(s_quadratic)

        try:
            this_reg_mat = self.reg_mat[self._ns:, self._ns:]
            dpsi_1d = self._dpsi_map_coarse[-1][~self.grid_obj.mask_dpsi]
            dpsi_quadratic = np.matmul(
                dpsi_1d.T, 
                np.matmul(
                    this_reg_mat, 
                    dpsi_1d
                )
            )
        except Exception as e:
            print(e)
            dpsi_quadratic = 0
        finally:
            info_dict['dpsi_quadratic'] = float(dpsi_quadratic)

        info_dict['total_penalty'] = info_dict['image_chi2'] + info_dict['s_quadratic'] + info_dict['dpsi_quadratic']
        return info_dict


    def rescale_lens_potential(self, psi_2d_in):
        if not hasattr(self, 'tri_psi_interp'):
            self.tri_psi_interp = Delaunay(
                list(zip(self.grid_obj.xgrid_data_1d, self.grid_obj.ygrid_data_1d))
            )
        psi_interpolator = LinearNDInterpolatorExt(self.tri_psi_interp, psi_2d_in[~self.grid_obj.mask_data])
        
        psi_anchor_values_new = psi_interpolator(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0])
        psi_2d_out, factor = pcu.rescale_psi_map(
            self._psi_anchor_values, 
            self._psi_anchor_points, 
            psi_anchor_values_new, 
            psi_2d_in, 
            self.grid_obj.xgrid_data, 
            self.grid_obj.ygrid_data,
            return_rescale_factor=True,
        )
        psi_2d_out[self.grid_obj.mask_data] = 0.0 #always set lens potential values at masked region to 0.0

        return psi_2d_out, factor