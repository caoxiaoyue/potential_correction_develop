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
from iterative_solve import IterativePotentialCorrect
import time
from scipy.optimize import differential_evolution


class IterativePotentialCorrectFixSource(IterativePotentialCorrect):
    def __init__(
        self, 
        masked_imaging, 
        shape_2d_dpsi=None, 
        shape_2d_src=(50,50),
        sub_size=None,
        fix_source_points=None,
        fix_source_values=None,
    ):
        super().__init__(
            masked_imaging, 
            shape_2d_dpsi=shape_2d_dpsi, 
            shape_2d_src=shape_2d_src,
            sub_size=sub_size,
        )

        self._s_points_0 = fix_source_points #the source are input as a set of pixelized values; 2d array like [(y1,x1), (y2,x2), (y3,x3), ...]
        self._s_values_0 = fix_source_values #1d array save the brightness of the source
        self._fix_src_model = LinearNDInterpolatorExt(self._s_points_0[:,::-1], self._s_values_0)
        self._B_matrix = self.masked_imaging.convolver.get_psf_blur_matrix()

        noise_1d = self.image_noise[~self.masked_imaging.mask]
        npix = len(noise_1d)
        self._inv_cov_matrix = np.zeros((npix, npix))
        for ii in range(npix):
            self._inv_cov_matrix[ii, ii] = 1.0/(noise_1d[ii])**2

    
    def forward_source_model(
        self,
        pix_mass_obj,
        ):
        alpha_data_yx = pix_mass_obj.eval_alpha_yx_at(self._data_grid_points) 
        alpha_data_yx = np.asarray(alpha_data_yx).T
        self.src_plane_data_yx = self._data_grid_points - alpha_data_yx #the location of dpsi grid on the source-plane

        lensed_src_1d_data = self._fix_src_model(self.src_plane_data_yx[:,1], self.src_plane_data_yx[:,0])
        # #bin the 1d data ##removed, since _B_matrix is a blur+bin operator
        # tmp = lensed_src_1d_data.reshape(self.sub_size, lensed_src_1d_data.shape[0]//self.sub_size)
        # bin_lensed_src_1d_data = tmp.mean(axis=0)

        blur_lensed_src_1d_data = np.matmul(self._B_matrix, lensed_src_1d_data)
        blur_lensed_src_2d_data = np.zeros_like(self.image_data, dtype='float')
        blur_lensed_src_2d_data[~self.grid_obj.mask_data] = blur_lensed_src_1d_data
        
        return blur_lensed_src_2d_data


    def initialize_iteration(
        self, 
        psi_2d_start=None, 
        niter=100, 
        lam_dpsi_start=1e9,
        scale_dpsi_start=None,
        dpsi_reg_type='4th',
        psi_anchor_points=None,
        subhalo_fiducial_point=None,
        output_dir='./result'
    ):
        """
        psi_2d_0: the lens potential map of the initial start mass model, typicall given by a macro model like elliptical power law model.
        niter: the upper limit of the number of the potential correctio iterations
        lam_s_0: the initial regularization strength of pixelized sources. 
        lam_dpsi_0: the initial regularization strength of potential correction (dpsi)
        dpsi_reg_type: the regularization type of dpsi
        psi_anchor_points: the anchor points of lens potential. we require the lens potential values at those anchor point
        remain unchanged during potential corrention, to avoid various degeneracy problems. (see sec.2.3 in our document);
        dpsi_anchor_points has the following form: [(y1,x1), (y2,x2), (y3,x3)]
        check_converge_points: the points where we check convergence, in autolens [(y1,x1), (y2,x2), (y3,x3), ...] order
        subhalo_fiducial_point: the fiducial location of subhalo, (y_sub, x_sub), mainly for mock test purpose.
        """
        self._niter = niter
        self._lam_dpsi_start = lam_dpsi_start
        self._scale_dpsi_start = scale_dpsi_start
        self._psi_anchor_points = psi_anchor_points
        self._psi_2d_start = psi_2d_start
        self._psi_2d_start[self.masked_imaging.mask] = 0.0 #set the lens potential of masked pixels to 0
        self._dpsi_reg_type = dpsi_reg_type
        self._output_dir = output_dir

        #do iteration-0, the macro model
        self.count_iter = 0 #count the iteration number
        #1--------regularization of source and lens potential of this iteration
        self.lam_dpsi_this_iter = self._lam_dpsi_start #potential correction reg strength
        self.scale_dpsi_this_iter = self._scale_dpsi_start #potential correction reg scale
        #2--------the lens mass model of currect iteration
        self.pix_mass_this_iter = self.pixelized_mass_from(self._psi_2d_start) #initialize with lens potential given by macro model

        self._dpsi_grid_points = np.vstack([self.grid_obj.ygrid_dpsi_1d, self.grid_obj.xgrid_dpsi_1d]).T #points of sparse potential correction grid
        self._data_grid_points = np.vstack([self.grid_obj.ygrid_data_1d, self.grid_obj.xgrid_data_1d]).T
        
        #Note: self.s_points_this_iter are given in autolens [(y1,x1),(y2,x2),...] order
        self._image_model_start = self.forward_source_model(self.pix_mass_this_iter)
        self._residual_start = (self.image_data - self._image_model_start)[~self.grid_obj.mask_data]
        self.image_model_this_iter = np.copy(self._image_model_start)
        self.residual_this_iter = np.copy(self._residual_start)

        #Init other auxiliary info
        self._psi_anchor_values = self.pix_mass_this_iter.eval_psi_at(self._psi_anchor_points)
        self._subhalo_fiducial_point = subhalo_fiducial_point
        self._np = len(self.grid_obj.xgrid_dpsi_1d) #number dpsi grids
        self._d_1d = self.image_data[~self.grid_obj.mask_data] #1d unmasked image data
        self._n_1d = self.image_noise[~self.grid_obj.mask_data] #1d unmasked noise
        self._dof = len(self._d_1d) - self._np
        self._Cf_matrix = np.copy(
            self.grid_obj.interp_matrix
        ) #see the $C_f$ matrix in our document (eq.7), which interpolate data defined on coarser dpsi grid to native image grid
        self._Dpsi_matrix = pcu.fine_dpsi_gradient_operator_from(
            self._Cf_matrix,
            self.grid_obj.Hx_dpsi, 
            self.grid_obj.Hy_dpsi
        ) #the potential correction gradient operator, see the eq.8 in our document

        #a list which save the potential correction map
        self._dpsi_map_coarse = [np.zeros_like(self.grid_obj.xgrid_dpsi)] #the potential correction map of iteration-0 is 0
        
        #calculate the total_penalty of initial macro model. see eq.16 in our document
        self.RTR_mat_from(self.lam_dpsi_this_iter, self.scale_dpsi_this_iter) 
        # self.RTR_mat = np.copy(self.lam_dpsi_this_iter * self._HTH_dpsi) #need init this to cal the dig_info
        self._diag_info = [self.ret_diag_info()]

        #visualize iteration-0
        self.visualize_iteration(basedir=self._output_dir, iter_num=self.count_iter) 

        #assign info of this iteration to the previous one
        self.update_iterations()


    def ret_diag_info(self):
        info_dict = {}
        image_chi2 = (self.residual_this_iter/self._n_1d)**2
        info_dict['image_chi2'] = np.sum(image_chi2)
        info_dict['image_rchi2'] = info_dict['image_chi2']/self._dof

        dpsi_1d = self._dpsi_map_coarse[-1][~self.grid_obj.mask_dpsi]
        dpsi_penalty = np.matmul(
            dpsi_1d.T, 
            np.matmul(
                self.RTR_mat, 
                dpsi_1d
            )
        )
        info_dict['dpsi_penalty'] = float(dpsi_penalty)
        info_dict['total_penalty'] = info_dict['image_chi2'] + info_dict['dpsi_penalty']
        return info_dict


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


    def update_lam_dpsi(self): ##TODO
        """
        update the regularization strength of potential correction with iterations
        """
        self.lam_dpsi_this_iter = self.lam_dpsi_prev_iter * 1.0 #* 0.1


    def update_iterations(self):
        self.count_iter += 1
        #this iteration becomes previous iteration
        self.lam_dpsi_prev_iter = self.lam_dpsi_this_iter
        self.pix_mass_prev_iter = copy.deepcopy(self.pix_mass_this_iter)
        self.residual_prev_iter = copy.deepcopy(self.residual_this_iter)

        #erase information of this iteration 
        self.lam_dpsi_this_iter = None
        self.pix_mass_this_iter = None
        self.residual_this_iter = None


    def Ds_matrix_from(self, pix_mass_obj):
        alpha_data_yx = pix_mass_obj.eval_alpha_yx_at(self._data_grid_points) #use previously found pix_mass_object to ray-tracing
        alpha_data_yx = np.asarray(alpha_data_yx).T
        src_plane_data_yx = self._data_grid_points - alpha_data_yx #the location of dpsi grid on the source-plane
        source_gradient = pcu.source_gradient_from(
            self._s_points_0, #previously found best-fit src pixlization grids
            self._s_values_0, #previously found best-fit src reconstruction
            src_plane_data_yx, 
            cross_size=0.01, #TODO, better way to calculate the gradient? cross-size?
        )
        return pcu.source_gradient_matrix_from(source_gradient)  

    
    def DSD_matrix_from(self, pix_mass_obj):
        Ds_mat = self.Ds_matrix_from(pix_mass_obj)
        DSD_mat = -1.0*np.matmul(
            Ds_mat,
            self._Dpsi_matrix,
        )
        return DSD_mat

    
    def linear_mapping_mat_from(self, pix_mass_obj):
        DSD_mat = self.DSD_matrix_from(pix_mass_obj)
        M_mat = np.matmul(self._B_matrix, DSD_mat)
        return M_mat 

    
    def M_mat_from(self, pix_mass_obj):
        DSD_mat = self.DSD_matrix_from(pix_mass_obj)
        self.M_mat = np.matmul(self._B_matrix, DSD_mat)


    def RTR_mat_from(self, lam_dpsi, scale_dpsi):
        if self._dpsi_reg_type == '2nd':
            if not hasattr(self, '_HTH_dpsi_cache'):
                self.grid_obj.get_diff_2nd_reg_operator_dpsi()
                self._HTH_dpsi_cache = np.matmul(self.grid_obj.Hx_dpsi_2nd_reg.T, self.grid_obj.Hx_dpsi_2nd_reg) + \
                    np.matmul(self.grid_obj.Hy_dpsi_2nd_reg.T, self.grid_obj.Hy_dpsi_2nd_reg)
            self.RTR_mat = np.copy(lam_dpsi * self._HTH_dpsi_cache)
        elif self._dpsi_reg_type == '4th':
            if not hasattr(self, '_HTH_dpsi_cache'):
                self.grid_obj.get_diff_4th_reg_operator_dpsi()
                self._HTH_dpsi_cache = np.matmul(self.grid_obj.Hx_dpsi_4th_reg.T, self.grid_obj.Hx_dpsi_4th_reg) + \
                    np.matmul(self.grid_obj.Hy_dpsi_4th_reg.T, self.grid_obj.Hy_dpsi_4th_reg)
            self.RTR_mat = np.copy(lam_dpsi * self._HTH_dpsi_cache)
        elif (self._dpsi_reg_type == 'vkl_exp') or (self._dpsi_reg_type == 'vkl_gauss'):
            self.RTR_mat = al.util.regularization.regularization_matrix_vkl_from(
                scale_dpsi, 
                lam_dpsi, 
                self._dpsi_grid_points, 
                self._dpsi_reg_type,
            )
        else:
            raise Exception(f"Not supported regularization type: {self._dpsi_reg_type}")


    def data_vec_from(self, M_mat):
        data_vector = np.matmul(
            np.matmul(M_mat.T, self._inv_cov_matrix),
            self.residual_prev_iter,
        )
        return data_vector

    
    def linear_inversion(self, lam_dpsi, scale_dpsi, cal_M_mat=True):
        if cal_M_mat:
            self.M_mat_from(self.pix_mass_prev_iter)
        self.RTR_mat_from(lam_dpsi, scale_dpsi) 
        self.data_vec = self.data_vec_from(self.M_mat)
        
        #solve the next source and potential corrections
        self.curve_term = np.matmul(
            np.matmul(self.M_mat.T, self._inv_cov_matrix),
            self.M_mat,
        )
        self.curve_reg_term = self.curve_term + self.RTR_mat
        # print('~~~~~~~~~~~~~~~~iteration-{}, r-condition number {:.5e}'.format(self.count_iter, 1/np.linalg.cond(self.curve_reg_term)))
        self.r_vec = linalg.solve(self.curve_reg_term, self.data_vec)
    

    def extract_info_from_linear_inversion(self):
        #extract infor of this iteration
        dpsi_2d = np.zeros_like(self._psi_2d_start, dtype='float')
        dpsi_2d[~self.grid_obj.mask_data] = np.matmul(
            self._Cf_matrix, 
            self.r_vec
        )
        #update lens potential with potential correction at this iteration
        psi_2d_this_iter = self.pix_mass_prev_iter.psi_map + dpsi_2d #the new 2d lens potential map
        # psi_2d_this_iter = self.pix_mass_prev_iter.psi_map + 1.2*self.grid_obj.xgrid_data + 0.5*self.grid_obj.ygrid_data + 2.0 #for testing
        #rescale the current lens potential, to avoid various degeneracy problems. (see sec.2.3 in our document);
        psi_2d_this_iter, factor = self.rescale_lens_potential(psi_2d_this_iter)
        #save the coarse potential correction map
        dpsi_map_coarse = np.zeros_like(self.grid_obj.xgrid_dpsi)
        dpsi_map_coarse[~self.grid_obj.mask_dpsi] = self.r_vec
        dpsi_map_coarse = dpsi_map_coarse + factor[0]*self.grid_obj.ygrid_dpsi + factor[1]*self.grid_obj.xgrid_dpsi + factor[2]

        pix_mass_this_iter = self.pixelized_mass_from(psi_2d_this_iter)
        image_model_this_iter = self.forward_source_model(pix_mass_this_iter)
        residual_this_iter = (self.image_data - image_model_this_iter)[~self.grid_obj.mask_data]

        return dpsi_map_coarse, pix_mass_this_iter, image_model_this_iter, residual_this_iter
        # self._dpsi_map_coarse.append(dpsi_map_coarse)
        # #get pixelized mass object of this iteration
        # self.pix_mass_this_iter = self.pixelized_mass_from(psi_2d_this_iter)
        # #update the residual for next iteration
        # self.image_model_this_iter = self.forward_source_model(self.pix_mass_this_iter)
        # self.residual_this_iter = (self.image_data - self.image_model_this_iter)[~self.grid_obj.mask_data]


    def evidence_from_reconstruction(self, dpsi_1d):
        sign, logval = np.linalg.slogdet(self.curve_reg_term) 
        self.log_curve_reg_mat_term = sign*logval

        sign, logval = np.linalg.slogdet(self.RTR_mat)
        self.log_reg_mat_term = sign*logval

        self.reg_term = np.matmul(
            dpsi_1d.T, 
            np.matmul(
                self.RTR_mat, 
                dpsi_1d
            )
        )

        self.noise_norm = float(np.sum(np.log(2 * np.pi * self.masked_imaging.noise_map ** 2.0)))

        ##NOTE prev_mass model should no clash when fit dpsi reg par
        ## enusre no updating_iteration when optimize dpsi reg par
        self.pesudo_residual = (self.residual_prev_iter - np.matmul(self.M_mat,  self.r_vec))/self._n_1d
        self.pesudo_chi2_term = np.sum(self.pesudo_residual**2)

        self.log_evidence = float(
            -0.5
            * (
                self.pesudo_chi2_term 
                + self.reg_term 
                + self.log_curve_reg_mat_term
                - self.log_reg_mat_term
                + self.noise_norm
            )
        )

        print('-----------------------------')
        print(self.pesudo_chi2_term, self.reg_term, self.log_curve_reg_mat_term, self.log_reg_mat_term, self.noise_norm)
        print('-----------------------------')

        return self.log_evidence


    def regularization_merit(self, X):
        self.linear_inversion(lam_dpsi=10**(X[0]), scale_dpsi=10**(X[1]), cal_M_mat=False)
        dpsi_map_coarse, pix_mass_this_iter, image_model_this_iter, residual_this_iter = self.extract_info_from_linear_inversion()
        dpsi_1d = dpsi_map_coarse[~self.grid_obj.mask_dpsi]
        return -1.0*self.evidence_from_reconstruction(dpsi_1d)


    def find_best_regularization(self, log10_lam_range=[-8, 8], log10_scale_range=[-3, 3]):
        '''
        return the best-fit regularization strength given a ``fixed'' mass model (pixelized_mass_obj)
        log10_lam_range: set the log range of regularization strength. default: 10^-5 to 10^4 
        '''
        self.M_mat_from(self.pix_mass_prev_iter)
        t0 = time.time()
        self.best_fit_reg_info = differential_evolution(self.regularization_merit, bounds=[log10_lam_range, log10_scale_range])
        self.mp_scale = 10**(self.best_fit_reg_info['x'][1]) #this regularization strength maximize the posterior
        t1 = time.time()
        self.mp_lam = 10**(self.best_fit_reg_info['x'][0]) #this regularization strength maximize the posterior
        self.mp_ev = -1.0*self.best_fit_reg_info['fun'] #this is the corrpesonding evidence values
        print(f'time elapse of it-{self.count_iter} is: {t1-t0}')


    def run_this_iter(self):
        if (self._dpsi_reg_type == '2nd') or (self._dpsi_reg_type == '4th'):
            self.update_lam_dpsi() 
            self.linear_inversion(self.lam_dpsi_this_iter, self.scale_dpsi_this_iter)
        else:
            #find the best reg scale and strength by maximizing the evidence
            self.find_best_regularization(log10_lam_range=[-8, 8], log10_scale_range=[-3, 3])
            self.lam_dpsi_this_iter = self.mp_lam
            self.scale_dpsi_this_iter = self.mp_scale
            # self.lam_dpsi_this_iter = 1e6
            # self.scale_dpsi_this_iter = 0.01
            self.linear_inversion(self.lam_dpsi_this_iter, self.scale_dpsi_this_iter)
            # print('evidence--------', self.regularization_merit([np.log10(self.lam_dpsi_this_iter), np.log10(self.scale_dpsi_this_iter)]))


        dpsi_map_coarse, pix_mass_this_iter, image_model_this_iter, residual_this_iter = self.extract_info_from_linear_inversion()
        self._dpsi_map_coarse.append(dpsi_map_coarse)
        self.pix_mass_this_iter = pix_mass_this_iter
        self.image_model_this_iter = image_model_this_iter
        self.residual_this_iter = residual_this_iter

        #do visualization
        self.visualize_iteration(basedir=self._output_dir, iter_num=self.count_iter)

        #get diagnostics info for this iteration
        self._diag_info.append(self.ret_diag_info())

        if self.has_converged():
            return True
            
        # if not converge, keep updating 
        self.update_iterations()
        return False 


    def has_converged(self):
        relative_change = (self._diag_info[-2]['total_penalty'] - self._diag_info[-1]['total_penalty'])/self._diag_info[-2]['total_penalty']
        print('previous VS current total_penalty:', self._diag_info[-2]['total_penalty'], self._diag_info[-1]['total_penalty'], relative_change)

        if relative_change < 1e-8:
            # return True ##TODO: need to find a better convergence criteria
            return False #currently, just ask the code not automatically stopping
        else:
            return False 


    def run_iter_solve(self):
        for ii in range(1, self._niter):
            condition = self.run_this_iter()
            if condition:
                print('------','code converge')
                break
            else:
                print('------',ii, self.count_iter)  

        
    def visualize_iteration(self, basedir='./result', iter_num=0):
        #SNR, data, mode_image, source, norm-residual, dpsi_this, kappa_this, dpsi_cum, kappa_cum
        abs_path = os.path.abspath(basedir)  #get absolute path
        if not os.path.exists(basedir):  #check if path exist
            os.makedirs(abs_path) #create new directory recursively

        it_plot.visualize_correction_true_src(
            self, 
            basedir=basedir, 
            iter_num=iter_num,
        )