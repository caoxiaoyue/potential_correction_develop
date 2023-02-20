import autolens as al
import numpy as np
import grid_util
import pixelized_mass
import pixelized_source
import potential_correction_util as pcu
import scipy.linalg as linalg
from scipy.spatial import Delaunay
from potential_correction_util import LinearNDInterpolatorExt
from matplotlib import pyplot as plt
import copy
from plot import pixelized_source as ps_plot
import os
from astropy.io import fits
import logging


class IterativePotentialCorrect(object):
    def __init__(
        self,
        image_shape=(200,200),
        pixel_scale=0.05,
        psf_shape=(11, 11),
        psf_sigma=0.05,
        exp_time=1200,
        backgroud=0.1,
        lens_galaxy=None,
        source_galaxy=None,
        image_mask=None,
        shape_2d_dpsi=None, 
        shape_2d_src=(50,50),
    ):
        grid = al.Grid2DIterate.uniform(
            shape_native=image_shape,
            pixel_scales=pixel_scale,
            fractional_accuracy=0.9999,
            sub_steps=[2, 4, 8, 16, 24],
        )

        psf = al.Kernel2D.from_gaussian(
            shape_native=psf_shape, sigma=psf_sigma, pixel_scales=grid.pixel_scales
        )

        simulator = al.SimulatorImaging(
            exposure_time=exp_time, psf=psf, background_sky_level=backgroud, add_poisson_noise=True, noise_seed=1
        )

        self.input_lens_galaxy = lens_galaxy

        self.input_source_galaxy = source_galaxy

        tracer = al.Tracer.from_galaxies(galaxies=[self.input_lens_galaxy, self.input_source_galaxy])
        imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

        grid_2d = al.Grid2DIterate.uniform(
            shape_native=(200, 200),
            pixel_scales=0.1,
            fractional_accuracy=0.9999,
            sub_steps=[2, 4, 8, 16, 24],
        )
        solver = al.PointSolver(
            grid=grid_2d,
            use_upscaling=True,
            pixel_scale_precision=0.001,
            upscale_factor=2,
            magnification_threshold=1.0,
        )
        self.image_positions = solver.solve(
            lensing_obj=tracer, source_plane_coordinate=self.input_source_galaxy.bulge.centre
        )

        self.masked_imaging = imaging.apply_mask(image_mask)

        """
        shape_2d_dpsi: the shape of potential correction grid, if not set, this will be set to the lens image shape
        shape_2d_src: the number of grid used for source reconstruction (defined on image-plane)
        """
        self.image_data = self.masked_imaging.image.native #native image resolution, not the oversanpling one
        self.image_noise = self.masked_imaging.noise_map.native
        self.psf_kernel =  self.masked_imaging.psf.native
        image_mask = self.masked_imaging.mask 
        dpix_data = self.masked_imaging.pixel_scales[0]

        if shape_2d_dpsi is None:
            shape_2d_dpsi = self.image_data.shape
        self.grid_obj = grid_util.SparseDpsiGrid(image_mask, dpix_data, shape_2d_dpsi=shape_2d_dpsi) #Note, mask_data has not been cleaned

        self.shape_2d_src = shape_2d_src


    def initialize_iteration(
        self, 
        psi_2d_start=None, 
        niter=100, 
        lam_dpsi_start=1e9,
        lam_dpsi_type='2nd',
        psi_anchor_points=None,
        check_converge_points=None,
        subhalo_fiducial_point=None,
        penalty_file="./penalty.txt",
    ):
        """
        psi_2d_0: the lens potential map of the initial start mass model, typicall given by a macro model like elliptical power law model.
        niter: the upper limit of the number of the potential correctio iterations
        lam_s_0: the initial regularization strength of pixelized sources. 
        lam_dpsi_0: the initial regularization strength of potential correction (dpsi)
        lam_dpsi_type: the regularization type of dpsi
        psi_anchor_points: the anchor points of lens potential. we require the lens potential values at those anchor point
        remain unchanged during potential corrention, to avoid various degeneracy problems. (see sec.2.3 in our document);
        dpsi_anchor_points has the following form: [(y1,x1), (y2,x2), (y3,x3)]
        check_converge_points: the points where we check convergence, in autolens [(y1,x1), (y2,x2), (y3,x3), ...] order
        subhalo_fiducial_point: the fiducial location of subhalo, (y_sub, x_sub), mainly for mock test purpose.
        """
        self._niter = niter
        self._lam_dpsi_start = lam_dpsi_start
        self._psi_anchor_points = psi_anchor_points
        self._check_converge_points = check_converge_points
        self._psi_2d_start = psi_2d_start
        self._psi_2d_start[self.masked_imaging.mask] = 0.0 #set the lens potential of masked pixels to 0
        self.merit_file = penalty_file
        self.penalty_file_handle = open(self.merit_file, 'w')
        self.dlm = " "*6
        self.penalty_file_handle.write(f"niter{self.dlm}dpsi_this{self.dlm}dpsi_cum{self.dlm}chi2_image{self.dlm}total\n")

        #do iteration-0, the macro model
        self.count_iter = 0 #count the iteration number
        #1--------regularization of source and lens potential of this iteration
        self.lam_dpsi_this_iter = self._lam_dpsi_start #potential correction reg
        #2--------the lens mass model of currect iteration
        self.pix_mass_this_iter = self.pixelized_mass_from(self._psi_2d_start) #initialize with lens potential given by macro model
        #3---------pix src obj is mainly used for evalulating lens mapping matrix given a lens mass model
        self.pix_src_obj = pixelized_source.PixelizedSource(
            self.masked_imaging, 
            pixelization_shape_2d=self.shape_2d_src,
        ) 

        #to begin the potential correction algorithm, we need a initial guess of source light
        #we use the initial lens potential map given by `self._psi_2d_start` to do the source inversion
        """
        self.pix_src_obj.source_inversion(
            self.pix_mass_this_iter, 
            lam_s=self.lam_s_this_iter,
        )
        #Note: self.s_points_this_iter are given in autolens [(y1,x1),(y2,x2),...] order
        self._s_values_start = self.pix_src_obj.src_recontruct[:] #the intensity values of current best-fit pixelized source model
        self._s_points_start = np.copy(self.pix_src_obj.relocated_pixelization_grid) #the location of pixelized source grids (on source-plane).
        """
        self.pix_src_obj.build_lens_mapping(
            self.pix_mass_this_iter
        )
        self._s_points_start, self._s_values_start, mapped_reconstructed_image = self.pesudo_source_reconstruct(self.pix_src_obj)
        self._residual_start = self.image_data[~self.grid_obj.mask_data] - mapped_reconstructed_image
        self.s_values_this_iter = np.copy(self._s_values_start) #src intentity of current iteration
        self.s_points_this_iter = np.copy(self._s_points_start)
        self.residual_this_iter = np.copy(self._residual_start)

        #Init other auxiliary info
        self._psi_anchor_values = self.pix_mass_this_iter.eval_psi_at(self._psi_anchor_points)
        self._subhalo_fiducial_point = subhalo_fiducial_point
        self.pix_src_obj.inverse_covariance_matrix()
        self._inv_cov_matrix =  np.copy(self.pix_src_obj.inv_cov_mat) #inverse covariance matrix
        self._ns = len(self.s_values_this_iter) #number source grids
        self._np = len(self.grid_obj.xgrid_dpsi_1d) #number dpsi grids
        self._d_1d = self.image_data[~self.grid_obj.mask_data] #1d unmasked image data
        self._n_1d = self.image_noise[~self.grid_obj.mask_data] #1d unmasked noise
        self._B_matrix = np.copy(self.pix_src_obj.psf_blur_matrix) #psf bluring matrix, see eq.7 in our document
        self._Cf_matrix = np.copy(
            self.grid_obj.map_matrix
        ) #see the $C_f$ matrix in our document (eq.7), which interpolate data defined on coarser dpsi grid to native image grid
        self._Dpsi_matrix = pcu.dpsi_gradient_operator_from(
            self.grid_obj.Hx_dpsi, 
            self.grid_obj.Hy_dpsi
        ) #the potential correction gradient operator, see the eq.8 in our document
        self._dpsi_grid_points = np.vstack([self.grid_obj.ygrid_dpsi_1d, self.grid_obj.xgrid_dpsi_1d]).T #points of sparse potential correction grid
        if lam_dpsi_type == '4th':
            self._HTH_dpsi = np.matmul(self.grid_obj.Hx_dpsi_4th_reg.T, self.grid_obj.Hx_dpsi_4th_reg) + \
                np.matmul(self.grid_obj.Hy_dpsi_4th_reg.T, self.grid_obj.Hy_dpsi_4th_reg)
        elif lam_dpsi_type == '2nd':
            self._HTH_dpsi = np.matmul(self.grid_obj.Hx_dpsi_2nd_reg.T, self.grid_obj.Hx_dpsi_2nd_reg) + \
                np.matmul(self.grid_obj.Hy_dpsi_2nd_reg.T, self.grid_obj.Hy_dpsi_2nd_reg)            
        
        #a list which save the potential correction map
        self._dpsi_map_coarse = [np.zeros_like(self.grid_obj.xgrid_dpsi)] #the potential correction map of iteration-0 is 0

        #visualize iteration-0
        self.visualize_iteration(iter_num=self.count_iter)

        norm_residual_1d = self._residual_start / self._n_1d
        chi2_image = np.sum(norm_residual_1d**2)
        self.penalty_file_handle.write(f"{self.count_iter}{self.dlm}{0:.2f}{self.dlm}{0:.2f}{self.dlm}{chi2_image:.2f}{self.dlm}{chi2_image:.2f}\n")

        #assign info of this iteration to the previous one
        self.update_iterations()


    def pesudo_source_reconstruct(self, pix_src_obj):
        src_grid = pix_src_obj.relocated_pixelization_grid
        src_recon = self.input_source_galaxy.image_2d_from(src_grid)
        mapped_reconstructed_image = al.util.leq.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=pix_src_obj.blurred_mapping_matrix, reconstruction=src_recon
        )
        return src_grid, src_recon, mapped_reconstructed_image


    def write_penalty_this_iter(self, iter_num):
        reg_dpsi =  np.matmul(
            self.dpsi_vector, 
            np.matmul(self.reg_term, self.dpsi_vector),
        )

        reg_dpsi_cum = np.asarray(self._dpsi_map_coarse).sum(axis=0)
        reg_dpsi_cum = reg_dpsi_cum[~self.grid_obj.mask_dpsi]
        reg_dpsi_cum = np.matmul(
            reg_dpsi_cum, 
            np.matmul(self.reg_term, reg_dpsi_cum),
        )

        logging.info(f'the log det of reg_dpsi_matrix {np.linalg.slogdet(self.reg_term)}')
        norm_residual_1d = self.residual_this_iter / self._n_1d
        chi2_image = np.sum(norm_residual_1d**2)

        total = chi2_image + reg_dpsi_cum
        self.penalty_file_handle.write(f"{iter_num}{self.dlm}{reg_dpsi:.2f}{self.dlm}{reg_dpsi_cum:.2f}{self.dlm}{chi2_image:.2f}{self.dlm}{total:.2f}\n")
        self.penalty_file_handle.flush()


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


    def update_lam_dpsi(self):
        """
        update the regularization strength of potential correction with iterations
        """
        self.lam_dpsi_this_iter = self.lam_dpsi_prev_iter * 1.0 #* 0.1
        pass 


    def update_iterations(self):
        self.count_iter += 1
        #this iteration becomes previous iteration
        self.lam_dpsi_prev_iter = self.lam_dpsi_this_iter
        self.pix_mass_prev_iter = copy.copy(self.pix_mass_this_iter)
        self.s_values_prev_iter = np.copy(self.s_values_this_iter)
        self.s_points_prev_iter = np.copy(self.s_points_this_iter)
        self.residual_prev_iter = np.copy(self.residual_this_iter)

        #erase information of this iteration 
        self.lam_dpsi_this_iter = None
        self.pix_mass_this_iter = None
        self.s_values_this_iter = None
        self.s_points_this_iter = None
        self.residual_this_iter = None


    def Ds_matrix_from(self, pix_mass_obj, source_points, source_values):
        self.alpha_dpsi_yx = pix_mass_obj.eval_alpha_yx_at(self._dpsi_grid_points) #use previously found pix_mass_object to ray-tracing
        self.alpha_dpsi_yx = np.asarray(self.alpha_dpsi_yx).T
        self.src_plane_dpsi_yx = self._dpsi_grid_points - self.alpha_dpsi_yx #the location of dpsi grid on the source-plane
        source_gradient = pcu.source_gradient_from(
            source_points, #previously found best-fit src pixlization grids
            source_values, #previously found best-fit src reconstruction
            self.src_plane_dpsi_yx, 
            cross_size=0.01,
        )
        return pcu.source_gradient_matrix_from(source_gradient)  

    
    def return_reg_matrix(self, lam_dpsi):
        #the regularization matrix for lens potential corrections.
        return self._HTH_dpsi * lam_dpsi  #Note, not self.lam_dpsi_this_iter**2 here


    def intensity_deficit_matrix_from(self, pix_mass_obj, source_points, source_values):
        self.Ds_matrix = self.Ds_matrix_from(pix_mass_obj, source_points, source_values)
        self.intensity_deficit_matrix = -1.0*np.matmul(
            self._Cf_matrix,
            np.matmul(
                self.Ds_matrix,
                self._Dpsi_matrix,
            )
        )
        return np.matmul(self._B_matrix, self.intensity_deficit_matrix)


    def return_data_vector(self, blur_intensity_deficit_matrix, residual_1d):
        data_vector = np.matmul(
            np.matmul(blur_intensity_deficit_matrix.T, self._inv_cov_matrix),
            residual_1d,
        )
        return data_vector

    
    def run_this_iteration(self):
        #update regularization parameters for this iteration
        self.update_lam_dpsi()

        self.blur_intensity_deficit_matrix = self.intensity_deficit_matrix_from(
            self.pix_mass_prev_iter, 
            self.s_points_prev_iter, 
            self.s_values_prev_iter
        )
        self.data_vector = self.return_data_vector(self.blur_intensity_deficit_matrix, self.residual_prev_iter)

        #solve the next source and potential corrections
        self.curve_term = np.matmul(
            np.matmul(self.blur_intensity_deficit_matrix.T, self._inv_cov_matrix),
            self.blur_intensity_deficit_matrix,
        )
        self.reg_term = self.return_reg_matrix(self.lam_dpsi_this_iter) 
        self.curve_reg_term = self.curve_term + self.reg_term
        
        # print('~~~~~~~~~~~~~~~~iteration-{}, r-condition number {:.5e}'.format(self.count_iter, 1/np.linalg.cond(self.curve_reg_term)))
        self.dpsi_vector = linalg.solve(self.curve_reg_term, self.data_vector)

        #extract the potential correction info
        dpsi_2d = np.zeros_like(self._psi_2d_start, dtype='float')
        dpsi_2d[~self.grid_obj.mask_data] = np.matmul(
            self._Cf_matrix, 
            self.dpsi_vector,
        )
        #update lens potential with potential correction at this iteration
        psi_2d_this_iter = self.pix_mass_prev_iter.psi_map + dpsi_2d #the new 2d lens potential map
        #rescale the current lens potential, to avoid various degeneracy problems. (see sec.2.3 in our document);
        psi_2d_this_iter, factor = self.rescale_lens_potential(psi_2d_this_iter)
        #save the coarse potential correction map
        dpsi_map_coarse = np.zeros_like(self.grid_obj.xgrid_dpsi)
        dpsi_map_coarse[~self.grid_obj.mask_dpsi] = self.dpsi_vector[:]
        dpsi_map_coarse = dpsi_map_coarse + factor[0]*self.grid_obj.ygrid_dpsi + factor[1]*self.grid_obj.xgrid_dpsi + factor[2]
        self._dpsi_map_coarse.append(dpsi_map_coarse)

        #get pixelized mass object of this iteration
        self.pix_mass_this_iter = self.pixelized_mass_from(psi_2d_this_iter)
        #sovle for the source under the lens potential of this iteration
        self.pix_src_obj.build_lens_mapping(
            self.pix_mass_this_iter
        )
        self.s_points_this_iter, self.s_values_this_iter, mapped_reconstructed_image = self.pesudo_source_reconstruct(self.pix_src_obj)
        self.residual_this_iter = self._d_1d - mapped_reconstructed_image

        self.write_penalty_this_iter(self.count_iter) 

        #do visualization
        self.visualize_iteration(iter_num=self.count_iter)

        #check convergence
        if self.has_converged(dpsi_2d, self.pix_mass_prev_iter.psi_map): #TODO, should not use dpsi_2d here, wrong, since dpsi_2d need rescale
            return True
            
        # if not converge, keep updating 
        self.update_iterations()
        return False 

    
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

        # #check psi rescaling-----------
        # psi_interpolator_check = LinearNDInterpolatorExt(self.tri_psi_interp, psi_2d_out[~self.grid_obj.mask_data])
        # psi_anchor_values_check = psi_interpolator_check(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0])
        # print('diff_origin', psi_anchor_values_new-self._psi_anchor_values)
        # print(self._psi_anchor_values,'psi rescaling check',psi_anchor_values_check, 'diff_rescale', psi_anchor_values_check-self._psi_anchor_values)
        # print('rescale factor', factor)
        return psi_2d_out, factor


    def has_converged(self, dpsi_2d, psi_2d):
        if not hasattr(self, 'tri_psi_interp'):
            self.tri_psi_interp = Delaunay(
                list(zip(self.grid_obj.xgrid_data_1d, self.grid_obj.ygrid_data_1d))
            )

        dpsi_interpolator = LinearNDInterpolatorExt(self.tri_psi_interp, dpsi_2d[~self.grid_obj.mask_data])
        psi_interpolator = LinearNDInterpolatorExt(self.tri_psi_interp, psi_2d[~self.grid_obj.mask_data])

        dpsi_values = dpsi_interpolator(self._check_converge_points[:,1], self._check_converge_points[:,0])
        psi_values = psi_interpolator(self._check_converge_points[:,1], self._check_converge_points[:,0])

        if_converge = True
        abs_delta_list = []
        for ii in range(0, len(dpsi_values)-1):
            for jj in range(ii+1, len(dpsi_values)):
                abs_delta = abs((dpsi_values[ii] - dpsi_values[jj])/(psi_values[ii] - psi_values[jj]))
                if abs_delta > 0.1/100:
                    if_converge = False
                    abs_delta_list.append(abs_delta)
        
        if len(abs_delta_list) > 0:
            print('max relative change of dpsi-----', max(abs_delta_list))

        return if_converge


    def run_iter_solve(self):
        for ii in range(1, self._niter):
            condition = self.run_this_iteration()
            if condition:
                print('------','code converge')
                self.penalty_file_handle.close()    
                break
            else:
                print('------',ii, 'of', self.count_iter)  

 
    def visualize_iteration(self, basedir='./result', iter_num=0, save_fits=False):
        plt.figure(figsize=(15, 15))
        percent = [0,100]
        cbpar = {}
        cbpar['fraction'] = 0.046
        cbpar['pad'] = 0.04
        myargs = {'origin':'upper'}
        cmap = copy.copy(plt.get_cmap('jet'))
        cmap.set_bad(color='white')
        myargs['cmap'] = cmap
        myargs['extent'] = copy.copy(self.grid_obj.image_bound)
        markersize = 10

        rgrid = np.sqrt(self.grid_obj.xgrid_data**2 + self.grid_obj.ygrid_data**2)
        limit = np.max(rgrid[~self.grid_obj.mask_data])

        #--------SN image
        plt.subplot(331)
        masked_SN_data = np.ma.masked_array(self.image_data/self.image_noise, mask=self.grid_obj.mask_data)
        plt.imshow(masked_SN_data, **myargs)
        plt.plot(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0], 'k+', ms=markersize)
        plt.plot(self._check_converge_points[:,1], self._check_converge_points[:,0], 'w+', ms=markersize)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title(f'Data-SN, Niter={iter_num}')
        plt.ylabel('Arcsec')

        #--------data image
        plt.subplot(332)
        vmin = np.percentile(self.image_data,percent[0]) 
        vmax = np.percentile(self.image_data,percent[1]) 
        masked_image_data = np.ma.masked_array(self.image_data, mask=self.grid_obj.mask_data)
        plt.imshow(masked_image_data, vmin=vmin, vmax=vmax,**myargs)
        plt.plot(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0], 'k+', ms=markersize)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title(f'Data-image')

        #---------model reconstruction given current mass model
        mapped_reconstructed_image_2d = np.zeros_like(self.image_data)
        self.pix_src_obj.build_lens_mapping(
            self.pix_mass_this_iter
        )
        s_points, s_values, mapped_reconstructed_image = self.pesudo_source_reconstruct(self.pix_src_obj)
        mapped_reconstructed_image_2d[~self.grid_obj.mask_data] = np.copy(mapped_reconstructed_image)
        plt.subplot(333)
        vmin = np.percentile(mapped_reconstructed_image_2d,percent[0]) 
        vmax = np.percentile(mapped_reconstructed_image_2d,percent[1])
        mapped_reconstructed_image_2d = np.ma.masked_array(
            mapped_reconstructed_image_2d, 
            mask=self.grid_obj.mask_data
        ) 
        plt.imshow(mapped_reconstructed_image_2d,vmin=vmin,vmax=vmax,**myargs)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.plot(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0], 'k+', ms=markersize)
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title('Model-image')

        #----------normalized residual
        plt.subplot(334)
        norm_residual_map_2d = np.zeros_like(self.image_data)
        norm_residual_map_2d[~self.grid_obj.mask_data] = (mapped_reconstructed_image - self._d_1d)/self._n_1d
        vmin = np.percentile(norm_residual_map_2d,percent[0]) 
        vmax = np.percentile(norm_residual_map_2d,percent[1])
        norm_residual_map_2d = np.ma.masked_array(
            norm_residual_map_2d, 
            mask=self.grid_obj.mask_data
        )  
        plt.imshow(norm_residual_map_2d,vmin=vmin,vmax=vmax,**myargs)
        plt.plot(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0], 'k+', ms=markersize)
        plt.plot(self._check_converge_points[:,1], self._check_converge_points[:,0], 'w+', ms=markersize)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.ylabel('Arcsec')
        plt.title('Norm-residual')

        #-------------source image given current mass model
        plt.subplot(335)
        this_ax = plt.gca()
        ps_plot.visualize_source(
            s_points, 
            s_values,
            ax=this_ax,
        )
        this_ax.set_title('Source')
        plt.xlabel('Arcsec')
        plt.ylabel('Arcsec')
        
        #-------------potential correction of this iteration
        plt.subplot(336)
        psi_correct_this_iter =  np.asarray(self._dpsi_map_coarse)[-1]
        masked_psi_correct_this_iter = np.ma.masked_array(
            psi_correct_this_iter, 
            mask=self.grid_obj.mask_dpsi
        )
        plt.imshow(masked_psi_correct_this_iter,**myargs)
        plt.plot(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0], 'k+', ms=markersize)
        plt.plot(self._check_converge_points[:,1], self._check_converge_points[:,0], 'w+', ms=markersize)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title(r'$\delta \psi$')

        #-------------convergence correction of this iteration
        plt.subplot(337)
        kappa_correct_this_iter =  np.zeros_like(psi_correct_this_iter)
        kappa_correct_this_iter_1d = np.matmul(
            self.grid_obj.hamiltonian_dpsi,
            psi_correct_this_iter[~self.grid_obj.mask_dpsi]
        )
        kappa_correct_this_iter[~self.grid_obj.mask_dpsi] = kappa_correct_this_iter_1d
        masked_kappa_correct_this_iter = np.ma.masked_array(
            kappa_correct_this_iter, 
            mask=self.grid_obj.mask_dpsi
        )
        plt.imshow(masked_kappa_correct_this_iter,**myargs)
        plt.plot(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0], 'k+', ms=markersize)
        plt.plot(self._check_converge_points[:,1], self._check_converge_points[:,0], 'w+', ms=markersize)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        if self._subhalo_fiducial_point is not None:
            for item in self._subhalo_fiducial_point: 
                plt.plot(item[1], item[0], 'k*', ms=markersize)
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title(r'$\delta \kappa$')
        plt.xlabel('Arcsec')
        plt.ylabel('Arcsec')

        #-------------cumulative potential correction
        plt.subplot(338)
        cumulative_psi_correct =  np.asarray(self._dpsi_map_coarse).sum(axis=0)
        masked_cumulative_psi_correct = np.ma.masked_array(
            cumulative_psi_correct, 
            mask=self.grid_obj.mask_dpsi
        )
        plt.imshow(masked_cumulative_psi_correct,**myargs)
        plt.plot(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0], 'k+', ms=markersize)
        plt.plot(self._check_converge_points[:,1], self._check_converge_points[:,0], 'w+', ms=markersize)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title(r'$\Sigma \left(\delta \psi \right)$')
        plt.xlabel('Arcsec')

        #-------------cumulative convergence correction
        plt.subplot(339)
        cumulative_kappa_correct = np.zeros_like(cumulative_psi_correct)
        cumulative_kappa_correct_1d = np.matmul(
            self.grid_obj.hamiltonian_dpsi,
            cumulative_psi_correct[~self.grid_obj.mask_dpsi]
        )
        cumulative_kappa_correct[~self.grid_obj.mask_dpsi] = cumulative_kappa_correct_1d
        masked_cumulative_kappa_correct = np.ma.masked_array(
            cumulative_kappa_correct, 
            mask=self.grid_obj.mask_dpsi
        )
        plt.imshow(masked_cumulative_kappa_correct,**myargs)
        plt.plot(self._psi_anchor_points[:,1], self._psi_anchor_points[:,0], 'k+', ms=markersize)
        plt.plot(self._check_converge_points[:,1], self._check_converge_points[:,0], 'w+', ms=markersize)
        if self._subhalo_fiducial_point is not None:
            for item in self._subhalo_fiducial_point: 
                plt.plot(item[1], item[0], 'k*', ms=markersize)
        cb=plt.colorbar(**cbpar)
        cb.ax.minorticks_on()
        cb.ax.tick_params(labelsize='small')
        plt.xlim(-1.0*limit, limit)
        plt.ylim(-1.0*limit, limit)
        plt.title(r'$\Sigma \left(\delta \kappa \right)$')
        plt.xlabel('Arcsec')

        plt.tight_layout()
        plt.savefig(f'{basedir}/{iter_num}.jpg', bbox_inches='tight')
        plt.close()

        if save_fits:
            self.save_correction_fits(
                basedir=basedir,
                iter_num=iter_num,
                kappa_correct=cumulative_kappa_correct,
                psi_correct=cumulative_psi_correct,
            )


    def save_correction_fits(
        self, 
        basedir=None, 
        iter_num=None,
        kappa_correct=None,
        psi_correct=None,
    ):
        abs_path = os.path.abspath(basedir)  #get absolute path
        if not os.path.exists(f"{abs_path}/fits"):  #check if path exist
            os.makedirs(f"{abs_path}/fits") #create new directory recursively
        fits.writeto(f'{basedir}/fits/kappa_correction_{iter_num}.fits', kappa_correct, overwrite=True)
        fits.writeto(f'{basedir}/fits/psi_correction_{iter_num}.fits', psi_correct, overwrite=True)