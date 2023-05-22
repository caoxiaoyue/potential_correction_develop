from autoarray.inversion.mappers.delaunay import MapperDelaunay
import autolens as al
import numpy as np
from scipy import linalg
from scipy.optimize import differential_evolution
import time

class PixelizedSource(object):
    def __init__(self, masked_imaging, pixelization_shape_2d=(30,30), reg_type='gradient'):
        self.masked_imaging = masked_imaging
        self.pixelization_shape = pixelization_shape_2d

        self.sparse_image_plane_grid = al.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
            grid=self.masked_imaging.grid, unmasked_sparse_shape=self.pixelization_shape
        ) #sparse pixelized source grid on image-plane

        self.image_data = self.masked_imaging.image.native
        self.image_noise = self.masked_imaging.noise_map.native
        self.image_mask = self.masked_imaging.mask
        self.reg_type = reg_type ##'gradient', 'vkl_exp', 'vkl_gauss'

    def deflection_from(self, pixelized_mass_obj):
        self.alphay_grid, self.alphax_grid = pixelized_mass_obj.eval_alpha_yx_at(self.masked_imaging.grid) #shape: [N_sub, 2]
        self.alphay_inversion, self.alphax_inversion = pixelized_mass_obj.eval_alpha_yx_at(self.masked_imaging.grid_inversion) #shape: [N_sub, 2]
        self.alphay_sparse, self.alphax_sparse = pixelized_mass_obj.eval_alpha_yx_at(self.sparse_image_plane_grid) #shape: [N_sparse_src,2]


    def source_position_from(self, pixelized_mass_obj):
        self.deflection_from(pixelized_mass_obj)

        deflections_2d = al.VectorYX2D(
            list(zip(self.alphay_grid, self.alphax_grid)), 
            self.masked_imaging.grid, 
            self.masked_imaging.mask
        ) #shape: [N_sub, 2]
        self.traced_grid = self.masked_imaging.grid - deflections_2d #shape: [N_sub, 2]

        deflections_2d_inversion = al.VectorYX2D(
            list(zip(self.alphay_inversion, self.alphax_inversion)), 
            self.masked_imaging.grid_inversion, 
            self.masked_imaging.mask,
        )
        self.traced_grid_inversion = self.masked_imaging.grid_inversion - deflections_2d_inversion

        deflections_2d_sparse_grid = al.VectorYX2D(
            list(zip(self.alphay_sparse, self.alphax_sparse)), 
            self.sparse_image_plane_grid, 
            self.masked_imaging.mask,
        )
        self.traced_sparse_grid = self.sparse_image_plane_grid - deflections_2d_sparse_grid

        self.border_relocate()

    
    def border_relocate(self):
        self.relocated_grid = self.traced_grid.relocated_grid_from(grid=self.traced_grid_inversion)
        self.relocated_pixelization_grid = self.traced_grid.relocated_pixelization_grid_from(
            pixelization_grid=self.traced_sparse_grid
        )


    def build_psf_matrix(self):
        self.psf_blur_matrix = self.masked_imaging.convolver.get_psf_blur_matrix() #shape: [N_data, N_data]
        #Note, I add this new method to `convolver` class, this is the `B` matrix in our document


    def build_lens_mapper(self, pixelized_mass_obj):
        self.source_position_from(pixelized_mass_obj)

        grid_delaunay = al.Grid2DDelaunay(
            grid=self.relocated_pixelization_grid, #TODO, better make Delaunay module in autolens change to [(y1,x1), (y2,x2), ...] order
            nearest_pixelization_index_for_slim_index=self.sparse_image_plane_grid.sparse_index_for_slim_index,
        )

        self.mapper = MapperDelaunay(
            source_grid_slim=self.relocated_grid,
            source_pixelization_grid=grid_delaunay,
            data_pixelization_grid=self.sparse_image_plane_grid, 
        )


    def build_lens_mapping(self, pixelized_mass_obj):
        self.build_lens_mapper(pixelized_mass_obj)

        self.mapping_matrix = al.util.mapper.mapping_matrix_from(
            pix_indexes_for_sub_slim_index=self.mapper.pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index=self.mapper.pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=self.mapper.pix_weights_for_sub_slim_index,
            pixels=self.mapper.pixels,
            total_mask_pixels=self.mapper.source_grid_slim.mask.pixels_in_mask,
            slim_index_for_sub_slim_index=self.mapper.slim_index_for_sub_slim_index,
            sub_fraction=self.mapper.source_grid_slim.mask.sub_fraction,
        ) #shape: [N_data, N_src_grid]; Note, not [N_sub_data, N_src_grid]

        # self.blurred_mapping_matrix = self.masked_imaging.convolver.convolve_mapping_matrix(
        #     mapping_matrix=self.mapping_matrix
        # )
        if not hasattr(self, 'psf_blur_matrix'):
            self.build_psf_matrix()
        self.blurred_mapping_matrix = np.matmul(self.psf_blur_matrix, self.mapping_matrix)

        #auxiliary info for source inversion
        self.data_vector = al.util.leq.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=self.blurred_mapping_matrix,
            image=self.masked_imaging.image, #suppose the lens light has been subtracted
            noise_map=self.masked_imaging.noise_map,
        )
        self.F_matrix = al.util.leq.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=self.blurred_mapping_matrix, noise_map=self.masked_imaging.noise_map
        )


    def inverse_covariance_matrix(self):
        noise_1d = self.image_noise[~self.image_mask]
        npix = len(noise_1d)
        self.inv_cov_mat = np.zeros((npix, npix))
        for ii in range(npix):
            self.inv_cov_mat[ii, ii] = 1.0/(noise_1d[ii])**2

    
    def build_reg_matrix(self, lam_s=0.1, scale_s=1.0):
        self.lam_s = lam_s
        self.scale_s = scale_s
        if self.reg_type == 'gradient':
            self.regularization_matrix = al.util.regularization.constant_regularization_matrix_from(
                coefficient=self.lam_s,
                pixel_neighbors=self.mapper.source_pixelization_grid.pixel_neighbors,
                pixel_neighbors_sizes=self.mapper.source_pixelization_grid.pixel_neighbors.sizes,
            )
        else:
            self.regularization_matrix = al.util.regularization.regularization_matrix_vkl_from(
                self.scale_s, 
                self.lam_s, 
                self.relocated_pixelization_grid, 
                self.reg_type,
            )
    
    def source_inversion(self, pixelized_mass_obj, lam_s=0.1, scale_s=1.0, new_lens_mapping=True):
        if new_lens_mapping:
            self.build_lens_mapping(pixelized_mass_obj)

        self.build_reg_matrix(lam_s, scale_s)

        self.F_reg_matrix = np.add(self.F_matrix, self.regularization_matrix)

        self.src_recontruct = np.linalg.solve(self.F_reg_matrix, self.data_vector)

        self.mapped_reconstructed_image = al.util.leq.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=self.blurred_mapping_matrix, reconstruction=self.src_recontruct
        )

        self.residual_map =  self.mapped_reconstructed_image - self.masked_imaging.image
        self.norm_residual_map = self.residual_map / self.masked_imaging.noise_map


    def evidence_from_reconstruction(self):
        # self.log_F_reg_matrix_term = 2.0 * np.sum(
        #     np.log(np.diag(np.linalg.cholesky(self.F_reg_matrix)))
        # ) 

        #sometimes, cholesky factorization fails due to the `non positive definite` error
        #the matrix we deal with is actully indeed positive definite (by checking its eigen values)
        #so this error is purely due to the numerical issue, autolens add a very small potive number 1e-8 to avoid this bug
        #but the code may still fails in certain circumstance (altough, add a larger small number `1e-6`, can remove this bug in my cases)
        #I find the np.linalg.slogdet which use LU factorization can totally avoid this bug, 
        # and its speed is as fast as the cholesky factorization
        sign, logval = np.linalg.slogdet(self.F_reg_matrix) 
        self.log_F_reg_matrix_term = sign*logval

        # self.log_reg_matrix_term = 2.0 * np.sum(
        #     np.log(np.diag(np.linalg.cholesky(self.regularization_matrix)))
        # )
        sign, logval = np.linalg.slogdet(self.regularization_matrix)
        self.log_reg_matrix_term = sign*logval

        self.reg_term = np.matmul(
            self.src_recontruct.T, np.matmul(self.regularization_matrix, self.src_recontruct)
        )

        self.noise_norm = float(np.sum(np.log(2 * np.pi * self.masked_imaging.noise_map ** 2.0)))

        self.chi_squared = np.sum(self.norm_residual_map**2)

        self.log_evidence = float(
            -0.5
            * (
                self.chi_squared 
                + self.reg_term 
                + self.log_F_reg_matrix_term
                - self.log_reg_matrix_term
                + self.noise_norm
            )
        )

        return self.log_evidence
    

    def regularization_merit(self, X):
        if self.reg_type == "gradient":
            self.source_inversion(pixelized_mass_obj=None, lam_s=10**(X[0]), new_lens_mapping=False)
        else:
            self.source_inversion(pixelized_mass_obj=None, lam_s=10**(X[0]), scale_s=10**(X[1]), new_lens_mapping=False)
        return -1.0*self.evidence_from_reconstruction()


    def find_best_regularization(self, pixelized_mass_obj, log10_lam_range=[-4, 4], log10_scale_range=[-3, 3]):
        '''
        return the best-fit regularization strength given a ``fixed'' mass model (pixelized_mass_obj)
        log10_lam_range: set the log range of regularization strength. default: 10^-5 to 10^4 
        '''
        self.build_lens_mapping(pixelized_mass_obj)

        t0 = time.time()
        if self.reg_type == "gradient":
            self.best_fit_reg_info = differential_evolution(self.regularization_merit, bounds=[log10_lam_range,])
        else:
            self.best_fit_reg_info = differential_evolution(self.regularization_merit, bounds=[log10_lam_range, log10_scale_range])
            self.mp_scale = 10**(self.best_fit_reg_info['x'][1]) #this regularization strength maximize the posterior
        t1 = time.time()
        self.mp_lam = 10**(self.best_fit_reg_info['x'][0]) #this regularization strength maximize the posterior
        self.mp_ev = -1.0*self.best_fit_reg_info['fun'] #this is the corrpesonding evidence values

        print('total time elapse:', t1-t0)


    #-------------------sampling the regularization parameters
    def reg_loglike(self, param_cube):
        if self.reg_type == "gradient":
            self.source_inversion(pixelized_mass_obj=None, lam_s=param_cube[0], scale_s=None, new_lens_mapping=False)
        else:
            self.source_inversion(pixelized_mass_obj=None, lam_s=param_cube[0], scale_s=param_cube[1], new_lens_mapping=False)
        return self.evidence_from_reconstruction()


    def reg_ptform(self, param_cube):
        if self.reg_type == "gradient":
            param_cube[0] = 10**((np.log10(self.lam_range[1])-np.log10(self.lam_range[0]))
                                   * param_cube[0]+np.log10(self.lam_range[0]))
        else:
            param_cube[0] = 10**((np.log10(self.lam_range[1])-np.log10(self.lam_range[0]))
                                   * param_cube[0]+np.log10(self.lam_range[0]))
            param_cube[1] = 10**((np.log10(self.scale_range[1])-np.log10(self.scale_range[0]))
                                   * param_cube[1]+np.log10(self.scale_range[0]))
        return param_cube


    def sampling_regularization(self, pixelized_mass_obj, lam_range=[1e-4, 1e4], scale_range=[1e-3, 1e3]):
        import dynesty
        from dynesty import utils as dyfunc

        self.build_lens_mapping(pixelized_mass_obj) ##initialize the lens mapping matrix things
        self.lam_range = lam_range
        self.scale_range = scale_range

        if self.reg_type == "gradient":
            nparams = 1
        else:
            nparams = 2

        self.sampler = dynesty.NestedSampler(
            self.reg_loglike, 
            self.reg_ptform, 
            nparams, 
            nlive=100, 
            bound='multi', 
            # sample='rwalk',
            # walks=10, 
            # facc=0.2,
        )
        self.sampler.run_nested()

        self.results = self.sampler.results  # similar to a python-dict
        self.samples = self.results.samples  # samples
        self.weights = np.exp(self.results.logwt - self.results.logz[-1])   # normalized weights

        quantiles = [dyfunc.quantile(samps, [0.16, 0.50, 0.84], weights=self.weights)
                     for samps in self.samples.T]

        if self.reg_type == "gradient":
            self.reg_lam_low, self.reg_lam_med, self.reg_lam_high = quantiles[0]
            self.reg_lam_err = (self.reg_lam_high - self.reg_lam_low)/2.0
        else:
            self.reg_lam_low, self.reg_lam_med, self.reg_lam_high = quantiles[0]
            self.reg_scale_low, self.reg_scale_med, self.reg_scale_high = quantiles[1]
            self.reg_lam_err = (self.reg_lam_high - self.reg_lam_low)/2.0
            self.reg_scale_err = (self.reg_scale_high - self.reg_scale_low)/2.0

