from matplotlib import pyplot as plt
from plot import pixelized_source as ps_plot
import copy
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_correction(potential_correcter, basedir='./result', iter_num=0):
    plt.figure(figsize=(15, 15))
    percent = [0,100]
    cbpar = {}
    cbpar['fraction'] = 0.046
    cbpar['pad'] = 0.04
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    myargs_data = {'origin':'upper'}
    myargs_data['cmap'] = cmap
    myargs_data['extent'] = copy.copy(potential_correcter.grid_obj.data_bound)
    myargs_dpsi = copy.deepcopy(myargs_data)
    myargs_dpsi['extent'] = copy.copy(potential_correcter.grid_obj.dpsi_bound)

    markersize = 10

    rgrid = np.sqrt(potential_correcter.grid_obj.xgrid_data**2 + potential_correcter.grid_obj.ygrid_data**2)
    limit = np.max(rgrid[~potential_correcter.grid_obj.mask_data])
    #--------SN image
    plt.subplot(331)
    masked_SN_data = np.ma.masked_array(potential_correcter.image_data/potential_correcter.image_noise, mask=potential_correcter.grid_obj.mask_data)
    plt.imshow(masked_SN_data, **myargs_data)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title(f'Data-SN, Niter={iter_num}')
    plt.ylabel('Arcsec')
    #--------data image
    plt.subplot(332)
    vmin = np.percentile(potential_correcter.image_data,percent[0]) 
    vmax = np.percentile(potential_correcter.image_data,percent[1]) 
    masked_image_data = np.ma.masked_array(potential_correcter.image_data, mask=potential_correcter.grid_obj.mask_data)
    plt.imshow(masked_image_data, vmin=vmin, vmax=vmax,**myargs_data)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title(f'Data-image')
    #---------model reconstruction given current mass model
    plt.subplot(333)
    mapped_reconstructed_image_2d = np.zeros_like(potential_correcter.image_data)
    try:
        #NOTE, the current best source regularization coupled with the previous best-fit mass model
        potential_correcter.pix_src_obj.source_inversion(potential_correcter.pix_mass_prev_iter, lam_s=potential_correcter.lam_s_this_iter, scale_s=potential_correcter.scale_s_this_iter)
    except:
        potential_correcter.pix_src_obj.source_inversion(potential_correcter.pix_mass_prev_iter, lam_s=potential_correcter.lam_s_this_iter)
    mapped_reconstructed_image_2d[~potential_correcter.grid_obj.mask_data] = np.copy(potential_correcter.pix_src_obj.mapped_reconstructed_image)
    vmin = np.percentile(mapped_reconstructed_image_2d,percent[0]) 
    vmax = np.percentile(mapped_reconstructed_image_2d,percent[1])
    mapped_reconstructed_image_2d = np.ma.masked_array(
        mapped_reconstructed_image_2d, 
        mask=potential_correcter.grid_obj.mask_data
    ) 
    plt.imshow(mapped_reconstructed_image_2d,vmin=vmin,vmax=vmax,**myargs_data)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title('Model-image')
    #----------normalized residual
    plt.subplot(334)
    norm_residual_map_2d = np.zeros_like(potential_correcter.image_data)
    norm_residual_map_2d[~potential_correcter.grid_obj.mask_data] = np.copy(potential_correcter.pix_src_obj.norm_residual_map)
    vmin = np.percentile(norm_residual_map_2d,percent[0]) 
    vmax = np.percentile(norm_residual_map_2d,percent[1])
    norm_residual_map_2d = np.ma.masked_array(
        norm_residual_map_2d, 
        mask=potential_correcter.grid_obj.mask_data
    )  
    plt.imshow(norm_residual_map_2d,vmin=vmin,vmax=vmax,**myargs_data)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
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
        potential_correcter.pix_src_obj.relocated_pixelization_grid, 
        potential_correcter.pix_src_obj.src_recontruct[:] ,
        ax=this_ax,
    )
    this_ax.set_title('Source')
    #-------------potential correction of this iteration
    plt.subplot(336)
    psi_correct_this_iter =  np.asarray(potential_correcter._dpsi_map_coarse)[-1]
    masked_psi_correct_this_iter = np.ma.masked_array(
        psi_correct_this_iter, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_psi_correct_this_iter,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.xlim(-1.0*limit, limit) ##her is inaccurate plot
    plt.ylim(-1.0*limit, limit)
    plt.title(r'$\delta \psi$')
    #-------------convergence correction of this iteration
    plt.subplot(337)
    kappa_correct_this_iter =  np.zeros_like(psi_correct_this_iter)
    kappa_correct_this_iter_1d = np.matmul(
        potential_correcter.grid_obj.hamiltonian_dpsi,
        psi_correct_this_iter[~potential_correcter.grid_obj.mask_dpsi]
    )
    kappa_correct_this_iter[~potential_correcter.grid_obj.mask_dpsi] = kappa_correct_this_iter_1d
    masked_kappa_correct_this_iter = np.ma.masked_array(
        kappa_correct_this_iter, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_kappa_correct_this_iter,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title(r'$\delta \kappa$')
    plt.xlabel('Arcsec')
    plt.ylabel('Arcsec')
    #-------------cumulative potential correction
    plt.subplot(338)
    cumulative_psi_correct =  np.asarray(potential_correcter._dpsi_map_coarse).sum(axis=0)
    masked_cumulative_psi_correct = np.ma.masked_array(
        cumulative_psi_correct, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_cumulative_psi_correct,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
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
        potential_correcter.grid_obj.hamiltonian_dpsi,
        cumulative_psi_correct[~potential_correcter.grid_obj.mask_dpsi]
    )
    cumulative_kappa_correct[~potential_correcter.grid_obj.mask_dpsi] = cumulative_kappa_correct_1d
    masked_cumulative_kappa_correct = np.ma.masked_array(
        cumulative_kappa_correct, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_cumulative_kappa_correct,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
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


def visualize_correction_true_src(potential_correcter, basedir='./result', iter_num=0):
    plt.figure(figsize=(15, 15))
    percent = [0,100]
    cbpar = {}
    cbpar['fraction'] = 0.046
    cbpar['pad'] = 0.04
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    myargs_data = {'origin':'upper'}
    myargs_data['cmap'] = cmap
    myargs_data['extent'] = copy.copy(potential_correcter.grid_obj.data_bound)
    myargs_dpsi = copy.deepcopy(myargs_data)
    myargs_dpsi['extent'] = copy.copy(potential_correcter.grid_obj.dpsi_bound)

    markersize = 10

    rgrid = np.sqrt(potential_correcter.grid_obj.xgrid_data**2 + potential_correcter.grid_obj.ygrid_data**2)
    limit = np.max(rgrid[~potential_correcter.grid_obj.mask_data])
    #--------SN image
    plt.subplot(331)
    masked_SN_data = np.ma.masked_array(potential_correcter.image_data/potential_correcter.image_noise, mask=potential_correcter.grid_obj.mask_data)
    plt.imshow(masked_SN_data, **myargs_data)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title(f'Data-SN, Niter={iter_num}')
    plt.ylabel('Arcsec')
    #--------data image
    plt.subplot(332)
    vmin = np.percentile(potential_correcter.image_data,percent[0]) 
    vmax = np.percentile(potential_correcter.image_data,percent[1]) 
    masked_image_data = np.ma.masked_array(potential_correcter.image_data, mask=potential_correcter.grid_obj.mask_data)
    plt.imshow(masked_image_data, vmin=vmin, vmax=vmax,**myargs_data)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title(f'Data-image')
    #---------model reconstruction given current mass model
    plt.subplot(333)
    mapped_reconstructed_image_2d = np.copy(potential_correcter.image_model_this_iter)
    vmin = np.percentile(mapped_reconstructed_image_2d,percent[0]) 
    vmax = np.percentile(mapped_reconstructed_image_2d,percent[1])
    mapped_reconstructed_image_2d = np.ma.masked_array(
        mapped_reconstructed_image_2d, 
        mask=potential_correcter.grid_obj.mask_data
    ) 
    plt.imshow(mapped_reconstructed_image_2d,vmin=vmin,vmax=vmax,**myargs_data)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title('Model-image')
    #----------normalized residual
    plt.subplot(334)
    norm_residual_map_2d = np.zeros_like(potential_correcter.image_data)
    norm_residual_map_2d[~potential_correcter.grid_obj.mask_data] = -1.0*potential_correcter.residual_this_iter/potential_correcter._n_1d
    vmin = np.percentile(norm_residual_map_2d,percent[0]) ##NOTE, -1 here is for consistency with the definition used in the pixelized src module
    vmax = np.percentile(norm_residual_map_2d,percent[1])
    norm_residual_map_2d = np.ma.masked_array(
        norm_residual_map_2d, 
        mask=potential_correcter.grid_obj.mask_data
    )  
    plt.imshow(norm_residual_map_2d,vmin=vmin,vmax=vmax,**myargs_data)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
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
        potential_correcter._s_points_0, 
        potential_correcter._s_values_0,
        ax=this_ax,
    )
    this_ax.scatter(potential_correcter.src_plane_data_yx[:,1], potential_correcter.src_plane_data_yx[:,0], s=0.1)
    this_ax.set_title('Source')
    #-------------potential correction of this iteration
    plt.subplot(336)
    psi_correct_this_iter =  np.asarray(potential_correcter._dpsi_map_coarse)[-1]
    masked_psi_correct_this_iter = np.ma.masked_array(
        psi_correct_this_iter, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_psi_correct_this_iter,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.xlim(-1.0*limit, limit) ##her is inaccurate plot
    plt.ylim(-1.0*limit, limit)
    plt.title(r'$\delta \psi$')
    #-------------convergence correction of this iteration
    plt.subplot(337)
    kappa_correct_this_iter =  np.zeros_like(psi_correct_this_iter)
    kappa_correct_this_iter_1d = np.matmul(
        potential_correcter.grid_obj.hamiltonian_dpsi,
        psi_correct_this_iter[~potential_correcter.grid_obj.mask_dpsi]
    )
    kappa_correct_this_iter[~potential_correcter.grid_obj.mask_dpsi] = kappa_correct_this_iter_1d
    masked_kappa_correct_this_iter = np.ma.masked_array(
        kappa_correct_this_iter, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_kappa_correct_this_iter,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title(r'$\delta \kappa$')
    plt.xlabel('Arcsec')
    plt.ylabel('Arcsec')
    #-------------cumulative potential correction
    plt.subplot(338)
    cumulative_psi_correct =  np.asarray(potential_correcter._dpsi_map_coarse).sum(axis=0)
    masked_cumulative_psi_correct = np.ma.masked_array(
        cumulative_psi_correct, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_cumulative_psi_correct,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
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
        potential_correcter.grid_obj.hamiltonian_dpsi,
        cumulative_psi_correct[~potential_correcter.grid_obj.mask_dpsi]
    )
    cumulative_kappa_correct[~potential_correcter.grid_obj.mask_dpsi] = cumulative_kappa_correct_1d
    masked_cumulative_kappa_correct = np.ma.masked_array(
        cumulative_kappa_correct, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_cumulative_kappa_correct,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
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


def visualize_correction_vkl(potential_correcter, basedir='./result', iter_num=0):
    plt.figure(figsize=(15, 15))
    percent = [0,100]
    cbpar = {}
    cbpar['fraction'] = 0.046
    cbpar['pad'] = 0.04
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    myargs_data = {'origin':'upper'}
    myargs_data['cmap'] = cmap
    myargs_data['extent'] = copy.copy(potential_correcter.grid_obj.data_bound)
    myargs_dpsi = copy.deepcopy(myargs_data)
    myargs_dpsi['extent'] = copy.copy(potential_correcter.grid_obj.dpsi_bound)

    markersize = 10

    rgrid = np.sqrt(potential_correcter.grid_obj.xgrid_data**2 + potential_correcter.grid_obj.ygrid_data**2)
    limit = np.max(rgrid[~potential_correcter.grid_obj.mask_data])
    #--------SN image
    plt.subplot(331)
    masked_SN_data = np.ma.masked_array(potential_correcter.image_data/potential_correcter.image_noise, mask=potential_correcter.grid_obj.mask_data)
    plt.imshow(masked_SN_data, **myargs_data)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title(f'Data-SN, Niter={iter_num}')
    plt.ylabel('Arcsec')
    #--------data image
    plt.subplot(332)
    vmin = np.percentile(potential_correcter.image_data,percent[0]) 
    vmax = np.percentile(potential_correcter.image_data,percent[1]) 
    masked_image_data = np.ma.masked_array(potential_correcter.image_data, mask=potential_correcter.grid_obj.mask_data)
    plt.imshow(masked_image_data, vmin=vmin, vmax=vmax,**myargs_data)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title(f'Data-image')
    #---------model reconstruction given current mass model
    plt.subplot(333)
    mapped_reconstructed_image_2d = np.zeros_like(potential_correcter.image_data)
    mapped_reconstructed_image_2d[~potential_correcter.grid_obj.mask_data] = np.copy(potential_correcter.model_image)
    vmin = np.percentile(mapped_reconstructed_image_2d,percent[0]) 
    vmax = np.percentile(mapped_reconstructed_image_2d,percent[1])
    mapped_reconstructed_image_2d = np.ma.masked_array(
        mapped_reconstructed_image_2d, 
        mask=potential_correcter.grid_obj.mask_data
    ) 
    plt.imshow(mapped_reconstructed_image_2d,vmin=vmin,vmax=vmax,**myargs_data)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title('Model-image')
    #----------normalized residual
    plt.subplot(334)
    norm_residual_map_2d = np.zeros_like(potential_correcter.image_data)
    norm_residual_map_2d[~potential_correcter.grid_obj.mask_data] = np.copy(potential_correcter.residual_this_iter/potential_correcter._n_1d)
    vmin = np.percentile(norm_residual_map_2d,percent[0]) 
    vmax = np.percentile(norm_residual_map_2d,percent[1])
    norm_residual_map_2d = np.ma.masked_array(
        norm_residual_map_2d, 
        mask=potential_correcter.grid_obj.mask_data
    )  
    plt.imshow(norm_residual_map_2d,vmin=vmin,vmax=vmax,**myargs_data)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
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
        potential_correcter.s_points_this_iter, 
        potential_correcter.s_values_this_iter,
        ax=this_ax,
    )
    this_ax.set_title('Source')
    #-------------potential correction of this iteration
    plt.subplot(336)
    psi_correct_this_iter =  np.asarray(potential_correcter.info_list[-1]['dpsi_map_coarse'])
    masked_psi_correct_this_iter = np.ma.masked_array(
        psi_correct_this_iter, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_psi_correct_this_iter,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    plt.xlim(-1.0*limit, limit) ##her is inaccurate plot
    plt.ylim(-1.0*limit, limit)
    plt.title(r'$\delta \psi$')
    #-------------convergence correction of this iteration
    plt.subplot(337)
    kappa_correct_this_iter =  np.zeros_like(psi_correct_this_iter)
    kappa_correct_this_iter_1d = np.matmul(
        potential_correcter.grid_obj.hamiltonian_dpsi,
        psi_correct_this_iter[~potential_correcter.grid_obj.mask_dpsi]
    )
    kappa_correct_this_iter[~potential_correcter.grid_obj.mask_dpsi] = kappa_correct_this_iter_1d
    masked_kappa_correct_this_iter = np.ma.masked_array(
        kappa_correct_this_iter, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_kappa_correct_this_iter,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    cb=plt.colorbar(**cbpar)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
    plt.xlim(-1.0*limit, limit)
    plt.ylim(-1.0*limit, limit)
    plt.title(r'$\delta \kappa$')
    plt.xlabel('Arcsec')
    plt.ylabel('Arcsec')
    #-------------cumulative potential correction
    plt.subplot(338)
    dpsi_map_coarse_all = [item['dpsi_map_coarse'] for item in potential_correcter.info_list]
    cumulative_psi_correct =  np.asarray(dpsi_map_coarse_all).sum(axis=0)
    masked_cumulative_psi_correct = np.ma.masked_array(
        cumulative_psi_correct, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_cumulative_psi_correct,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
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
        potential_correcter.grid_obj.hamiltonian_dpsi,
        cumulative_psi_correct[~potential_correcter.grid_obj.mask_dpsi]
    )
    cumulative_kappa_correct[~potential_correcter.grid_obj.mask_dpsi] = cumulative_kappa_correct_1d
    masked_cumulative_kappa_correct = np.ma.masked_array(
        cumulative_kappa_correct, 
        mask=potential_correcter.grid_obj.mask_dpsi
    )
    plt.imshow(masked_cumulative_kappa_correct,**myargs_dpsi)
    plt.plot(potential_correcter._psi_anchor_points[:,1], potential_correcter._psi_anchor_points[:,0], 'k+', ms=markersize)
    if potential_correcter._subhalo_fiducial_point is not None:
        plt.plot(potential_correcter._subhalo_fiducial_point[1], potential_correcter._subhalo_fiducial_point[0], 'k*', ms=markersize)
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



def plot_dpsi(potential_corrector, ax=None):
    # cbpar = {}
    # cbpar['fraction'] = 0.046
    # cbpar['pad'] = 0.04
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    myargs_data = {'origin':'upper'}
    myargs_data['cmap'] = cmap
    myargs_data['extent'] = copy.copy(potential_corrector.grid_obj.data_bound)
    myargs_dpsi = copy.deepcopy(myargs_data)
    myargs_dpsi['extent'] = copy.copy(potential_corrector.grid_obj.dpsi_bound)
    markersize = 10

    rgrid = np.sqrt(potential_corrector.grid_obj.xgrid_data**2 + potential_corrector.grid_obj.ygrid_data**2)
    limit = np.max(rgrid[~potential_corrector.grid_obj.mask_data])

    psi_correct_this_iter =  np.asarray(potential_corrector.info_list[-1]['dpsi_map_coarse'])
    masked_psi_correct_this_iter = np.ma.masked_array(
        psi_correct_this_iter, 
        mask=potential_corrector.grid_obj.mask_dpsi
    )

    im = ax.imshow(masked_psi_correct_this_iter,**myargs_dpsi)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # cb=plt.colorbar(**cbpar)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    ax.plot(potential_corrector._psi_anchor_points[:,1], potential_corrector._psi_anchor_points[:,0], 'k+', ms=markersize)

    ax.set_xlim(-1.0*limit, limit) 
    ax.set_ylim(-1.0*limit, limit)
    ax.set_xlabel('Arcsec')
    ax.set_ylabel('Arcsec')
    ax.set_title(r'$\delta \psi$')


def plot_dkappa(potential_corrector, ax=None):
    cmap = copy.copy(plt.get_cmap('jet'))
    cmap.set_bad(color='white')
    myargs_data = {'origin':'upper'}
    myargs_data['cmap'] = cmap
    myargs_data['extent'] = copy.copy(potential_corrector.grid_obj.data_bound)
    myargs_dpsi = copy.deepcopy(myargs_data)
    myargs_dpsi['extent'] = copy.copy(potential_corrector.grid_obj.dpsi_bound)
    markersize = 10

    rgrid = np.sqrt(potential_corrector.grid_obj.xgrid_data**2 + potential_corrector.grid_obj.ygrid_data**2)
    limit = np.max(rgrid[~potential_corrector.grid_obj.mask_data])

    kappa_correct_this_iter =  np.zeros_like(potential_corrector.grid_obj.mask_dpsi, dtype='float')
    psi_correct_this_iter =  np.asarray(potential_corrector.info_list[-1]['dpsi_map_coarse'])
    kappa_correct_this_iter_1d = np.matmul(
        potential_corrector.grid_obj.hamiltonian_dpsi,
        psi_correct_this_iter[~potential_corrector.grid_obj.mask_dpsi]
    )
    kappa_correct_this_iter[~potential_corrector.grid_obj.mask_dpsi] = kappa_correct_this_iter_1d
    masked_kappa_correct_this_iter = np.ma.masked_array(
        kappa_correct_this_iter, 
        mask=potential_corrector.grid_obj.mask_dpsi
    )

    im = ax.imshow(masked_kappa_correct_this_iter,**myargs_dpsi)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.minorticks_on()
    cb.ax.tick_params(labelsize='small')
    ax.plot(potential_corrector._psi_anchor_points[:,1], potential_corrector._psi_anchor_points[:,0], 'k+', ms=markersize)

    ax.set_xlim(-1.0*limit, limit) 
    ax.set_ylim(-1.0*limit, limit)
    ax.set_xlabel('Arcsec')
    ax.set_ylabel('Arcsec')
    ax.set_title(r'$\delta \kappa$')