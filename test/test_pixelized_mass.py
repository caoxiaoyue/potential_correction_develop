#%%
import numpy as np
import autolens as al 
import pixelized_mass

def test_pixelized_mass():
    grid = al.Grid2D.uniform(
        shape_native=(200, 200),
        pixel_scales=0.05,
        sub_size=2,  
    )
    ygrid = grid.binned.native[:,:,0]
    xgrid = grid.binned.native[:,:,1]
    
    lens_galaxy = al.Galaxy(
        redshift=0.2,
        mass=al.mp.SphIsothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.2,
        )
    )
    
    psi_2d_true = lens_galaxy.potential_2d_from(grid).binned.native
    alpha_2d_true = lens_galaxy.deflections_yx_2d_from(grid).binned.native
    alphay_2d_true = alpha_2d_true[:,:,0]
    alphax_2d_true = alpha_2d_true[:,:,1]
    kappa_2d_true = lens_galaxy.convergence_2d_from(grid).binned.native
    
    mask = al.Mask2D.circular_annular(
        shape_native=(200, 200), pixel_scales=0.05, inner_radius=1.2*(1-0.4), outer_radius=1.2*(1+0.4)
    )
    
    psi_1d_true = psi_2d_true[~mask]
    alphay_1d_true = alphay_2d_true[~mask]
    alphax_1d_true = alphax_2d_true[~mask]
    kappa_1d_true = kappa_2d_true[~mask]
    
    pix_mass = pixelized_mass.PixelizedMass(xgrid, ygrid, psi_2d_true, mask)
    
    xgrid_1d = xgrid[~mask]
    ygrid_1d = ygrid[~mask]
    points = np.array(list(zip(ygrid_1d, xgrid_1d))) #use autolens [(y1,x1),(y2,x2),...] order
    psi_1d = pix_mass.eval_psi_at(points)
    alphay_1d, alphax_1d = pix_mass.eval_alpha_yx_at(points)
    kappa_1d = pix_mass.eval_kappa_at(points)
    
    np.allclose(kappa_1d, kappa_1d_true, rtol=0.15, atol=1e-8) #TODO, need improve
    np.allclose(psi_1d, psi_1d_true, rtol=0.01, atol=1e-8)
    np.allclose(alphax_1d, alphax_1d_true, rtol=0.01, atol=1e-8)
    np.allclose(alphay_1d, alphay_1d_true, rtol=0.01, atol=1e-8)
