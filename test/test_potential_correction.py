import autolens as al
import numpy as np
import grid_util
import potential_correction_util as pcu
import os
import pickle

current_dir, current_file_name = os.path.split(os.path.abspath(__file__))

def test_source_gradient():
    grid_data = al.Grid2D.uniform(shape_native=(200,200), pixel_scales=0.05, sub_size=1)
    xgrid_data = np.array(grid_data.slim[:,1])
    ygrid_data = np.array(grid_data.slim[:,0])

    dpis_points_source_plane = np.vstack([ygrid_data, xgrid_data]).T

    def src_func(x, y):
        return 2*x**2 + 3*y**2 + 2

    eval_points = np.array([(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)]) #[[y1,x1], [y2,x2],...]
    source_values = src_func(dpis_points_source_plane[:,1], dpis_points_source_plane[:,0])
    source_gradient = pcu.source_gradient_from(dpis_points_source_plane, source_values, eval_points, cross_size=1e-3)
    
    source_gradient_true = np.array([(0.0, 0.0), (0.0, 2.0), (3.0, 0.0), (3.0, 2.0)]) #(y,x) directional derivatie at [[y1,x1], [y2,x2],...]
    assert np.isclose(source_gradient, source_gradient_true, rtol=1e-05, atol=1e-08, equal_nan=False).all()


def test_source_gradient_matrix():
    source_gradient = np.array([(0.1, 0.2), (-0.1, 2.0), (3.0, 1.5), (3.0, 2.0)])
    source_gradient_matrix = pcu.source_gradient_matrix_from(source_gradient)

    #see eq-9 in our potential correction document
    source_gradient_matrix_true = np.array([
        [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.5, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0],
    ])

    assert np.isclose(source_gradient_matrix, source_gradient_matrix_true, rtol=1e-05, atol=1e-08, equal_nan=False).all()


def test_dpsi_gradient_operator_matrix():
    grid_data = al.Grid2D.uniform(shape_native=(10,10), pixel_scales=0.1, sub_size=1)
    xgrid_data = grid_data.native[:,:,1]
    ygrid_data = grid_data.native[:,:,0]
    rgrid = np.sqrt(xgrid_data**2 + ygrid_data**2)
    mask = rgrid>0.25
    grid_obj = grid_util.SparseDpsiGrid(mask, 0.1, shape_2d_dpsi=(5,5))

    dpsi_gradient_matrix = pcu.dpsi_gradient_operator_from(grid_obj.Hx_dpsi, grid_obj.Hy_dpsi)
    dpsi_gradient_matrix_true = np.loadtxt(f'{current_dir}/data/potential_correction/dpsi_gradient_matrix.txt')
    assert np.isclose(dpsi_gradient_matrix, dpsi_gradient_matrix_true, rtol=1e-05, atol=1e-08, equal_nan=False).all()


def test_linear_image_correction():
    #---------------load some pre-computed data.
    #seems autolens calculate the lensing potential slowly (due to the integration calculation). Check this point
    with open('test/data/potential_correction/linear_image_correction.pkl','rb') as f:
        preload_data = pickle.load(f)
        dpsi_sparse_1d = preload_data['dpsi_sparse_1d']
        pt_image_correction_true = preload_data['pt_image_correction_true']

    #--------tracer with subhalo
    lens_galaxy = al.Galaxy(
        redshift=0.2,
        mass=al.mp.EllIsothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.2,
            elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
        ),
        subhalo=al.mp.SphIsothermal(
            centre=(1.15, 0.0),
            einstein_radius=0.01,
        )
    )
    source_galaxy = al.Galaxy(
        redshift=0.6,
        bulge=al.lp.EllSersic(
            centre=(0.0, 0.0),
            elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, angle=60.0),
            intensity=0.8,
            effective_radius=0.1,
            sersic_index=1.0,
        ),
    )
    tracer_with_sub = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    #--------tracer without subhalo
    lens_galaxy = al.Galaxy(
        redshift=0.2,
        mass=al.mp.EllIsothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.2,
            elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
        ),
    )
    tracer_no_sub = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    #----------get dpsi sparse grid object
    mask_data = al.Mask2D.circular_annular(
        shape_native=(200, 200), pixel_scales=0.05, inner_radius=1.2*(1-0.4), outer_radius=1.2*(1+0.4)
    )
    grid_obj = grid_util.SparseDpsiGrid(mask_data, 0.05, (200, 200))

    #-----------------get dpsi gradient matrix, see eq.7 $D_{\psi}$ term in our document
    dpsi_gradient_matrix = pcu.dpsi_gradient_operator_from(grid_obj.Hx_dpsi, grid_obj.Hy_dpsi) 

    #-------src gradient matrix
    image_grid_vec_numpy = np.vstack([grid_obj.ygrid_data_1d, grid_obj.xgrid_data_1d]).T
    image_grid_vec_al = al.Grid2DIrregular(image_grid_vec_numpy)
    alpha_src_yx = tracer_no_sub.deflections_yx_2d_from(image_grid_vec_al)
    source_points = image_grid_vec_al - alpha_src_yx #I suppose this represents the source pixelization grid
    source_values = tracer_with_sub.galaxies[1].image_2d_from(grid=source_points) #I suppose this represents the source reconstruction

    dpsi_grid_vec_numpy = np.vstack([grid_obj.ygrid_dpsi_1d, grid_obj.xgrid_dpsi_1d]).T
    dpsi_grid_vec_al = al.Grid2DIrregular(dpsi_grid_vec_numpy)
    alpha_dpsi_yx = tracer_no_sub.deflections_yx_2d_from(dpsi_grid_vec_al)
    src_plane_dpsi_yx = dpsi_grid_vec_al - alpha_dpsi_yx #the location of dpsi grid on the source-plane, under the lens mapping relation given by macro model (without subhalo)

    source_gradient = pcu.source_gradient_from(
        source_points, 
        source_values, 
        src_plane_dpsi_yx, 
        cross_size=1e-3,
    )
    source_gradient_matrix = pcu.source_gradient_matrix_from(source_gradient) #shape: [Np, 2Np]

    #------------conformation matrix, see the C_f matrix (eq.7) in our document
    Cf_matrix = np.copy(grid_obj.map_matrix)

    #-------------linear response
    pt_image_correction = -1.0*np.matmul(
        Cf_matrix,
        np.matmul(
            source_gradient_matrix,
            np.matmul(dpsi_gradient_matrix, dpsi_sparse_1d),
        )
    )

    assert np.isclose(pt_image_correction, pt_image_correction_true, rtol=1e-05, atol=1e-08, equal_nan=False).all()


def test_rescale_psi_map():
    lens_galaxy = al.Galaxy(
        redshift=0.2,
        mass=al.mp.EllIsothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.2,
            elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
        ),
    )
    grid = al.Grid2D.uniform(shape_native=(10,10), pixel_scales=0.1, sub_size=1)
    ygrid = grid.native[:,:,0]
    xgrid = grid.native[:,:,1]

    fix_points = al.Grid2DIrregular([(0.0, 0.2), (0.1, 0.0), (0.0,-0.05)])
    fix_psi_values = lens_galaxy.potential_1d_from(fix_points)
    a_y_transfor = -0.9
    a_x_transfor = 1.1
    c_transfor = 5.0
    psi_new = np.sum(fix_points*np.tile([a_y_transfor, a_x_transfor], 3).reshape(3,-1), axis=1) + c_transfor + fix_psi_values

    a_y, a_x, c = pcu.solve_psi_rescale_factor(fix_psi_values, fix_points, psi_new)
    psi_transform = np.sum(fix_points*np.tile([a_y, a_x], 3).reshape(3,-1), axis=1) + c + psi_new 
    assert np.isclose(psi_transform, fix_psi_values, rtol=1e-05, atol=1e-08, equal_nan=False).all()


    psi_map_origin = lens_galaxy.potential_2d_from(grid).native
    psi_map_new = a_y_transfor*ygrid + a_x_transfor*xgrid + c_transfor + psi_map_origin
    psi_map_rescale = pcu.rescale_psi_map(fix_psi_values, fix_points, psi_new, psi_map_new, xgrid, ygrid)
    assert np.isclose(psi_map_rescale, psi_map_origin, rtol=1e-05, atol=1e-08, equal_nan=False).all()


def test_LinearNDInterpolatorExt():
    x = np.linspace(0,10, 11)
    y = np.linspace(0,10, 11)
    x, y = np.meshgrid(x, y)
    def linear_func(x, y):
        return 2*x + 3*y + 1
    z = linear_func(x,y)

    linear_ext_object = pcu.LinearNDInterpolatorExt(list(zip(x.flatten(),y.flatten())),z.flatten())
    # linear_ext_object = LinearNDInterpolator(list(zip(x.flatten(),y.flatten())),z.flatten())

    z_inter = linear_ext_object(x, y)
    assert np.allclose(z_inter, z, rtol=1e-05, atol=1e-08, equal_nan=False)


    x_out = np.array([-1, 10])
    y_out = np.array([-1, 10])
    x_bound = np.array([0, 9])
    y_bound = np.array([0, 9])
    z_out = linear_ext_object(x_out, y_out)
    z_bound = linear_ext_object(x_out, y_out)
    assert np.allclose(z_out, z_bound, rtol=1e-05, atol=1e-08, equal_nan=False)