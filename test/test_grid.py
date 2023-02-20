import autolens as al
import numpy as np
from grid import util as grid_util
from grid.sparse_grid import SparseDpsiGrid
import os
import pickle

current_dir, current_file_name = os.path.split(os.path.abspath(__file__))

def test_clean_data_mask():
    data_mask = [
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,1,1],
        [1,1,0,0,0,0,1,1],
        [1,1,0,0,0,0,1,1],
        [1,1,1,1,0,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
    ]
    data_mask = np.array(data_mask, dtype='bool')
    out_mask = grid_util.clean_mask(data_mask)
    print(out_mask)
    correct_clean_mask = [
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,1,1],
        [1,1,0,0,0,0,1,1],
        [1,1,0,0,0,0,1,1],
        [1,1,1,1,1,1,1,1], #clean mask should remove the exposed pixel with numpy array indices (5,4)
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
    ]
    correct_clean_mask = np.array(correct_clean_mask, dtype='bool')

    assert (out_mask==correct_clean_mask).all()


def test_linear_weight_from_box():
    box_x = [1.0, 2.0, 1.0, 2.0] #[top-left,top-right, bottom-left, bottom-right] order
    box_y = [2.0, 2.0, 1.0, 1.0]

    point_intp_1 = [1.7, 1.2] #Note, this is in [y,x] order, not the [x,y] order
    true_weight = np.array([0.56, 0.14, 0.24, 0.06])
    func_weight = grid_util.linear_weight_from_box(box_x, box_y, position=point_intp_1)
    assert np.isclose(func_weight, true_weight, rtol=1e-05, atol=1e-08, equal_nan=False).all()

    point_intp_2 = [1.0, 1.0] #Note, this is in [y,x] order, not the [x,y] order
    true_weight = np.array([0.0, 0.0, 1.0, 0.0])
    func_weight = grid_util.linear_weight_from_box(box_x, box_y, position=point_intp_2)
    assert np.isclose(func_weight, true_weight, rtol=1e-05, atol=1e-08, equal_nan=False).all()

    point_intp_3 = [2.1, 0.9] #test extrapolation
    true_weight = np.array([1.21, -0.11, -0.11, 0.01])
    func_weight = grid_util.linear_weight_from_box(box_x, box_y, position=point_intp_3)
    assert np.isclose(func_weight, true_weight, rtol=1e-05, atol=1e-08, equal_nan=False).all()


def test_gradient_operator_from_mask():
    grid_data = al.Grid2D.uniform(shape_native=(100,100), pixel_scales=0.1, sub_size=1)
    xgrid_data = grid_data.native[:,:,1]
    ygrid_data = grid_data.native[:,:,0]
    rgrid = np.sqrt(xgrid_data**2 + ygrid_data**2)
    annular_mask = (rgrid>4.0) #np.logical_or(rgrid<1.0, rgrid>4.0)
    grid_obj = SparseDpsiGrid(annular_mask, 0.1, shape_2d_dpsi=(50,50))

    def linear_func(xgrid, ygrid):
        return 2*xgrid + 3*ygrid + 1

    data_image1d_true = linear_func(grid_obj.xgrid_data_1d, grid_obj.ygrid_data_1d)
    Hy, Hx = grid_util.diff_1st_operator_from_mask(grid_obj.mask_data, grid_obj.dpix_data)
    y_gradient = np.matmul(Hy, data_image1d_true)
    x_gradient = np.matmul(Hx, data_image1d_true)

    assert np.isclose(y_gradient, 3, rtol=1e-05, atol=1e-08, equal_nan=False).all()
    assert np.isclose(x_gradient, 2, rtol=1e-05, atol=1e-08, equal_nan=False).all()


class TestSparseDpsiGrid:
    def test_2d_sparse_grid(self):
        mask = np.zeros((10,10)).astype('bool')
        mask[0:2, :] = True #I don't handle the image bound currently, so removing boundary region here to avoid code error due to the `index overflow`.
        mask[-2:, :] = True
        mask[:, 0:2] = True
        mask[:, -2:] = True

        grid_obj = SparseDpsiGrid(mask, 0.1, shape_2d_dpsi=(5,5))
        true_sparse_grid = al.Grid2D.uniform(shape_native=(5,5), pixel_scales=0.2)
        assert grid_obj.dpix_dpsi == 0.2
        assert (grid_obj.xgrid_dpsi==np.array(true_sparse_grid.native[:,:,1])).all()
        assert (grid_obj.ygrid_dpsi==np.array(true_sparse_grid.native[:,:,0])).all()
        assert grid_obj.image_bound==[-0.5,0.5,-0.5,0.5]


    def test_1d_data_dpsi_grid(self):
        mask = np.loadtxt(f'{current_dir}/data/grid/mask_data.txt').astype('bool')
        grid_obj = SparseDpsiGrid(mask, 0.1, shape_2d_dpsi=(12,12))

        true_mask_dpsi = np.loadtxt(f'{current_dir}/data/grid/mask_dpsi.txt').astype('bool')
        assert (grid_obj.mask_dpsi==true_mask_dpsi).all()

        true_indices_1d_dpsi = np.loadtxt(f'{current_dir}/data/grid/indices_1d_dpsi.txt').astype('int64')
        assert (grid_obj.indices_1d_dpsi==true_indices_1d_dpsi).all()

        true_xgrid_dpsi_1d = np.loadtxt(f'{current_dir}/data/grid/xgrid_dpsi_1d.txt').astype('float')
        assert np.isclose(grid_obj.xgrid_dpsi_1d, true_xgrid_dpsi_1d, rtol=1e-05, atol=1e-08, equal_nan=False).all()

        true_ygrid_dpsi_1d = np.loadtxt(f'{current_dir}/data/grid/ygrid_dpsi_1d.txt').astype('float')
        assert np.isclose(grid_obj.ygrid_dpsi_1d, true_ygrid_dpsi_1d, rtol=1e-05, atol=1e-08, equal_nan=False).all()

        true_mask_data = np.loadtxt(f'{current_dir}/data/grid/mask_data.txt').astype('bool')
        assert (grid_obj.mask_data==true_mask_data).all()

        true_indices_1d_data = np.loadtxt(f'{current_dir}/data/grid/indices_1d_data.txt').astype('int64')
        assert (grid_obj.indices_1d_data==true_indices_1d_data).all()

        true_xgrid_data_1d = np.loadtxt(f'{current_dir}/data/grid/xgrid_data_1d.txt').astype('float')
        assert np.isclose(grid_obj.xgrid_data_1d, true_xgrid_data_1d, rtol=1e-05, atol=1e-08, equal_nan=False).all()

        true_ygrid_data_1d = np.loadtxt(f'{current_dir}/data/grid/ygrid_data_1d.txt').astype('float')
        assert np.isclose(grid_obj.ygrid_data_1d, true_ygrid_data_1d, rtol=1e-05, atol=1e-08, equal_nan=False).all()


    def test_sparse_box_grid(self):
        mask = np.loadtxt(f'{current_dir}/data/grid/mask_data.txt').astype('bool')
        grid_obj = SparseDpsiGrid(mask, 0.1, shape_2d_dpsi=(12,12))  

        true_sparse_box_xcenter = np.loadtxt(f'{current_dir}/data/grid/sparse_box_xcenter.txt').astype('float')
        assert np.isclose(grid_obj.sparse_box_xcenter, true_sparse_box_xcenter, rtol=1e-05, atol=1e-08, equal_nan=False).all()

        true_sparse_box_ycenter = np.loadtxt(f'{current_dir}/data/grid/sparse_box_ycenter.txt').astype('float')
        assert np.isclose(grid_obj.sparse_box_ycenter, true_sparse_box_ycenter, rtol=1e-05, atol=1e-08, equal_nan=False).all()

        true_mask_sparse_box = np.loadtxt(f'{current_dir}/data/grid/mask_sparse_box.txt').astype('bool')
        assert (grid_obj.mask_sparse_box==true_mask_sparse_box).all()

        true_indices_1d_sparse_box = np.loadtxt(f'{current_dir}/data/grid/indices_1d_sparse_box.txt').astype('int64')
        assert (grid_obj.indices_1d_sparse_box==true_indices_1d_sparse_box).all()

        true_sparse_box_xcenter_1d = np.loadtxt(f'{current_dir}/data/grid/sparse_box_xcenter_1d.txt').astype('float')
        assert np.isclose(grid_obj.sparse_box_xcenter_1d, true_sparse_box_xcenter_1d, rtol=1e-05, atol=1e-08, equal_nan=False).all()

        true_sparse_box_ycenter_1d = np.loadtxt(f'{current_dir}/data/grid/sparse_box_ycenter_1d.txt').astype('float')
        assert np.isclose(grid_obj.sparse_box_ycenter_1d, true_sparse_box_ycenter_1d, rtol=1e-05, atol=1e-08, equal_nan=False).all()


    def test_pair_data_dpsi_pixel(self):
        grid_data = al.Grid2D.uniform(shape_native=(10,10), pixel_scales=0.1, sub_size=1)
        xgrid_data = grid_data.native[:,:,1]
        ygrid_data = grid_data.native[:,:,0]
        rgrid = np.sqrt(xgrid_data**2 + ygrid_data**2)
        mask = (rgrid>0.25)
        grid_obj = SparseDpsiGrid(mask, 0.1, shape_2d_dpsi=(10,10))

        with open('test/data/grid/data_dpsi_pair_info.pkl','rb') as f:
            true_data_dpsi_pair_info = pickle.load(f).astype('float')
        assert np.isclose(grid_obj.data_dpsi_pair_info, true_data_dpsi_pair_info, rtol=1e-05, atol=1e-08, equal_nan=False).all()


    def test_dpsi2data_mapping(self):
        grid_data = al.Grid2D.uniform(shape_native=(100,100), pixel_scales=0.1, sub_size=1)
        xgrid_data = grid_data.native[:,:,1]
        ygrid_data = grid_data.native[:,:,0]
        rgrid = np.sqrt(xgrid_data**2 + ygrid_data**2)
        annular_mask = (rgrid>4.0) #np.logical_or(rgrid<1.0, rgrid>4.0)
        grid_obj = SparseDpsiGrid(annular_mask, 0.1, shape_2d_dpsi=(50,50))

        def linear_func(xgrid, ygrid):
            return 2*xgrid + 3*ygrid

        data_image1d_true = linear_func(grid_obj.xgrid_data_1d, grid_obj.ygrid_data_1d)
        dpsi_image1d_true = linear_func(grid_obj.xgrid_dpsi_1d, grid_obj.ygrid_dpsi_1d)
        data_image1d_map = np.matmul(grid_obj.map_matrix, dpsi_image1d_true)
        
        assert np.isclose(data_image1d_map, data_image1d_true, rtol=1e-05, atol=1e-08, equal_nan=False).all()

    
    def test_get_gradient_operator(self):
        grid_data = al.Grid2D.uniform(shape_native=(100,100), pixel_scales=0.1, sub_size=1)
        xgrid_data = grid_data.native[:,:,1]
        ygrid_data = grid_data.native[:,:,0]
        rgrid = np.sqrt(xgrid_data**2 + ygrid_data**2)
        annular_mask = (rgrid>4.0) #np.logical_or(rgrid<1.0, rgrid>4.0)
        grid_obj = SparseDpsiGrid(annular_mask, 0.1, shape_2d_dpsi=(50,50))

        def linear_func(xgrid, ygrid):
            return 2*xgrid + 3*ygrid + 1

        data_image1d_true = linear_func(grid_obj.xgrid_data_1d, grid_obj.ygrid_data_1d)
        y_gradient_data = np.matmul(grid_obj.Hy_data, data_image1d_true)
        x_gradient_data = np.matmul(grid_obj.Hx_data, data_image1d_true)

        dpsi_image1d_true = linear_func(grid_obj.xgrid_dpsi_1d, grid_obj.ygrid_dpsi_1d)
        y_gradient_dpsi = np.matmul(grid_obj.Hy_dpsi, dpsi_image1d_true)
        x_gradient_dpsi = np.matmul(grid_obj.Hx_dpsi, dpsi_image1d_true)

        assert np.isclose(y_gradient_data, 3, rtol=1e-05, atol=1e-08, equal_nan=False).all()
        assert np.isclose(x_gradient_data, 2, rtol=1e-05, atol=1e-08, equal_nan=False).all()
        assert np.isclose(y_gradient_dpsi, 3, rtol=1e-05, atol=1e-08, equal_nan=False).all()
        assert np.isclose(x_gradient_dpsi, 2, rtol=1e-05, atol=1e-08, equal_nan=False).all()

    
    def test_diff_4th_dpsi_operator(self):
        grid_data = al.Grid2D.uniform(shape_native=(20,20), pixel_scales=0.5, sub_size=1)
        xgrid_data = grid_data.native[:,:,1]
        ygrid_data = grid_data.native[:,:,0]
        rgrid = np.sqrt(xgrid_data**2 + ygrid_data**2)
        annular_mask = np.logical_or(rgrid<1.5, rgrid>4.0)
        grid_obj = SparseDpsiGrid(annular_mask, 0.5, shape_2d_dpsi=(10,10))

        true_Hy_dpsi_4th = np.loadtxt(f'{current_dir}/data/grid/Hy_dpsi_4th_reg.txt')
        true_Hx_dpsi_4th = np.loadtxt(f'{current_dir}/data/grid/Hx_dpsi_4th_reg.txt')

        assert np.isclose(grid_obj.Hy_dpsi_4th_reg, true_Hy_dpsi_4th, rtol=1e-05, atol=1e-08, equal_nan=False).all()
        assert np.isclose(grid_obj.Hx_dpsi_4th_reg, true_Hx_dpsi_4th, rtol=1e-05, atol=1e-08, equal_nan=False).all()   


    def test_diff_2nd_dpsi_operator(self):
        grid_data = al.Grid2D.uniform(shape_native=(20,20), pixel_scales=0.5, sub_size=1)
        xgrid_data = grid_data.native[:,:,1]
        ygrid_data = grid_data.native[:,:,0]
        rgrid = np.sqrt(xgrid_data**2 + ygrid_data**2)
        annular_mask = np.logical_or(rgrid<1.5, rgrid>4.0)
        grid_obj = SparseDpsiGrid(annular_mask, 0.5, shape_2d_dpsi=(10,10))

        true_Hy_dpsi_2nd = np.loadtxt(f'{current_dir}/data/grid/Hy_dpsi_2nd_reg.txt')
        true_Hx_dpsi_2nd = np.loadtxt(f'{current_dir}/data/grid/Hx_dpsi_2nd_reg.txt')

        assert np.isclose(grid_obj.Hy_dpsi_2nd_reg, true_Hy_dpsi_2nd, rtol=1e-05, atol=1e-08, equal_nan=False).all()
        assert np.isclose(grid_obj.Hx_dpsi_2nd_reg, true_Hx_dpsi_2nd, rtol=1e-05, atol=1e-08, equal_nan=False).all()   