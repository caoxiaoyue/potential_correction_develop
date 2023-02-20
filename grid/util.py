import numpy as np
import autolens as al 
from matplotlib import pyplot as plt
import pickle


def clean_mask(in_mask):
    '''
    in_mask: the input 2d mask, a 2d bool numpy array
    out_mask: the output 2d mask after clipping

    the cliping scheme following the method in https://arxiv.org/abs/0804.2827 (see figure-11), remove the so-called "exposed pixels".
    "exposed pixels" has no ajacent pixels so that the gradient can not be calculated via the finite difference.
    '''
    out_mask = np.ones_like(in_mask).astype('bool')
    n1, n2 = in_mask.shape
    for i in range(2,n1-2): #Not range(n1), because I don't want to deal the index error related to the bound
        for j in range(2,n2-2):
            if not in_mask[i,j]:
                if_exposed_0 = True
                if_exposed_1 = True

                if ~in_mask[i-1,j] and ~in_mask[i-2,j]:
                    if_exposed_0 = False
                if ~in_mask[i+1,j] and ~in_mask[i+2,j]:
                    if_exposed_0 = False
                if ~in_mask[i-1,j] and ~in_mask[i+1,j]:
                    if_exposed_0 = False
                    
                if ~in_mask[i,j-1] and ~in_mask[i,j-2]:
                    if_exposed_1 = False
                if ~in_mask[i,j+1] and ~in_mask[i,j+2]:
                    if_exposed_1 = False
                if ~in_mask[i,j-1] and ~in_mask[i,j+1]:
                    if_exposed_1 = False

                if not (if_exposed_0 or if_exposed_1):
                    out_mask[i,j] = False

    return out_mask


def iter_clean_mask(in_mask, max_iter=50):
    niter = 0
    old_mask = np.copy(in_mask)

    while True:
        new_mask = clean_mask(old_mask)
        diff = ((new_mask.astype('int') - old_mask.astype('int')) == 0)
        niter += 1

        if niter > max_iter:
            raise Exception(f"The mask are not fully cleaned after {max_iter} iterations")
        if np.all(diff):
            break

        old_mask = np.copy(new_mask)
    
    return new_mask


def linear_weight_from_box(box_x, box_y, position=None):
    """
    The function find the linear interpolation (extrapolation) at `position`,
    given the box with corrdinates box_x and box_y
    box_x: An 4 elements list/tuple/array; save the x-coordinate of box, in the order of [top-left,top-right, bottom-left, bottom-right]
    box_y: An 4 elements list/tuple/array; similar to box_x, save the y-coordinates
    position: the location of which we estimate the linear interpolation weight; a tuple with (y,x) coordinaes, such as (1.0, 0.0),
    the location at x=0,y=1

    return an array with shape [4,], which save the linear interpolation weight in
    [top-left,top-right, bottom-left, bottom-right] order.
    """
    y, x = position
    box_size = box_x[1] - box_x[0]
    wx = (x - box_x[0])/box_size  #x direction weight 
    wy = (y - box_y[2])/box_size   #y direction weight 

    weight_top_left = (1-wx)*wy
    weight_top_right = wx*wy
    weight_bottom_left = (1-wx)*(1-wy)
    weight_bottom_right = wx*(1-wy)

    return np.array([weight_top_left, weight_top_right, weight_bottom_left, weight_bottom_right])


def diff_1st_operator_from_mask(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 1st differential operator matrix Hx and Hy.
    Hx (Hy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the 1st x/y-derivative of the unmasked data.

    dpix: pixel size in unit of arcsec.
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        #------check y-direction
        #try 1st diff first
        if unmask[i-1,j] and unmask[i+1,j]: #1st central diff
            indices_tmp = np.ravel_multi_index([(i-1, i+1), (j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/(2*step_y)   
        elif unmask[i,j] and unmask[i+1,j]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1), (j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/step_y
        elif unmask[i-1,j] and unmask[i,j]: #2nd backward diff
            indices_tmp = np.ravel_multi_index([(i-1, i), (j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/step_y  
        else:
            raise("the pixel has no neighbour to calculate the y-gradient")
        #------check x-direction  
        if unmask[i,j-1] and unmask[i,j+1]: #1st central diff
            indices_tmp = np.ravel_multi_index([(i, i), (j-1, j+1)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/(2*step_x)   
        elif unmask[i,j] and unmask[i,j+1]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i), (j, j+1)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/step_x
        elif unmask[i,j-1] and unmask[i,j]: #2nd backward diff
            indices_tmp = np.ravel_multi_index([(i, i), (j-1, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 1.0])/step_x  
        else:
            raise("the pixel has no neighbour to calculate the x-gradient")

    return Hy, Hx


def diff_2nd_operator_from_mask(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 2nd differential operator matrix Hxx and Hyy.
    Hxx (Hyy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the 2nd x/y-derivative of the unmasked data.

    dpix: pixel size in unit of arcsec.
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hxx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hyy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        #------check y-direction
        #try 2nd diff first
        if unmask[i-1,j] and unmask[i+1,j]: #2nd central diff
            indices_tmp = np.ravel_multi_index([(i-1, i, i+1), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2   
        elif unmask[i+1,j] and unmask[i+2,j]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2   
        elif unmask[i-1,j] and unmask[i-2,j]: #2nd backward diff
            indices_tmp = np.ravel_multi_index([(i, i-1, i-2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2     
        #if 2nd diff fails, just do nothing, so that the 2nd diff along y-directon is 0
        else:
            pass 
        #------check x-direction  
        #try 2nd diff;
        if unmask[i, j-1] and unmask[i, j+1]: #2nd central diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j-1, j, j+1)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2   
        elif unmask[i, j+1] and unmask[i, j+2]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2   
        elif unmask[i, j-1] and unmask[i, j-2]: #2nd backward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j-1, j-2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2  
        #if 2nd diff fails, just do nothing, so that the 2nd diff along x-directon is 0
        else:
            pass

    return Hyy, Hxx


def diff_2nd_reg_nopad_operator_from_mask_koopman(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 2nd differential operator matrix Hxx and Hyy.
    Hxx (Hyy) has a shape of [n_unmasked_pixels, n_unmasked_pixels],
    when it act on the unmasked data, generating the 2nd x/y-derivative of the unmasked data.

    dpix: pixel size in unit of arcsec.
    """
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hxx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hyy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        #------check y-direction
        #try 2nd diff first 
        if unmask[i+1,j] and unmask[i+2,j]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2 
        # elif unmask[i-1,j] and unmask[i-2,j]: #2nd backward diff
        #     pass
        #     # indices_tmp = np.ravel_multi_index([(i, i-1, i-2), (j, j, j)], unmask.shape)
        #     # indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
        #     # Hyy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2       
        else:
            # print('no-padding pixel, y-direction: ',i,j)
            pass

        #------check x-direction  
        #try 2nd diff;
        if unmask[i, j+1] and unmask[i, j+2]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2
        # elif unmask[i, j-1] and unmask[i, j-2]: #2nd backward diff
        #     pass
        #     # indices_tmp = np.ravel_multi_index([(i, i, i), (j, j-1, j-2)], unmask.shape)
        #     # indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
        #     # Hxx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2     
        else:
            # print('no-padding pixel, x-direction: ',i,j)
            pass

    diag_mat = np.diagflat([1e-8]*n_unmasked_pixels)
    Hyy += diag_mat
    Hxx += diag_mat
    # print(diag_mat,'------------------')
    return Hyy, Hxx


def diff_4th_reg_nopad_operator_from_mask(mask, dpix=1.0):
    """
    Receive a mask, use it to generate the 4th differential operator matrix Hx_4th and Hy_4th.
    different from the `diff_4th_operator_from_mask`, we only use the `forward differential` in this fucntion
    """ 
    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)
    indices_1d_unmasked = np.where(unmask.flatten())[0]
    n_unmasked_pixels = len(i_indices_unmasked) 
    Hx = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #x-direction gradient operator matrix
    Hy = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #y-direction gradient operator matrix
    step_y = -1.0*dpix #the minus sign is due to the y-coordinate decrease the pixel_size as index i along axis-0 increase 1.
    step_x = 1.0*dpix #no minus, becasue the x-coordinate increase as index j along axis-1 increase.

    for count in range(n_unmasked_pixels):
        i, j = i_indices_unmasked[count], j_indices_unmasked[count]
        #------check y-direction
        #try 4th diff first
        if unmask[i+1,j] and unmask[i+2,j] and unmask[i+3,j] and unmask[i+4,j]: #4th forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2, i+3, i+4), (j, j, j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_y**4   
        #if 4th diff fails, try 3rd diff; Note, we don't need to try 3rd central if 4th central fails
        elif unmask[i+1,j] and unmask[i+2,j] and unmask[i+3,j]: #3rd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2, i+3), (j, j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 3.0, -3.0, 1.0])/step_y**3          
        #if 3rd diff fails, try 2nd diff;   
        elif unmask[i+1,j] and unmask[i+2,j]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i+1, i+2), (j, j, j)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hy[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_y**2   
        #if 2nd diff fails, do noting, no padding to lower order!
        else:
            pass

        #------check x-direction
        #try 4th diff first
        if unmask[i, j+1] and unmask[i, j+2] and unmask[i, j+3] and unmask[i, j+4]: #4th forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i, i, i), (j, j+1, j+2, j+3, j+4)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -4.0, 6.0, -4.0, 1.0])/step_x**4   
        #if 4th diff fails, try 3rd diff; Note, we don't need to try 3rd central if 4th central fails
        elif unmask[i, j+1] and unmask[i, j+2] and unmask[i, j+3]: #3rd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i, i), (j, j+1, j+2, j+3)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([-1.0, 3.0, -3.0, 1.0])/step_x**3       
        #if 3rd diff fails, try 2nd diff;
        elif unmask[i, j+1] and unmask[i, j+2]: #2nd forward diff
            indices_tmp = np.ravel_multi_index([(i, i, i), (j, j+1, j+2)], unmask.shape)
            indices_of_indices_1d_unmasked = [np.where(indices_1d_unmasked == item)[0][0] for item in indices_tmp]
            Hx[count,indices_of_indices_1d_unmasked] = np.array([1.0, -2.0, 1.0])/step_x**2   
        #if 2nd diff fails, do noting, no padding to lower order!
        else:
            pass

    return Hy, Hx


def exp_cov_matrix_from_mask(mask, dpix=1.0, scale_length=1.0):
    """
    Receive a mask, use it to generate the exponential covariance matrix 
    dpix: pixel size in unit of arcsec.
    """
    grid = al.Grid2D.uniform(shape_native=mask.shape, pixel_scales=dpix, sub_size=1)
    xgrid = np.array(grid.native[:,:,1])
    ygrid = np.array(grid.native[:,:,0])

    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)

    n_unmasked_pixels = len(i_indices_unmasked) 
    C_mat = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #covariance matrix

    for i in range(n_unmasked_pixels):
        for j in range(n_unmasked_pixels):
            xi = xgrid[i_indices_unmasked[i], j_indices_unmasked[i]]
            yi = ygrid[i_indices_unmasked[i], j_indices_unmasked[i]]
            xj = xgrid[i_indices_unmasked[j], j_indices_unmasked[j]]
            yj = ygrid[i_indices_unmasked[j], j_indices_unmasked[j]]
            d_ij = np.sqrt((xi-xj)**2 + (yi-yj)**2) #distance between the pixel i and j
            C_mat[i, j] = np.exp(-1.0*d_ij/scale_length)

    return C_mat


def gauss_cov_matrix_from_mask(mask, dpix=1.0, scale_length=1.0):
    """
    Receive a mask, use it to generate the gaussian covariance matrix
    dpix: pixel size in unit of arcsec.
    """
    grid = al.Grid2D.uniform(shape_native=mask.shape, pixel_scales=dpix, sub_size=1)
    xgrid = np.array(grid.native[:,:,1])
    ygrid = np.array(grid.native[:,:,0])

    unmask = ~mask
    i_indices_unmasked, j_indices_unmasked = np.where(unmask)

    n_unmasked_pixels = len(i_indices_unmasked) 
    C_mat = np.zeros((n_unmasked_pixels, n_unmasked_pixels)) #covariance matrix

    for i in range(n_unmasked_pixels):
        for j in range(n_unmasked_pixels):
            xi = xgrid[i_indices_unmasked[i], j_indices_unmasked[i]]
            yi = ygrid[i_indices_unmasked[i], j_indices_unmasked[i]]
            xj = xgrid[i_indices_unmasked[j], j_indices_unmasked[j]]
            yj = ygrid[i_indices_unmasked[j], j_indices_unmasked[j]]
            d_ij = np.sqrt((xi-xj)**2 + (yi-yj)**2) #distance between the pixel i and j
            C_mat[i, j] = np.exp(-1.0*d_ij**2/2.0/scale_length**2)

    return C_mat

        
if __name__ == '__main__':
    pass