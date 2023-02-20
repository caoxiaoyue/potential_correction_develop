import numpy as np
from grid import util as grid_util
from scipy.spatial import Delaunay
from potential_correction_util import LinearNDInterpolatorExt

class PixelizedMass(object):
    """
    A mass model defined on the pixelized grid
    """

    def __init__(self, xgrid, ygrid, psi_map, mask, Hx=None, Hy=None, Hxx=None, Hyy=None):
        """
        xgrid: 2d x-grid
        ygrid: 2d y-grid
        psi_map: 2d lens potential values on x/y grid
        mask: the 2d image mask, a bool arrary. Pixels with true values are masked
        Hx,Hy: the gradient operator matrix, see `get_gradient_operator_data` in grid_util.py
        """
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.psi_map = psi_map
        self.mask = mask
        self.dpix = np.abs(xgrid[0,1] - xgrid[0,0])
        self.psi_map[self.mask] = 0.0 #always ensure the psi values in masked region are 0

        self.psi_1d = self.psi_map[~self.mask]
        self.xgrid_1d = self.xgrid[~self.mask]
        self.ygrid_1d = self.ygrid[~self.mask]
        self.tri = Delaunay(list(zip(self.xgrid_1d, self.ygrid_1d)))
        
        if (Hx is None) or (Hy is None):
            self.Hy, self.Hx = grid_util.diff_1st_operator_from_mask(self.mask, self.dpix)
        else:
            self.Hy, self.Hx = Hy, Hx

        if (Hxx is None) or (Hyy is None):
            self.Hyy, self.Hxx = grid_util.diff_2nd_operator_from_mask(self.mask, self.dpix)
        else:
            self.Hyy, self.Hxx = Hyy, Hxx
        self.hamiltonian_operator = self.Hyy + self.Hxx

        self.alpha_kappa_map_from_psi_map()
        self.construct_interpolator()


    def deflecton_map_from_psi_map(self):
        self.alphay_1d = np.matmul(self.Hy, self.psi_1d)
        self.alphax_1d = np.matmul(self.Hx, self.psi_1d)


    def convergence_map_from_psi_map(self):
        self.kappa_1d =  0.5 * np.matmul(self.hamiltonian_operator, self.psi_1d)


    def alpha_kappa_map_from_psi_map(self):
        self.deflecton_map_from_psi_map()
        self.convergence_map_from_psi_map()


    def construct_interpolator(self):
        self.interp_psi = LinearNDInterpolatorExt(self.tri, self.psi_1d)
        self.interp_alphax = LinearNDInterpolatorExt(self.tri, self.alphax_1d)
        self.interp_alphay = LinearNDInterpolatorExt(self.tri, self.alphay_1d)
        self.interp_kappa = LinearNDInterpolatorExt(self.tri, self.kappa_1d)

    
    def eval_psi_at(self, points=None):
        """
        points: an 2d array. [(y1,x1), (y2,x2), (y3,x3),...]
        """
        return self.interp_psi(points[:,1], points[:,0])


    def eval_alpha_yx_at(self, points=None):
        """
        points: an 2d array. [(y1,x1), (y2,x2), (y3,x3),...]
        return: deflection-y and deflection-x
        """
        return self.interp_alphay(points[:,1], points[:,0]), self.interp_alphax(points[:,1], points[:,0])


    def eval_kappa_at(self, points=None):
        """
        points: an 2d array. [(y1,x1), (y2,x2), (y3,x3),...]
        """
        return self.interp_kappa(points[:,1], points[:,0])