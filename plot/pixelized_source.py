from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import griddata
import copy

def visualize_source(
    points, 
    values, 
    ax=None, 
    enlarge_factor=1.1,
    npixels=100,
    cmap='jet',
):
    """
    Points is defined as autolens [(y1,x1), (y2,x2), ...] order
    """
    points = np.asarray(points)
    points  = points[:, ::-1] #change to numpy/scipy api format -- [(x1,y2), (x2,y2),...] order

    half_width = max(np.abs(points.min()), np.abs(points.max()))
    half_width *= enlarge_factor
    # extent = [-1.0*half_width, half_width, -1.0*half_width, half_width]

    coordinate_1d, dpix = np.linspace(-1.0*half_width, half_width, npixels, endpoint=True, retstep=True)
    xgrid, ygrid = np.meshgrid(coordinate_1d, coordinate_1d)
    extent = [-1.0*half_width-0.5*dpix, half_width+0.5*dpix, -1.0*half_width-0.5*dpix, half_width+0.5*dpix]

    source_image = griddata(points, values, (xgrid, ygrid), method='linear', fill_value=0.0)

    im = ax.imshow(source_image, origin='lower', extent=extent, cmap=cmap) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def visualize_unmasked_1d_image(
    unmasked_1d_image, 
    mask, 
    dpix, 
    ax=None, 
    cmap='jet',
    origin='upper',
):
    """
    mask: the 2d data mask
    """
    cmap = copy.copy(plt.get_cmap(cmap))
    cmap.set_bad(color='white')

    unmasked_2d_image = np.zeros_like(mask, dtype='float')
    unmasked_2d_image[~mask] = unmasked_1d_image

    half_width = len(mask)*0.5*dpix
    extent = [-1.0*half_width, half_width, -1.0*half_width, half_width]

    unmasked_2d_image = np.ma.masked_array(unmasked_2d_image, mask=mask)

    im = ax.imshow(unmasked_2d_image, origin=origin, extent=extent, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    coordinate_1d = np.arange(len(mask)) * dpix
    coordinate_1d = coordinate_1d - np.mean(coordinate_1d)
    xgrid, ygrid = np.meshgrid(coordinate_1d, coordinate_1d)
    rgrid = np.sqrt(xgrid**2 + ygrid**2)
    limit = np.max(rgrid[~mask])
    ax.set_xlim(-1.0*limit, limit)
    ax.set_ylim(-1.0*limit, limit)



