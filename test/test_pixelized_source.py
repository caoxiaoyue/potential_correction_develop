#%%
import numpy as np
import autolens as al 
import pickle
import sys
sys.path.append('/home/cao/data_disk/autolens_xycao/potential_correction')
import pixelized_mass
import pixelized_source
import os 
import pytest
current_dir, current_file_name = os.path.split(os.path.abspath(__file__))

def test_pixelized_source():
    with open(f'{current_dir}/data/pixelized_source/masked_imaging.pkl','rb') as f:
        masked_imaging = pickle.load(f)

    with open(f'{current_dir}/data/pixelized_source/mass_psi_for_inversion.pkl','rb') as f:
        psi_2d = pickle.load(f)

    ygrid =  masked_imaging.grid.unmasked_grid.binned.native[:,:,0]
    xgrid =  masked_imaging.grid.unmasked_grid.binned.native[:,:,1]
    pix_mass = pixelized_mass.PixelizedMass(
        xgrid, 
        ygrid, 
        psi_2d, 
        masked_imaging.mask
    )

    pix_src = pixelized_source.PixelizedSource(masked_imaging, pixelization_shape_2d=(50,50))
    pix_src.source_inversion(pix_mass, lam_s=2.3142021886297797) #best reg strength
    ev = pix_src.evidence_from_reconstruction()
    assert 4926.626057906501==pytest.approx(ev, 1e-6)