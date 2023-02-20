To use this code, you need:
step-1: install the conda enviroment, then use the command `conda env create -f potential_correction.yaml` to install dependencies
step-2: activate the new envroment you create, `conda activate AlPT`
step-3: replace the `/home/cao/anaconda3/envs/AlPT/lib/python3.8/site-packages/autoarray/operators/convolver.py` (this the path of autoarray package on my PC, yours might be different) with the `convolver.py` in the current folder
step-4: `conda develop the_path_of_potential_correction_code_folder`. On my PC, i.e, `conda develop /home/cao/data_disk/autolens_xycao/potential_correction_for_sam`
step-5: you can then run codes under the exmaple folder:
0_simulate_data.py: simulate a lens with a sis subhalo around the Einstein radius of the main lens
1_macro_model.ipynb: run a pesudo macro-mass model, the best-fit mass model of this step is feed as the initial staring point for the potential correction
2_potential_correction.py: run the potential correction, the result is output to `result` folder.

Subhalo_potential_correction.pdf is the team document of potential correction, I hope it is useful for some conceptions
Good luck and have fun :)
