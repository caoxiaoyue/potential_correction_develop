from os import path
import autolens as al
import autolens.plot as aplt


dataset_name = "sie_sis_sersic"


dataset_path = path.join('dataset',dataset_name)


grid = al.Grid2DIterate.uniform(
    shape_native=(200, 200),
    pixel_scales=0.05,
    fractional_accuracy=0.9999,
    sub_steps=[2, 4, 8, 16, 24],
)


psf = al.Kernel2D.from_gaussian(
    shape_native=(11, 11), sigma=0.05, pixel_scales=grid.pixel_scales
)


simulator = al.SimulatorImaging(
    exposure_time=1200.0, psf=psf, background_sky_level=0.1, add_poisson_noise=True, noise_seed=1
)


lens_galaxy = al.Galaxy(
    redshift=0.2,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.2,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    subhalo=al.mp.SphIsothermal(
        centre=(1.25, 0.0),
        einstein_radius=0.1,
    )
    # shear=al.mp.ExternalShear(elliptical_comps=(0.05, 0.05)),
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

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
imaging dataset.
"""
imaging = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
Lets plot the simulated `Imaging` dataset before we output it to fits.
"""
imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
Output the simulated dataset to the dataset path as .fits files.
"""
imaging.output_to_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    overwrite=True,
)

"""
Output a subplot of the simulated dataset, the image and a subplot of the `Tracer`'s quantities to the dataset path 
as .png files.
"""
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

imaging_plotter = aplt.ImagingPlotter(imaging=imaging, mat_plot_2d=mat_plot_2d)
imaging_plotter.subplot_imaging()
imaging_plotter.figures_2d(image=True)

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot_2d)
tracer_plotter.subplot_tracer()

"""
Pickle the `Tracer` in the dataset folder, ensuring the true `Tracer` is safely stored and available if we need to 
check how the dataset was simulated in the future. 

This will also be accessible via the `Aggregator` if a model-fit is performed using the dataset.
"""
tracer.output_to_json(file_path=path.join(dataset_path, "tracer.json"))

"""
The dataset can be viewed in the folder `autolens_workspace/imaging/no_lens_light/mass_sie__source_sersic`.
"""

solver = al.PointSolver(
    grid=grid,
    use_upscaling=True,
    upscale_factor=2,
    pixel_scale_precision=0.001,
    distance_to_source_centre=0.001,
)

positions = solver.solve(
    lensing_obj=tracer,
    source_plane_coordinate=(0.0, 0.0),
)
positions.output_to_json(path.join(dataset_path, "positions.json"), overwrite=True)


import numpy as np
def findOverlap(positions, min_distance):
    """
    finds overlapping solutions, deletes multiples and deletes non-solutions and if it is not a solution, deleted as well
    """
    x_mins = positions[:,1]
    y_mins = positions[:,0]
    n = len(x_mins)
    idex = []
    for i in range(n):
        if i == 0:
            pass
        else:
            for j in range(0, i):
                if abs(x_mins[i] - x_mins[j] < min_distance and abs(y_mins[i] - y_mins[j]) < min_distance):
                    idex.append(i)
                    break
    x_mins = np.delete(x_mins, idex, axis=0)
    y_mins = np.delete(y_mins, idex, axis=0)
    return np.vstack([y_mins, x_mins]).T

positions = findOverlap(positions, 0.1)
positions = [tuple(value) for value in positions]
import json
with open(path.join(dataset_path, "positions.json"), 'w') as f:
    json.dump(positions, f, indent=4)
