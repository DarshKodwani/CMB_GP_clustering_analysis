from src.map_making import initialise_sky, make_sky_map
from src.compute_clusters import get_noiseless_clusters
import pysm3.units as units
import time
import json


with open('inputs.json') as in_file:
    inputs = json.load(in_file)

frequencies = inputs['frequencies']
noise = inputs['noise']
nside = inputs['nside']
nside_dg = inputs['nside_dg']
preset_strings = inputs['preset_strings']
sky_unit = inputs['sky_unit']
fwhm = inputs['fwhm'] * units.arcmin
noise_seed = inputs['noise_seed']
output_directory = inputs['output_directory']
output_prefix = inputs['output_prefix']

filenames = []
for freq, instrument_noise in zip(frequencies, noise):
    sky = initialise_sky(
        nside,
        preset_strings=preset_strings,
        output_unit=sky_unit
    )
    filenames.append(
        make_sky_map(
            sky,
            freq=freq,
            fwhm=fwhm,
            nside=nside,
            nside_downgraded=nside_dg,
            instrument_noise_freq=instrument_noise,
            noise_seed=noise_seed,
            output_directory=output_directory,
            output_prefix=output_prefix
        )
    )

start = time.time()
sky_regions = get_noiseless_clusters(filenames=filenames, frequencies=frequencies)
print(f"Time taken = {time.time() - start}")
