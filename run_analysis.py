from src.map_making import initialise_sky, make_sky_map
from src.compute_clusters import get_noiseless_clusters
import pysm3.units as units
import time

litebird_freq = [40, 50, 60, 68, 78, 89, 100, 119, 140, 166, 195, 235,
                 280, 337, 402]

litebird_noise = [37.42, 33.46, 21.31, 16.87, 12.07, 11.30, 6.56, 4.58,
                  4.79, 5.57, 5.85, 10.79, 13.80, 21.95, 47.45]

nside = 128
nside_dg = 64
preset_strings = ["d1", "s1", "c1"]
sky_unit = "uK_CMB"
fwhm = 60*units.arcmin
nside = 128
nside_dg = 64
noise_seed = 1111
output_directory = "outputs"
output_prefix = 'litebird'


filenames = []
for freq, instrument_noise in zip(litebird_freq, litebird_noise):
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
sky_regions = get_noiseless_clusters(filenames=filenames, frequencies=litebird_freq)
print(f"Time taken = {time.time() - start}")