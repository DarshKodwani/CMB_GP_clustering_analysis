from inspect import signature
import sys
from typing import overload
import pysm3
import pysm3.units as units
import numpy as np
import healpy as hp
from mk_noise_map import cov_noise_map
import matplotlib.pyplot as plt
from pathlib import Path

# PySM sky model
nside = 512
nside_dg = 128
sky = pysm3.Sky(
    nside=nside,
    preset_strings=["d1", "s1", "c1"],
    output_unit='uK_CMB'
)

# LiteBIRD instrument data

litebird_freq = [
    40,
    50,
    60,
    68,
    78,
    89,
    100,
    119,
    140,
    166,
    195,
    235,
    280,
    337,
    402
]

litebird_noise = [
    37.42,
    33.46,
    21.31,
    16.87,
    12.07,
    11.30,
    6.56,
    4.58,
    4.79,
    5.57,
    5.85,
    10.79,
    13.80,
    21.95,
    47.45
]  # Q: What is this? Value and unit?

fwhm = 60 * units.arcmin

# Get sky emission at each frequency

frequency_maps = {}
for freq in litebird_freq:
    frequency_maps[freq] = sky.get_emission(freq * units.GHz)

# Q: why are there 3 columns for each frequency? What do they represent

# Add noise, smooth and generate cov maps

seed = 1111
_, noises1 = cov_noise_map(
    sigma_I=np.array(litebird_noise),
    sigma_P=np.array(litebird_noise),
    nu=np.array(litebird_freq),
    nside=nside,
    fwhm=fwhm/60.0,
    out_prefix='full_mission_litebird',
    out_dir='../outputs',
    seed=seed
)

frequency_noise1 = {}
for i, freq in enumerate(litebird_freq):
    frequency_noise1[freq] = noises1[i]

frequency_full_sky_maps = {}
for freq in litebird_freq:
    frequency_full_sky_maps[freq] = frequency_maps[freq] + \
        frequency_noise1[freq] * units.uK_CMB


seed = 2222
_, noises2 = cov_noise_map(
    sigma_I=np.array(litebird_noise),
    sigma_P=np.array(litebird_noise),
    nu=np.array(litebird_freq),
    nside=nside,
    fwhm=fwhm/60.0,
    out_prefix='half1_mission_litebird',
    out_dir='../outputs',
    seed=seed
)

frequency_noise2 = {}
for i, freq in enumerate(litebird_freq):
    frequency_noise2[freq] = noises2[i]

frequency_half1_sky_maps = {}
for freq in litebird_freq:
    frequency_half1_sky_maps[freq] = frequency_maps[freq] + \
        frequency_noise2[freq] * units.uK_CMB

seed = 3333
_, noises3 = cov_noise_map(
    sigma_I=np.array(litebird_noise),
    sigma_P=np.array(litebird_noise),
    nu=np.array(litebird_freq),
    nside=nside,
    fwhm=fwhm/60.0,
    out_prefix='half2_mission_litebird',
    out_dir='../outputs',
    seed=seed
)

frequency_noise3 = {}
for i, freq in enumerate(litebird_freq):
    frequency_noise3[freq] = noises3[i]

frequency_half2_sky_maps = {}
for freq in litebird_freq:
    frequency_half2_sky_maps[freq] = frequency_maps[freq] + \
        frequency_noise3[freq] * units.uK_CMB


# Q: All three same the same to me? sqrt(2) is missing in the last two if that is what is needed

# Smooth maps with noise added

for i, freq in enumerate(litebird_freq):
    frequency_full_sky_maps[freq] = pysm3.apply_smoothing_and_coord_transform(
        frequency_full_sky_maps[freq], fwhm)
    frequency_half1_sky_maps[freq] = pysm3.apply_smoothing_and_coord_transform(
        frequency_half1_sky_maps[freq], fwhm)
    frequency_half2_sky_maps[freq] = pysm3.apply_smoothing_and_coord_transform(
        frequency_half2_sky_maps[freq], fwhm)

out_dir = 'ouputs'
Path(out_dir).mkdir(parents=True, exist_ok=True)


for freq in litebird_freq:
    hp.write_map(
        f"{out_dir}/full_mission_litebird_nu{freq}GHz_{fwhm.value}arcmin_nside{nside}",
        frequency_full_sky_maps[freq],
        overwrite=True
    )

    hp.write_map(
        f"{out_dir}/half1_mission_litebird_nu{freq}GHz_{fwhm.value}arcmin_nside{nside}",
        frequency_half1_sky_maps[freq],
        overwrite=True
    )

    hp.write_map(
        f"{out_dir}/half2_mission_litebird_nu{freq}GHz_{fwhm.value}arcmin_nside{nside}",
        frequency_half2_sky_maps[freq],
        overwrite=True
    )

