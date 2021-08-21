import sys
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
] # Q: What is this? Value and unit?

fwhm = 60 * units.arcmin

# Get sky emission at each frequency

frequency_maps = {}
for freq in litebird_freq:
    frequency_maps[freq] = sky.get_emission(freq * units.GHz)

