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
    present_strings=["d1", "s1", "c1"],
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
] * units.GHz

fwhm = 60 * units.arcmin