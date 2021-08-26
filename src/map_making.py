import os

import numpy as np

import pysm3
import pysm3.units as units
import healpy as hp

from mk_noise_map import cov_noise_map


def dg_map(in_map, nside_in, nside_out):

    alm = hp.map2alm(in_map)
    pw = hp.pixwin(nside=nside_in, pol=True)
    ipw = 1 / pw[0]
    ppw = 1 / pw[1]
    ppw[0:2] = 0
    alm[0] = hp.almxfl(alm[0], ipw)
    alm[1] = hp.almxfl(alm[1], ppw)
    alm[2] = hp.almxfl(alm[2], ppw)

    return hp.alm2map(alm, nside=nside_out, pol=True, pixwin=True)


def make_sky_map(freq, fwhm, nside, nside_downgraded,
                 instrument_noise_freq, noise_seed, output_directory,
                 preset_strings, output_unit, output_prefix,
                 smooth_map=True, save_map=True, downgrade_map=True):
    # Check if directory exists
    if save_map:
        if output_directory == None:
            raise ValueError(
                "output_direction cannot be None. Please specify a valid path and filename")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    # Initialise sky - abstractify this into a different function
    sky = pysm3.Sky(
        nside=nside,
        # These are noise and cmb models. These should be inputs
        preset_strings=preset_strings,
        output_unit=output_unit
    )
    # Make signal and noise maps
    signal_map = sky.get_emission(freq * units.GHz)
    _, noise_map = cov_noise_map(
        sigma_I=np.array([instrument_noise_freq]),
        sigma_P=np.array([instrument_noise_freq]),
        nu=np.array([freq]),
        nside=nside,
        fwhm=fwhm/60.0,
        out_prefix=output_prefix,
        out_dir=output_directory,
        seed=noise_seed
    )  # Instrumental noise - from litebird (please cite)
    full_map = signal_map + noise_map[0] * units.uK_CMB

    # Smooth map
    if smooth_map:
        print(f"-----Smoothing map with fwhm = {fwhm}-----")
        full_map = pysm3.apply_smoothing_and_coord_transform(
            full_map, fwhm)

    # Downgrade the maps - should be optional

    nside_name = nside
    if downgrade_map:
        print(
            f"----- Downgrading map from nside = {nside} to nside = {nside_downgraded}-----")
        full_map = dg_map(full_map, nside_in=nside,
                          nside_out=nside_downgraded)
        nside_name = nside_downgraded

    # Save or return map
    if save_map:
        filename = f"nside{nside_name}_freq{int(freq)}Ghz_fwhm{int(fwhm.value)}_noise{int(instrument_noise_freq)}_map"
        file_path = os.path.join(output_directory, filename)
        hp.write_map(file_path, full_map, overwrite=True)
    else:
        return full_map


if __name__ == "__main__":
    make_sky_map(40, 60*units.arcmin, 128, 64,
                 37.42, 1111, "outputs",
                 ["d1", "s1", "c1"], "uK_CMB", "litebird",
                 True, True, True)
