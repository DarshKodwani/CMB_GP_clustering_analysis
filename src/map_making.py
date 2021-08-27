import os

import numpy as np

import pysm3
import pysm3.units as units
import healpy as hp
from pathlib import Path


h = 6.62607004*10**(-34)
kb = 1.38064852*10**(-23)
Tcmb = 2.7255

def Kcmb2Krj(nu):

    gamma = h / (kb * Tcmb)
    return (gamma * nu * 10**9)**2 * np.exp(gamma * nu * 10**9) / (np.exp(gamma * nu * 10**9) - 1)**2


def cov_noise_map(sigma_I, sigma_P, nu, nside, fwhm, out_dir, out_prefix, seed=1111, hits_map=None,
                  red_noise_tilt=None, lknee=None):

    """

    Function for generating uniform white noise sensitivity maps.

    Inputs
    ------
    sigma_I: I sigma levels at corresponding frequencies. Units should be uK-arcmin - numpy.ndarray.
    sigma_P: P sigma levels at corresponding frequencies. Units should be uK-arcmin - numpy.ndarray.
    nu: Corresponding map frequencies in GHz - numpy.ndarray
    nside: Map nside - float.
    fwhm: FWHM map was smoothed to (degrees) - float.
    out_dir: Output directory for the sensitivity maps - str.
    out_prefix: Output prefix for the sensitivity maps - str.
    seed: Random number seed - int.
    hits_map: Optional hits map, by which we weight the noise map - str.
    red_noise_tilt: Optional red noise tilt at each frequency, if using a correlated noise model - numpy.ndarray.
    lknee: Optional knee multiple for red noise power spectrum at each frequency - numpy.ndarray.

    Returns
    -------
    cov: List containing [II, QQ, UU] maps as elements - list.
    Saves the covariance maps.

    """

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    pix_area = hp.nside2pixarea(nside, degrees=True)    
    pix_area = pix_area*60.0**2 # Pixel area in arcmin^2
    sigma_I = np.sqrt(sigma_I**2/pix_area)
    sigma_P = np.sqrt(sigma_P**2/pix_area)

    omega = np.pi*np.radians(fwhm)**2/(4*np.log(2))
    eta = hp.nside2pixarea(nside)/(2*omega)
    beam_sigma = np.radians(fwhm)/(2*np.sqrt(2*np.log(2)))

    cov = []
    noises = []
    
    if hits_map is not None:

        hits = hp.read_map(hits_map)
    
    for i in range (0, len(sigma_I)):

        cov_II = eta * np.ones(hp.nside2npix(nside)) * sigma_I[i]**2
        cov_QQ = eta * np.ones(hp.nside2npix(nside)) * sigma_P[i]**2
        cov_UU = eta * np.ones(hp.nside2npix(nside)) * sigma_P[i]**2

        if hits_map is not None:

            cov_II = cov_II / (hits / np.amax(hits))
            cov_QQ = cov_QQ / (hits / np.amax(hits))
            cov_UU = cov_UU / (hits / np.amax(hits))
            cov_II[hits==0] = hp.UNSEEN
            cov_QQ[hits==0] = hp.UNSEEN
            cov_UU[hits==0] = hp.UNSEEN
            
        diag_cov = np.array([cov_II, cov_QQ, cov_UU])
        cov.append(diag_cov)

        freq_str = nu[i]
        print('Saving {0}/{1}_nu{2}GHz_cov_nside{3:04d}.fits'.format(out_dir, out_prefix, freq_str, nside))
        hp.write_map('{0}/{1}_nu{2}GHz_cov_nside{3:04d}.fits'.format(out_dir, out_prefix, freq_str, nside), diag_cov, overwrite=True)
        
        if red_noise_tilt is None or lknee is None:

            print('Generating white noise realisations - note these are not beam smoothed.')

            np.random.seed(seed)
            noise_I = np.random.normal(scale=sigma_I[i], size=len(cov_II))
            noise_Q = np.random.normal(scale=sigma_P[i], size=len(cov_QQ))
            noise_U = np.random.normal(scale=sigma_P[i], size=len(cov_UU))

            noises.append(np.array([noise_I, noise_Q, noise_U]))
            
            hp.write_map('{0}/{1}_nu{2}GHz_white_noise_IQU_seed{3}_nside{4:04d}.fits'.format(out_dir, out_prefix, freq_str, int(seed), nside),
                         np.array([noise_I, noise_Q, noise_U]), overwrite=True)
                        
        elif red_noise_tilt is not None and lknee is not None:

            print('Generating white plus red noise realisations - note these are not beam smoothed.')
            np.random.seed(seed)
            # Convert sigmas to steradians!
            sigma_I_str = np.sqrt(pix_area * sigma_I[i]**2 / (180 * 60 / np.pi)**2)
            sigma_Q_str = np.sqrt(pix_area * sigma_P[i]**2 / (180 * 60 / np.pi)**2)
            sigma_U_str = np.sqrt(pix_area * sigma_P[i]**2 / (180 * 60 / np.pi)**2)

            ell = np.arange(3 * nside)
            nl_I = sigma_I_str**2 * (1 + (ell / lknee[i])**red_noise_tilt[i])
            nl_Q = sigma_Q_str**2 * (1 + (ell / lknee[i])**red_noise_tilt[i])
            nl_U = sigma_U_str**2 * (1 + (ell / lknee[i])**red_noise_tilt[i])
            nl_I[0:2] = 0
            nl_Q[0:2] = 0
            nl_U[0:2] = 0

            if hits_map is not None:
                noise_I = hp.synfast(nl_I, nside) / np.sqrt(hits / np.amax(hits))
                noise_Q = hp.synfast(nl_Q, nside) / np.sqrt(hits / np.amax(hits))
                noise_U = hp.synfast(nl_U, nside) / np.sqrt(hits / np.amax(hits))
                noise_I[hits==0] = hp.UNSEEN
                noise_Q[hits==0] = hp.UNSEEN
                noise_U[hits==0] = hp.UNSEEN
            elif hits_map is None:
                noise_I = hp.synfast(nl_I, nside)
                noise_Q = hp.synfast(nl_Q, nside)
                noise_U = hp.synfast(nl_U, nside)

            noises.append(np.array([noise_I, noise_Q, noise_U]))
                
            hp.write_map('{0}/{1}_nu{2}GHz_red_noise_IQU_seed{3}_nside{4:04d}.fits'.format(out_dir, out_prefix, freq_str, int(seed), nside),
                         np.array([noise_I, noise_Q, noise_U]), overwrite=True)
            
    return cov, noises


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


def initialise_sky(nside, preset_strings, output_unit):
    return pysm3.Sky(
        nside=nside,
        preset_strings=preset_strings,
        output_unit=output_unit
    )


def make_sky_map(sky, freq, fwhm, nside, nside_downgraded,
                 instrument_noise_freq, noise_seed,
                 output_directory, output_prefix,
                 smooth_map=True, save_map=True, downgrade_map=True):
    # Check if directory exists
    if save_map:
        if output_directory == None:
            raise ValueError(
                "output_direction cannot be None. Please specify a valid path and filename")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

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
        return file_path
    else:
        return full_map, file_path


if __name__ == "__main__":
    sky = initialise_sky(128, ["d1", "s1", "c1"], "uK_CMB")
    _ = make_sky_map(sky, 60, 60*units.arcmin, 128, 64,
                 37.42, 1111, "outputs",
                 "litebird",
                 True, True, True)
