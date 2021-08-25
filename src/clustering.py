"""Python module to cluster pixels in the sky into regions.

Original author: Luke Jew
Adapted by Richard Grumitt (25/05/2021)
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from sklearn.cluster import MeanShift, estimate_bandwidth
from astropy.constants import si
from astropy.cosmology import default_cosmology


T_cmb = default_cosmology.get().Tcmb0

def compute_P_maps(map_name, freq=None, smooth_fwhm=None, uK_cmb=True, offset=0):
    map_IQU = hp.read_map(map_name,(0,1,2))
    
    if smooth_fwhm is not None:
        map_IQU = hp.smoothing(map_IQU, np.radians(smooth_fwhm))
    if uK_cmb:
        x1 = si.h * freq * 1e9 / si.k_B / T_cmb
        map_IQU = x1.value ** 2 * np.exp(x1.value) * map_IQU / (np.exp(x1.value) - 1) ** 2
    pmap = np.sqrt(np.power(map_IQU[1],2)+np.power(map_IQU[2],2))
    Nside = hp.get_nside(pmap)
    Npix = hp.nside2npix(Nside)
    mask = np.ones(Npix)
    mask[np.where((map_IQU[1]==hp.UNSEEN))] = 0.
    pmap += offset
    pmap[mask == 0] = hp.UNSEEN
    
    return pmap
    

def compute_pixel_vectors(Nside):
    """Computes the vectors to the pixel centres for every pixel in a map.
    
    Paramters
    ---------
    Nside : int
            Nside of map
            
    Returns
    -------
    vec : array_like
          Array of vectors to the pixel centres
    """
    Npix = hp.nside2npix(Nside)
    Ipix = np.arange(Npix)
    vec = np.asarray(hp.pix2vec(Nside, Ipix))
    return vec
