from src.clustering import compute_P_maps
import numpy as np
import hdbscan
import healpy as hp
import matplotlib.pyplot as plt

def get_noiseless_clusters(filenames, frequencies, smoothing_fwhm=None, uK_cmb=None):

    """

    Function for getting clusters from healpix sky maps. It assigns a cluster if to each pixel in the map. In addition it saves some clustered sky maps and related histograms. 

    Inputs
    ------
    filenames: path to files containing the healpix maps. This can be a list of files, with sky maps for different frequencies as well
    frequencies: a list of frequencies corresponding to the sky maps of those frequences
    smoothing_fwhm: the smoothing scale for the maps in arcmin. By default set to None, meaning no smoothing
    uK_cmb: Set the units of the CMB

    Returns
    -------
    noiseless_clusters: an array with the length of each pixel and ordered in the same order of the input pixels, that contains the cluster ids for each pixel. 

    """

    pmaps = compute_P_maps(
        filenames[0],
        freq=frequencies[0],
        smooth_fwhm=smoothing_fwhm,  # We already smooth the maps when they are generated?
        uK_cmb=uK_cmb  # We already set this to be true before? Please chedk these
    )

    if len(filenames) > 1:
        for file, frequency in zip(filenames[1:], frequencies[1:]):
            pmap = compute_P_maps(
                file,
                freq=frequency,
                smooth_fwhm=smoothing_fwhm,
                uK_cmb=uK_cmb
            )
            if len(pmaps.shape)==1:
                pmaps = pmaps[:,np.newaxis]
            pmaps = np.hstack((pmaps, pmap[:, np.newaxis]))

    print(np.shape(pmaps))
    hdbscan_runner = hdbscan.HDBSCAN(
        min_cluster_size=15, min_samples=15, prediction_data=True).fit(pmaps)
    labels = hdbscan_runner.labels_
    hp.mollview(labels, norm='hist')
    plt.savefig('outputs/regions_before_soft_clustering.png')
    plt.close()

    plt.hist(labels)
    plt.savefig('outputs/regions_before_soft_clustering_histograms.png')
    plt.close()

    print(np.unique(labels))
    soft_clusters = hdbscan.all_points_membership_vectors(hdbscan_runner)

    # Assign noise points to "most likely" cluster based on soft membership scores

    noiseless_clusters = np.argmax(soft_clusters, axis=1)
    
    np.save("outputs/regions.npy", noiseless_clusters)
    hp.mollview(noiseless_clusters, norm='hist')
    plt.savefig('outputs/regoins_map.png')
    plt.close()

    plt.hist(noiseless_clusters)
    plt.savefig('outputs/regions_hist.png')
    plt.close()

    return noiseless_clusters


if __name__ == "__main__":
    import time
    start = time.time()
    filenames = [test_file1, test_file2]
    frequencies = [40,40]
    sky_regions = get_noiseless_clusters(filenames, frequencies)
    print(f"Time taken = {time.time() - start}")
