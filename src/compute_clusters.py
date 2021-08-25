from clustering import compute_P_maps
import numpy as np
import hdbscan

test_file1 = "outputs/full_mission_litebird_nu40GHz_60.0arcmin_nside128"
test_file2 = "outputs/full_mission_litebird_nu60GHz_60.0arcmin_nside128"


def get_noiseless_clusters(test_file1, test_file2):
    pmap1 = compute_P_maps(
        test_file1,
        freq=40,
        smooth_fwhm=None,
        uK_cmb=False
    )

    pmap2 = compute_P_maps(
        test_file2,
        freq=40,
        smooth_fwhm=None,
        uK_cmb=False
    )

    pmaps = np.concatenate(
        [pmap1[:, np.newaxis], pmap2[:, np.newaxis]], axis=1)
    len_max = 10000
    pmaps = pmaps[:len_max, :len_max]
    hdbscan_runner = hdbscan.HDBSCAN(
        min_cluster_size=15, min_samples=15, prediction_data=True)
    hdbscan_runner.fit(pmaps)
    labels = hdbscan_runner.labels_
    print(np.unique(labels))
    soft_clusters = hdbscan.all_points_membership_vectors(hdbscan_runner)

    # Assign noise points to "most likely" cluster based on soft membership scores

    noiseless_clusters = np.argmax(soft_clusters, axis=1)
    return noiseless_clusters
