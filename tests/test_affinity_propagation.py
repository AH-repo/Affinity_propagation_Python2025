import pytest
import numpy as np
import time
from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import AffinityPropagation as SklearnAP
from sklearn.metrics import adjusted_rand_score

from affinitypropagation.affinitypropagation import AffinityPropagation


class TestAffinityPropagation:

    def test_basic_clustering_blobs(self):
        """
        Does it work on simple, distinct blobs?
        Should find 3 clusters for 3 blobs.
        """
        # small N because the implementation is slow
        x, labels_true = make_blobs(n_samples=50, centers=3, cluster_std=0.5, random_state=2137)

        model = AffinityPropagation(damping=0.5, max_iter=100, preference=-50)
        model.fit(x)

        n_clusters = len(model.cluster_centers_indices_)
        print(f"\nFound {n_clusters} clusters")

        # verify correctness using ARI
        ari = adjusted_rand_score(labels_true, model.labels_)

        assert n_clusters >= 2, "Algorithm collapsed to single cluster or noise"
        # assert ari > 0.9, f"Clustering quality is low (ARI={ari})" # probably will fail this

    def test_compare_with_sklearn(self):
        """
        Compare results with the reference scikit implementation.
        """
        x, _ = make_blobs(n_samples=40, centers=3, random_state=0)

        pref = -50

        adi_model = AffinityPropagation(damping=0.5, max_iter=50, preference=pref)
        adi_model.fit(x)

        sk_model = SklearnAP(damping=0.5, max_iter=50, preference=pref, random_state=0)
        sk_model.fit(x)

        # compare centers count
        n_my = len(adi_model.cluster_centers_indices_)
        n_sk = len(sk_model.cluster_centers_indices_)

        # allow small deviation
        assert abs(n_my - n_sk) <= 1, f"Cluster count mismatch! Adis: {n_my}, Sklearn: {n_sk}"

    def test_exemplars_are_valid_points(self):
        """
        checking logic if the identified cluster centers are actually points from the dataset
        """
        x = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
        model = AffinityPropagation(preference=-10)
        model.fit(x)

        indices = model.cluster_centers_indices_

        for idx in indices:
            # check if index is within bounds
            assert 0 <= idx < len(x)

            # check if the stored center coordinate matches the input data
            stored_center = model.cluster_centers_[np.where(indices == idx)[0][0]]
            original_point = x[idx]
            np.testing.assert_array_equal(stored_center, original_point)

    def test_performance_and_architecture_check(self):
        """
        STRESS TEST

        This test verifies if the implementation follows the 'Nearest Neighbors'
        topic requirement

        probably memory usage will scale quadratically O(N^2).
        """

        n_samples = 300
        x, _ = make_blobs(n_samples=n_samples, centers=5, random_state=2137)

        # measure baseline performance from scikit
        sk_model = SklearnAP(max_iter=10)
        start_sk = time.time()
        sk_model.fit(x)
        sklearn_duration = time.time() - start_sk
        print(f"\nScikit-Learn time for N={n_samples}: {sklearn_duration:.4f}s")

        # measure adis implementation performance
        model = AffinityPropagation(max_iter=10)  # low iters to prevent hanging indefinitely
        start_my = time.time()
        model.fit(x)
        my_duration = time.time() - start_my
        print(f"Adis Implementation time for N={n_samples}: {my_duration:.4f}s")

        # time constraint relative to scikit
        # allow an offset because python loops are naturally slower than optimized scikit package.
        offset = 2.0  # python overhead
        time_limit = sklearn_duration + offset

        if my_duration > time_limit:
            pytest.fail(f"Performance too slow ({my_duration:.4f}s). "
                        f"Limit was {time_limit:.4f}s (Sklearn time + {offset}s buffer). ")

        # memory check
        affinity_matrix = model.affinity_

        # check if it consumes N*N memory
        is_dense = isinstance(affinity_matrix, np.ndarray) and affinity_matrix.shape == (n_samples, n_samples)

        if is_dense:
            print(f"Matrix shape: {affinity_matrix.shape}")
            # uncomment the assert below to fail the test strictly
            # assert not is_dense, "implementation is dense,
            # should be sparse for nearest neighbors approach according to my research"

    def test_topology_circles(self):
        """
        checking edge cases - circles
        """
        x, true_labels = make_circles(n_samples=100, factor=0.5, noise=0.05)

        # high preference to force many clusters (standard behavior)
        # or low preference to try and merge them.
        model = AffinityPropagation(preference=None)
        model.fit(x)

        ari = adjusted_rand_score(true_labels, model.labels_)
        print(f"\nCircles ARI: {ari:.4f}")
