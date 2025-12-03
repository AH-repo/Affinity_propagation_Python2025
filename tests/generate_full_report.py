import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import AffinityPropagation as SklearnAP
from sklearn.metrics import adjusted_rand_score, silhouette_score

from affinitypropagation.affinitypropagation import AffinityPropagation


def get_datasets():
    """Generates a dictionary of datasets to test."""
    n_samples = 200
    random_state = 42

    datasets = []

    x, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.7, random_state=random_state)
    datasets.append(("Blobs", x, y, -50))

    x, y = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    datasets.append(("Varied Variance", x, y, -50))

    x, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    x = np.dot(x, transformation)
    datasets.append(("Anisotropic", x, y, -20))

    x, y = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
    datasets.append(("Moons", x, y, -20))

    x, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
    datasets.append(("Circles", x, y, -20))

    return datasets


def run_model(name, model, X, y_true):
    """Runs a single model and returns metrics."""
    start_time = time.time()
    try:
        model.fit(X)
        end_time = time.time()

        duration = end_time - start_time

        # handle cases where model has no centers
        if hasattr(model, 'cluster_centers_indices_') and model.cluster_centers_indices_ is not None:
            n_clusters = len(model.cluster_centers_indices_)
            centers = model.cluster_centers_indices_
        else:
            n_clusters = 0
            centers = []

        labels = model.labels_

        # calculate metrics
        # if labels are all -1 or all same, metrics might fail or warn
        if len(set(labels)) > 1:
            ari = adjusted_rand_score(y_true, labels)
            try:
                sil = silhouette_score(X, labels) if n_clusters > 1 else -1
            except:
                sil = -1
        else:
            ari = 0.0
            sil = -1.0

        return {
            "Time (s)": duration,
            "Clusters": n_clusters,
            "ARI": ari,
            "Silhouette": sil,
            "Labels": labels,
            "Centers": centers
        }
    except Exception as e:
        print(f"Error running {name}: {e}")
        return None


def plot_comparison(x, adis_res, sk_res, dataset_name, ax1, ax2):
    """Visualizes the comparison side-by-side."""

    def _plot(ax, res, title):
        if res is None:
            ax.text(0.5, 0.5, "Error / Did not converge", ha='center')
            return

        labels = res["Labels"]
        centers_idx = res["Centers"]
        unique_labels = set(labels)

        if len(unique_labels) > 0:
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        else:
            colors = []

        for i, k in enumerate(unique_labels):
            class_member_mask = (labels == k)
            xy = x[class_member_mask]

            # use black for noise or if something went wrong
            if k == -1:
                col = (0, 0, 0, 1)
            else:
                col = tuple(colors[i])

            ax.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=col, markeredgecolor='k', markersize=6)

            # highlight center\
            # only if we have valid centers and this cluster isn't noise
            if k != -1 and len(centers_idx) > 0:
                pass

        ax.set_title(f"{title}\nTime: {res['Time (s)']:.4f}s | ARI: {res['ARI']:.2f}")
        ax.grid(True, alpha=0.3)

    _plot(ax1, adis_res, "Adis Code")
    _plot(ax2, sk_res, "Scikit-Learn")


def generate_full_report():
    datasets = get_datasets()
    results_table = []

    CODE_IMPLEMENTATION = "Adis Code"
    SCIKIT_IMPLEMENTATION = "Scikit-Learn"

    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 4 * len(datasets)))
    if len(datasets) == 1: axes = [axes]

    print(f"{'Dataset':<15} | {'Impl':<12} | {'Time (s)':<10} | {'ARI':<6} | {'Clusters':<8}")
    print("-" * 65)

    for i, (name, X, y, pref) in enumerate(datasets):
        adis_model = AffinityPropagation(damping=0.5, max_iter=100, preference=pref)
        adis_res = run_model("Custom", adis_model, X, y)

        sk_model = SklearnAP(damping=0.5, max_iter=100, preference=pref, random_state=42)
        sk_res = run_model("Sklearn", sk_model, X, y)

        if adis_res:
            results_table.append({
                "Dataset": name, "Implementation": CODE_IMPLEMENTATION,
                "Time (s)": adis_res["Time (s)"], "ARI": adis_res["ARI"],
                "Clusters": adis_res["Clusters"]
            })
            print(
                f"{name:<15} | {CODE_IMPLEMENTATION:<12} | {adis_res['Time (s)']:<10.4f} | {adis_res['ARI']:<6.2f} | {adis_res['Clusters']:<8}")

        if sk_res:
            results_table.append({
                "Dataset": name, "Implementation": SCIKIT_IMPLEMENTATION,
                "Time (s)": sk_res["Time (s)"], "ARI": sk_res["ARI"],
                "Clusters": sk_res["Clusters"]
            })
            print(
                f"{name:<15} | {SCIKIT_IMPLEMENTATION:<12} | {sk_res['Time (s)']:<10.4f} | {sk_res['ARI']:<6.2f} | {sk_res['Clusters']:<8}")

        ax1, ax2 = axes[i]
        plot_comparison(X, adis_res, sk_res, name, ax1, ax2)
        ax1.set_ylabel(name, fontsize=12, fontweight='bold')

    if not results_table:
        print("No results generated!")
        return

    df = pd.DataFrame(results_table)

    try:
        pivot = df.pivot(index="Dataset", columns="Implementation", values="Time (s)")

        if CODE_IMPLEMENTATION in pivot.columns and SCIKIT_IMPLEMENTATION in pivot.columns:
            pivot["Slowdown (x times)"] = pivot[CODE_IMPLEMENTATION] / pivot[SCIKIT_IMPLEMENTATION]

        # save to excel
        with pd.ExcelWriter("benchmark_results.xlsx") as writer:
            df.to_excel(writer, sheet_name="Raw Data", index=False)
            pivot.to_excel(writer, sheet_name="Performance Analysis")

        print("\nResults saved to 'benchmark_results.xlsx'")

    except Exception as e:
        print(f"Could not save Excel report due to data shape issues: {e}")

    plt.tight_layout()
    plt.savefig("benchmark_visualization.png")
    print("Plots saved to 'benchmark_visualization.png'")


if __name__ == "__main__":
    generate_full_report()