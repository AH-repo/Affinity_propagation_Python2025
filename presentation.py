import numpy as np
import matplotlib.pyplot as plot
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation as SklearnAP
from affinitypropagation import AffinityPropagation

# Dataset
X, _ = make_blobs(n_samples=100, centers=3)


my_ap = AffinityPropagation(damping=0.7)
my_labels = my_ap.fit_predict(X)

# Scikit-learn
sk_ap = SklearnAP(damping=0.7)
sk_labels = sk_ap.fit_predict(X)

# Plot
fig, (ax1, ax2) = plot.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=my_labels, s=80)
ax1.scatter(X[my_ap.cluster_centers_indices_][:, 0],
            X[my_ap.cluster_centers_indices_][:, 1],
            c='red', marker='*', s=400, edgecolors='black', linewidth=2)
ax1.set_title(f'AH implementation ({len(np.unique(my_labels))} clusters)')

ax2.scatter(X[:, 0], X[:, 1], c=sk_labels, s=80)
ax2.scatter(X[sk_ap.cluster_centers_indices_][:, 0],
            X[sk_ap.cluster_centers_indices_][:, 1],
            c='red', marker='*', s=400, edgecolors='black', linewidth=2)
ax2.set_title(f'Scikit-learn ({len(np.unique(sk_labels))} clusters)')

plot.tight_layout()
plot.show()