import numpy as np


class AffinityPropagation:
    """
    Implementation of Affinity Propagation clustering algorithm.
    The algorithm automatically finds the number of clusters.
    """

    def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, preference=None, affinity='euclidean', copy=True):
        """
        Parameters:
        - damping: damping factor (0.5 to 1.0), prevents oscillations
        - max_iter: maximum number of iterations
        - convergence_iter: number of iterations with no change that stops the algorithm
        - preference: how much points want to be exemplars (higher = more clusters)
        - affinity: 'euclidean' or 'precomputed' - how to compute similarity
        - copy: whether to make a copy of input data
        """
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.affinity = affinity
        self.copy = copy

        # Attributes after fitting
        self.cluster_centers_indices_ = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.n_iter_ = 0
        self.affinity_ = None

    def _compute_similarity_matrix(self, X):
        """
        Computes similarity matrix based on negative euclidean distance.
        Higher similarity means closer points.
        """
        n = X.shape[0]
        S = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # Negative squared euclidean distance
                dist = np.sum((X[i] - X[j]) ** 2)
                S[i, j] = -dist

        return S

    def fit(self, X):
        """
        Trains the model on data X.
        X: array (n_samples, n_features) or precomputed similarity matrix
        """
        # Validate affinity parameter
        if self.affinity not in ['euclidean', 'precomputed']:
            raise ValueError("affinity must be 'euclidean' or 'precomputed'")

        # Copy data if requested
        if self.copy:
            X = X.copy()


        n_samples = X.shape[0]

        # 1. Compute or use precomputed similarity matrix
        if self.affinity == 'precomputed':
            # X is already a similarity matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("Precomputed matrix must be square")
            S = X.copy()
            self.affinity_ = S
        else:
            # Compute similarity matrix
            S = self._compute_similarity_matrix(X)
            self.affinity_ = S

        # 2. Set preference - how much a point wants to be an exemplar
        if self.preference is None:
            # Default: median of similarities
            preference = np.median(S)
        else:
            preference = self.preference

        # Set diagonal (similarity of point to itself)
        np.fill_diagonal(S, preference)

        # 3. Initialize message matrices
        R = np.zeros((n_samples, n_samples))  # responsibility
        A = np.zeros((n_samples, n_samples))  # availability

        # Track convergence
        old_exemplars = set()
        exemplars_history = []
        no_change_count = 0

        # 4. Main algorithm loop
        for iteration in range(self.max_iter):

            # r(i,k) = s(i,k) - max{ a(i,k') + s(i,k') } for k' != k
            # how well-suited point k is to be exemplar for i
            AS = A + S  # sum of availability and similarity

            # For each point find the best candidate
            max_AS = np.max(AS, axis=1)

            R_new = np.zeros_like(R)
            for i in range(n_samples):
                for k in range(n_samples):
                    # Temporarily remove k from consideration
                    temp = AS[i].copy()
                    temp[k] = -np.inf
                    max_other = np.max(temp)

                    R_new[i, k] = S[i, k] - max_other

            # Damping - mix old and new values
            R = self.damping * R + (1 - self.damping) * R_new

            # a(i,k) = min(0, r(k,k) + sum{ max(0, r(i',k)) } for i' != i,k
            # how much point i should choose k
            A_new = np.zeros_like(A)
            for i in range(n_samples):
                for k in range(n_samples):
                    if i == k:
                        # For diagonal: sum of positive responsibilities
                        temp = R[:, k].copy()
                        temp[k] = 0
                        A_new[i, k] = np.sum(np.maximum(temp, 0))
                    else:
                        # For off-diagonal
                        temp = R[:, k].copy()
                        temp[i] = 0
                        temp[k] = 0
                        suma = R[k, k] + np.sum(np.maximum(temp, 0))
                        A_new[i, k] = min(0, suma)

            # Damping
            A = self.damping * A + (1 - self.damping) * A_new

            # Exemplars are points where r(k,k) + a(k,k) > 0
            E = R + A
            exemplars = set(np.where(np.diag(E) > 0)[0])

            exemplars_history.append(exemplars)

            # Check if number of clusters stabilized
            if exemplars == old_exemplars:
                no_change_count += 1
                if no_change_count >= self.convergence_iter:
                    print(f"Converged after {iteration + 1} iterations")
                    break
            else:
                no_change_count = 0
                old_exemplars = exemplars

        self.n_iter_ = iteration + 1

        # 5. Extract results
        E = R + A
        self.cluster_centers_indices_ = np.where(np.diag(E) > 0)[0]

        if len(self.cluster_centers_indices_) == 0:
            print("WARNING: No exemplars found!")
            self.labels_ = np.full(n_samples, -1)
            self.cluster_centers_ = np.array([])
            return self

        # 6. Assign each point to nearest exemplar
        self.labels_ = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            # Find exemplar with highest similarity
            similarities = S[i, self.cluster_centers_indices_]
            best_exemplar_idx = np.argmax(similarities)
            self.labels_[i] = best_exemplar_idx

        # center coordinates
        if self.affinity == 'euclidean':
            self.cluster_centers_ = X[self.cluster_centers_indices_]
        else:
            self.cluster_centers_ = None

        print(f"Found {len(self.cluster_centers_indices_)} clusters")

        return self

    def fit_predict(self, X):
        """
        Trains the model and returns cluster labels in one call.

        Parameters:
        - X: array (n_samples, n_features) or precomputed similarity matrix

        Returns:
        - labels: array of cluster labels
        """
        self.fit(X)
        return self.labels_