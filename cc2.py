"""
A simple implementation of the consensus clustering method based on [1]_.
References
----------
.. [1] S. Monti, ‘Consensus Clustering: A Resampling-Based Method for
   Class Discovery and Visualization of Gene Expression Microarray Data’, p. 28.
"""

import numpy as np


def resample_indices(n, size=None, replace=True):
    """Resample a dataset defined by a collection of consecutive sample indices.
    Parameters
    ----------
    n : int
        The number of samples in the non-resampled dataset.
    size : int, optional
        The number of samples to draw. If None, will be equal to the size of
        non-resampled dataset.
    replace : bool
        Whether to draw samples with replacement, creating a bootstrap
        resample. If False and size < n, a subsample of the data will be
        created (default True).
    Returns
    -------
    np.ndarray
        The resampled dataset indices.
    Notes
    -----
        The dataset is defined by a collection of consecutive sample indices.
    """
    if not size:
        size = n
    return np.random.choice(np.arange(n), size=size, replace=replace)


def connectivity_matrix(labels, indices, size):
    """Compute the connectivity matrix, given sample indices and cluster
    associations.
    The connectivity matrix is defined as (eq. 1 in the paper):
    M(i, j) = 1 if items (i, j) belong to the same cluster else 0.
    Parameters
    ----------
    labels : np.ndarray, shape (n_samples,)
        The cluster association of each sample in the dataset.
    indices : np.ndarray, shape (n_samples,)
        The original dataset index of each sample.
    size : int
        The number of samples in the original dataset.
    Returns
    -------
    np.ndarray, shape (n_samples, n_samples)
        The connectivity matrix.
    """
    M = np.zeros((size, size), dtype=np.float)
    # create pairwise indexing grid
    ii, jj = np.meshgrid(indices, indices, indexing="ij")
    # pairwise compare all cluster labels
    M[ii, jj] = np.equal.outer(labels, labels)
    return M


def indicator_matrix(indices, size):
    """Compute the indicator matrix, given sample indices and original dataset
    size.
    The indicator matrix is defined as:
    I(i, j) = 1 if items (i, j) are present in indices else 0.
    Its purpose is to keep track of which samples from the original dataset
    are present in each resampled dataset (e.g. when bootstrapping or
    subsampling).
    Parameters
    ----------
    indices : np.ndarray, shape (n_samples,)
        The original dataset index of each sample.
    size : int
        The number of samples in the original dataset.
    Returns
    -------
    np.ndarray
        The indicator matrix.
    """
    in_dataset = np.isin(np.arange(size), indices)
    # do all pairwise comparisons
    return np.logical_and.outer(in_dataset, in_dataset).astype(np.float)


def consensus_clustering(X, estimator, n_iter=1000, resample_size=None, replace=True):
    """Perform consensus clustering on a given dataset.
    Returns the consensus matrix, which is defined as the sum of connectivity
    matrices over all resamples, normalized by the sum of indicator matrices
    (eq. 2 in the paper).
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The dataset to cluster.
    estimator : sklearn.BaseEstimator
        Scikit-learn estimator object with a `fit_predict` method that implements
        the clustering algorithm.
    n_iter : int, optional
        How many resampling iterations to perform (default 1000).
    resample_size : int, optional
        How many samples to include in each resampled dataset. If None, the
        resampled dataset will have the same size as the original (default None).
    replace : bool, optional
        Whether to sample with replacement, resulting in bootstrap resampling
        (default True).
    Returns
    -------
    np.ndarray, shape (n_samples, n_samples)
        The consensus matrix.
    """
    M = np.zeros((X.shape[0], X.shape[0]))
    I = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[1]):
        row_ids = [j for j in range(X.shape[0])]
        labels = X[:, i]
        M += connectivity_matrix(labels, row_ids, X.shape[0])
        I += indicator_matrix(row_ids, X.shape[0])
    cons = M / I
    cons = M
    cons = np.where(np.isinf(cons), 0, cons)
    cons = np.where(np.isnan(cons), 0, cons)
    return cons


def cdf(M):
    """Compute the empirical cumulative distribution function of a consensus
    matrix.
    Implements eq. 5 from the paper. The CDF can be used as a relative measure
    of cluster stability in selecting optimal number of clusters. 
    Parameters
    ----------
    M : np.ndarray, shape (n_samples, n_samples)
        The consensus matrix as returned by `consensus_clustering`.
    Returns
    -------
    xx : np.ndarray, shape (n_samples,)
        The points at which the CDF is computed.
    f : np.ndarray, shape (n_samples,)
        The values of the empirical CDF at each point in `xx`.
    """
    xx = np.sort(M[np.triu_indices_from(M)])
    f = np.arange(len(xx)) / len(xx)
    return xx, f


def cdf_diff(M_list):
    """Compute the proportion increase in area under CDF for a sequence of
    consensus matrices.
    This function can be used to select the optimal number of clusters by
    using the value with the largest increase in area under CDF.
    Parameters
    ----------
    M_list : list of np.ndarray
        The sequence of consensus matrices corresponding to increasing cluster
        numbers.
    Returns
    -------
    np.ndarray
        The proportion increase in area under CDF for each consensus matrix in
        `M_list`.
    Notes
    -----
    This function assumes that the consensus matrices correspond to consecutive
    increasing cluster numbers starting from 2.
    """
    # The first matrix corresponds to K = 2 clusters.
    xx, f = cdf(M_list[0])
    areas = np.array([np.trapz(*reversed(cdf(M))) for M in M_list])
    return np.r_[areas[0], (areas[1:] - areas[:-1]) / areas[:-1]]
