"""
Bootstrapping tools
"""

import numpy as np
from typing import Optional

def draw_bs_replicates(data: np.ndarray, estimator: str ='mean', nboots: int = 10000, axis: Optional[int]=None) -> np.ndarray:
    """
    Creates a bootstrap sample, computes replicates, and returns an array of replicates.

    Parameters
    ----------
    data : np.ndarray
        The input data.
    estimator : str, optional
        The estimator to use to calculate the bootstrap replicates, by default 'mean'.
    nboots : int, optional
        The number of bootstrapped samples to generate, by default 10000.
    axis : Optional[int], optional
        The axis along which to draw the bootstrap samples, by default None.
    
    Returns
    -------
    np.ndarray
        An array of bootstrapped replicates.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> draw_bs_replicates(data)
    array([3.2, 2.6, 3.6, 2.8, 2.8, 2.6, 3.2, 3.2, 3. , 3. , 2.6, 2.6, ...
    """
    # Create an empty array to store replicates
    if axis is None:
        bs_replicates = np.empty(nboots)
        length = data.size
        data = data.ravel()
    else:
        shape = list(data.shape)
        shape.pop(axis)
        bs_replicates = np.empty((nboots,)+tuple(shape))
        length = data.shape[axis]
    
    if isinstance(estimator, str):
        estimator = getattr(np, estimator)
    
    # Create bootstrap replicates as much as size
    for i in range(nboots):
        # Create a bootstrap sample
        idcs = np.random.randint(0, length, size=length)
        bs_sample = np.take(data, idcs, axis=axis)
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = estimator(bs_sample, axis=axis)
    
    return bs_replicates


def bs_cis(data: np.ndarray, alpha: float=0.05, estimator: str='mean', nboots: int=10000, axis: Optional[int]=None):
    """
    Compute bootstrap confidence intervals.
    
    This function computes the bootstrap confidence intervals for a given data and estimator.
    
    Parameters
    ----------
    data : np.ndarray
        The data to compute bootstrap confidence intervals for.
    alpha : float, optional
        The significance level (default is 0.05).
    estimator : str, optional
        The estimator to use to compute the replicates. Valid options are 'mean', 'median', etc. (default is 'mean').
    nboots : int, optional
        The number of bootstrap replicates to compute (default is 10000).
    axis : int, optional
        The axis along which to compute the bootstrap confidence intervals (default is None, i.e. ravel the array).
    
    Returns
    -------
    np.ndarray
        A 2D array containing the lower and upper bounds of the bootstrap confidence intervals.
    
    See Also
    --------
    draw_bs_replicates : Creates bootstrap replicates.
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> bs_cis(data, nboots=1000)
    array([[1.55, 4.45]])
    >>> data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> bs_cis(data, axis=1, nboots=1000)
    array([[2.1, 8.9],
        [3.7, 9.3]])
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> bs_cis(data, estimator='median', nboots=1000)
    array([[2., 4.]])
    """
    samples = draw_bs_replicates(data, estimator=estimator, nboots=nboots, axis=axis)
    return np.percentile(samples, axis=0, q=[alpha/2*100, (1-alpha/2)*100])


def sig_directional(data: np.ndarray, axis: Optional[int]=None, alpha: float=0.05, **kwargs):
    """
    Determine if the data is significantly positive or negative.
    
    This function returns a binary array indicating if the data is significantly positive or negative.
    
    Parameters
    ----------
    data : np.ndarray
        The data to test.
    axis : int, optional
        The axis along which to perform the test (default is None, i.e. ravel the array).
    alpha : float, optional
        The significance level (default is 0.05).
    kwargs : additional keyword arguments
        Additional keyword arguments passed to `bs_cis`.
    
    Returns
    -------
    np.ndarray
        A binary array with the same shape as `data` indicating if the data is significantly positive (1) or negative (-1).
    
    See Also
    --------
    bs_cis : Computes bootstrap confidence intervals.
    """
    cis = bs_cis(data, axis=axis, alpha=alpha, **kwargs)
    pos = (cis > 0).all(0).astype(int)
    neg = (cis < 0).all(0).astype(int)
    return pos-neg

def sig_overlap(data1, data2, **kwargs):
    """
    Determine if two sets of bootstrapped confidence intervals have overlapping ranges.
    
    Parameters
    ----------
    data1 : array_like
        The first set of data to compute bootstrapped confidence intervals for.
    data2 : array_like
        The second set of data to compute bootstrapped confidence intervals for.
    **kwargs : optional
        Keyword arguments to be passed to the `bs_cis` function, such as `alpha`, `estimator`, `nboots`, and `axis`.
        
    Returns
    -------
    bool
        Boolean indicating if the confidence intervals for `data1` and `data2` overlap (True) or not (False).
    
    Examples
    --------
    >>> import numpy as np
    >>> data1 = np.array([1, 2, 3, 4, 5])
    >>> data2 = np.array([1, 2, 3, 4, 5])
    >>> sig_overlap(data1, data2, nboots=1000, alpha=0.05)
    True
    
    >>> data1 = np.array([1, 2, 3, 4, 5])
    >>> data2 = np.array([6, 7, 8, 9, 10])
    >>> sig_overlap(data1, data2, nboots=1000, alpha=0.05)
    False
    """
    cis1 = bs_cis(data1, **kwargs)
    cis2 = bs_cis(data2, **kwargs)
    return (cis1[0] < cis2[1]) & (cis1[1] > cis2[0])