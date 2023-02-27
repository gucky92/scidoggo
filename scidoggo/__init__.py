from ._rbf._rbfinterp import RbfRegression
from ._pls import PLSCanonical, PLSRegression, PLSSVD

from ._version import __version__

__all__ = [
    'RbfRegression',
    "PLSCanonical", "PLSRegression", "PLSSVD", 
    '__version__'
]
