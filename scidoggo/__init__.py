from ._rbf._rbfinterp import RbfRegression
from ._pls import PLSCanonical, PLSRegression, PLSSVD
from ._tikhonov import Tikhonov
from ._rank_constraint import RankConstraint
from ._rank_plus_sparse import Rank1PlusSparse

from ._version import __version__

__all__ = [
    'RbfRegression',
    "PLSCanonical", "PLSRegression", "PLSSVD", 
    "Tikhonov", "RankConstraint", "Rank1PlusSparse",
    '__version__'
]
