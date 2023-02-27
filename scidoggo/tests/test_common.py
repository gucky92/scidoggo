import pytest

from sklearn.utils.estimator_checks import check_estimator

from scidoggo import PLSCanonical, PLSSVD, PLSRegression, RbfRegression


@pytest.mark.parametrize(
    "estimator",
    [PLSRegression(), PLSSVD(), PLSCanonical(), RbfRegression()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
