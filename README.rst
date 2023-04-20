.. -*- mode: rst -*-

.. .. |Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. .. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master
.. .. _Travis: https://travis-ci.org/scikit-learn-contrib/project-template

.. .. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. .. _AppVeyor: https://ci.appveyor.com/project/glemaitre/project-template

.. .. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. .. _Codecov: https://codecov.io/gh/scikit-learn-contrib/project-template

.. .. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. .. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

.. .. |ReadTheDocs| image:: https://readthedocs.org/projects/scidoggo/badge/?version=latest
.. .. _ReadTheDocs: https://scidoggo.readthedocs.io/en/latest/?badge=latest

WIP: A collection of shallow and deep models
============================================

This project is still under development. 
This project contains a selection of models and data tools I have created, modified, and used for different projects.


Models
======

* Selectivity Model: model that non-linearly integrates input features to produce concave and convex isoresponse surfaces
* Partial Least Squares: modification of `sklearn.pls` models to allow for optional bias removal and orthogonalization of rotations and loadings
* RbfModel: modified and sklearn-compatible version of the `scipy.interpolate.RBFInterpolator` model
* TwoLayerEncodingModel: two layer encoding model with different objective functions
* Rank1PlusSparse: linear model with a rank one constraint and an added sparse weight matrix
* RankConstraint: linear model with a rank constraint (different approach to PLS)
* Tikhonov regression
* Collection of probabilistic circuit model designs using pytorch and pyro


Acknowledgments
===============

This package was created with the help of the scikit-learn templating tool: https://github.com/scikit-learn-contrib/project-template
