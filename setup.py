#! /usr/bin/env python
"""Collection of models with a scikit-learn API."""
# commands:
# python setup.py sdist bdist_wheel
# twine upload dist/*

import codecs
import os
from setuptools import setup, find_packages
import setuptools

# get __version__ from _version.py
version_dict = {}
folderpath = os.path.dirname(__file__)
ver_file = os.path.join(folderpath, 'scidoggo', '_version.py')
with open(ver_file) as f:
    exec(f.read(), version_dict)

DISTNAME = 'scidoggo'
DESCRIPTION = 'Collection of models with a scikit-learn API.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()

MAINTAINER = 'Matthias Christenson'
MAINTAINER_EMAIL = 'gucky@gucky.eu'
URL = 'https://github.com/gucky92/scidoggo'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/gucky92/scidoggo'
VERSION = version_dict['__version__']
INSTALL_REQUIRES = [
    'numpy', 
    'scipy', 
    'scikit-learn'
]
CLASSIFIERS = [
]
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'
    ],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

try:
    from pythran.dist import PythranExtension, PythranBuildExt
    setuptools.dist.Distribution(dict(setup_requires='pythran'))
    setup_args = {
        'cmdclass': {"build_ext": PythranBuildExt},
        'ext_modules': [
            PythranExtension(
                'scidoggo._rbf._rbfinterp_pythran',
                ['scidoggo/_rbf/_rbfinterp_pythran.py']
            ),
        ],
    }
except ImportError:
    print("not building Pythran extension - install pythran for more efficient code")
    setup_args = {}


setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE, 
    include_package_data=True, 
    **setup_args
)

