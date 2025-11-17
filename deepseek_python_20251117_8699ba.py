"""
Hyperbolic Brane Dynamics (HBD) Cosmology Package

A comprehensive implementation of the HBD cosmological model featuring
geometric dark energy and distinctive CMB signatures from hyperboloid
brane embedding in a 5D hyperspherical bulk.
"""

from .background import HBDBackground
from .perturbations import HBDPerturbations
from .spectra import HBDSpectra
from .class_interface import HBDClassInterface
from .parameters import HBDParameters

__version__ = "1.0.0"
__author__ = "Nicholas R. C. Jimison"
__email__ = "nrcj1987@gmail.com"

__all__ = [
    'HBDBackground',
    'HBDPerturbations', 
    'HBDSpectra',
    'HBDClassInterface',
    'HBDParameters',
]