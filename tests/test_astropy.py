"""Test out astropy
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import astropy
import numpy as np

def test_astropy_installed():
    np.testing.assert_equal(astropy.__version__, '1.3.3')
