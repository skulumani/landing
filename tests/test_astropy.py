"""Test out astropy
"""

import astropy
import numpy as np

def test_astropy_installed():
    np.testing.assert_equal(astropy.__version__, '1.3.2')
