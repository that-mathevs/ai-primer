"""Pytest setup shared by every spec.

The lessons multiply tiny matrices. Multi-threaded BLAS (Anaconda's default)
spends far longer coordinating threads than computing on matrices that
small, and several test runs at once can thrash the machine. One thread is
faster here. This must run before NumPy is first imported.
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")


import sys  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    # Specs that call a lesson's figures() leave figures open; matplotlib keeps them until closed.
    yield
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")
