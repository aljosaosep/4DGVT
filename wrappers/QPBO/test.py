#!/usr/bin/env python

import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np

"""
Demonstrates how to use the QPBO solver with Python
"""

x = np.array([[5.63258, -359.244, -307.072, -256.236], [-359.244,  5.03157, -307.129, -256.352], [-307.072, -307.129,  4.47083, -255.776], [-256.236, -256.352, -255.776, 3.75671]], dtype="double")
mat_size = x.shape[0]
ret_result = np.empty((mat_size,))
mylib = ctypes.cdll.LoadLibrary("/home/osep/projects/generic_object_tracker_new_framework/wrappers/QPBO/qpbo_solver.so")


test = mylib.SolveQPBOMultiBranch
test.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t]
test(x,  ret_result, mat_size)

print ret_result