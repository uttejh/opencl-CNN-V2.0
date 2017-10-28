import numpy as np
from numpy import inf
import time
x = np.array([np.inf, -np.inf, np.nan, -128, 128])# Create array with inf values

# print x # Show x array
start = time.clock()
x[x == inf] = 0 # Replace inf by 0
print x
# print x # Show the result
print time.clock() - start