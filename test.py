import numpy as np
from numpy import array
import time
from procedures import *

# p = Procedures()

x = numpy.random.rand((10))

y = numpy.random.rand(10,100)
z = numpy.random.rand((100))


# start = time.clock()
# # p.derivative(x)

# # print x
# # print 

# for i in range(1000):
# 	1.0 if x[i] <= 0. else 0.0

# print time.clock() - start
# start = time.clock()
# b = []
# Dw_HL_to_output = []
# for jj in range(10):
# 	for kk in range(100):
# 		b.append(x[jj]*(1.0 if z[jj] <= 0 else 0.0)*y[jj][kk])
# 	Dw_HL_to_output.append(b)

# print time.clock() - start
a=[]
b=[]
for i in range(10):
	for j in range(100):
		b.append(1)
	a.append(b)
print array(a).shape