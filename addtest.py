import numpy 
# from numpy import array
import time
# import pyopencl as cl
# from pyopencl import array
# import pyopencl.tools
# from procedures import *
import os
from numpy import array
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'
# p = Procedures()


# # start = time.clock()

# # p =Procedures()
# x = numpy.ones((4,4))

# ge = p.globalError(4,4,x)
# print ge[0]
# # a = []
# # print numpy.sum(x[0])
# # print time.clock() - start

import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue
x = pycl_array.to_device(queue, np.random.rand(3920,100).astype(np.float32))
# a = pycl_array.to_device(queue, np.random.rand(50000).astype(np.float32))
# b = pycl_array.to_device(queue, np.random.rand(50000).astype(np.float32))  
y = np.random.rand(3920,100).astype(np.float32)
# Create two random pyopencl arrays
# c = pycl_array.empty_like(a)  # Create an empty pyopencl destination array
start = time.clock()
d=[]
for i in range(3920):	
	d.append(pycl_array.sum(x[i]))
	# d = numpy.sum(y[i])

print time.clock() - start


# print("x: {}".format(x)) 
# print("d: {}".format(d))  
# Print all three arrays, to show sum() worked

