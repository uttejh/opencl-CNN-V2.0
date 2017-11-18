import numpy
from PIL import Image
from numpy import array
import pyopencl as cl 
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'
from procedures import *
# def pad(x, order):
# 	kernelsource = """
# 	#include <string.h>
# 	__kernel void pad(
# 	__global const double* A,
# 	__global double* B,
# 	const unsigned int M)
# 	{
# 		char str[] = "fcba73";
# 		char keys[] = "1234567890";
# 		int i;
# 		i = strstr(str,keys);
# 		printf("The first number in str is at position %s",i);
# 		//return 0;
# 	}
# 	"""
# 	context = cl.create_some_context()
# 	queue = cl.CommandQueue(context)
# 	program = cl.Program(context, kernelsource).build()

# 	out_order = order + 2

# 	h_a = x
# 	d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)

# 	h_b = numpy.empty((out_order,out_order))
# 	d_b = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_b.nbytes)

# 	pad = program.pad
# 	pad.set_scalar_arg_dtypes([None, None, numpy.uint32])

# 	pad(queue, h_b.shape, None, d_a, d_b, out_order)
# 	queue.finish()
# 	cl.enqueue_copy(queue, h_b, d_b)

# 	return h_b

# arr = numpy.random.rand(3,3)
# pad(arr,3)
# image = Image.open('1.jpg')

# data = numpy.array(image,dtype="double")

# # print pad(arr,3)
# print pad(data,data.shape[0])
# # print data.dtype

# x = [1,2,3,4,5,6,7,8,9]
# y = [3,8,1]

# c = index for x[index] in y
# print c
# map(a.__getitem__, b)
p=Procedures()
x= numpy.random.rand(40,20,7,7).astype(numpy.float64)
y=numpy.random.rand(40,20,7,7).astype(numpy.float64)

z=numpy.random.rand(40,3,3).astype(numpy.float64)
a = numpy.random.rand(3,3,3)
print a
a.sort(axis=0)
a.sort(axis=1)
a.sort(axis=2)

print a