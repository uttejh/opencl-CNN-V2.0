import numpy
import time
from procedures import *
import pyopencl.array as cl_array 
# # from PIL import Image
# # from numpy import array

# # image = Image.open('3.jpg')
# # # Converts image into array of pixels

# # image = image.resize((28,28),Image.ANTIALIAS)
# # image.save("image_scaled_opt.jpg",optimize=True,quality=95)
# # arr = array(image)
# # print arr.shape
# # start = time.clock()
# # filters = dict()
# # for x in range(256):
# # 	filters[x] = []
# # 	filters[x] = numpy.random.uniform(-0.5,0.5,(3,3,3))
	
# # # print(filters[1])
# # # print(filters[4])
# # tt = time.clock() - start
# # print('Time:'+str(tt))

# p = Procedures()

# p.initFilters1(128,9,1)
# print(filters1[1])

# kernelsource = """
# __kernel void convolute(
# __global float* a,
# __global float* b,
# __global float* c,
# const unsigned int N)
# {
# 	int i = get_global_id(0); 
# 	int j = get_global_id(1);
# 	int k;
# 	int count1 = 0;
# 	int count2 = 0;

# 	if ((i < N) && (j < N))
# 	{
# 		for (k = 0; k < N; k++)
# 			count1 = count1 + 1;
# 		count2 = count2 + 1;
# 		c[i*N+j] = count2;
# 	}


# }
# """

context = cl.create_some_context()
queue = cl.CommandQueue(context)


# def general_clrand(queue, shape, dtype):
#     from pyopencl.clrandom import rand as clrand

#     dtype = numpy.dtype(dtype)
#     if dtype.kind == "c":
#         real_dtype = dtype.type(0).real.dtype
#         return clrand(queue, shape, real_dtype) + 1j*clrand(queue, shape, real_dtype)
#     else:
#         return clrand(queue, shape, dtype)

# a_gpu = general_clrand(queue, (200000,), numpy.float32)
# b_gpu = general_clrand(queue, (200000,), numpy.float32)
# a = a_gpu.get()
# b = b_gpu.get()
# start = time.clock()
# dot_ab_gpu = cl_array.dot(a_gpu, b_gpu).get()
# print(dot_ab_gpu)
# tt = time.clock() - start
# print(tt)
# dot_ab = numpy.dot(a, b)
# print(dot_ab)
# tt = time.clock() - start
# print(tt)
# program = cl.Program(context, kernelsource).build()

			

# # h_a = filters1[0]
# h_a = numpy.ones((3,3)).astype(numpy.float32)
# h_b = numpy.ones((3,3)).astype(numpy.float32)
# # h_c = b1
# h_d = numpy.empty((3,3)).astype(numpy.float32)

# d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
# d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
# # d_c = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_c)
# d_d = cl.Buffer(context, cl.mem_flags.WRITE_O.NLY, h_d.nbytes)

# convolute = program.convolute
# convolute.set_scalar_arg_dtypes([None, None, None, numpy.uint32])
# convolute(queue, h_a.shape, None, d_a, d_b, d_d, 3)
# queue.finish()
# cl.enqueue_copy(queue, h_d, d_d)

# print(h_d)

a = numpy.ones((3,3)).astype(numpy.float32)
x = a.get()
b = numpy.ones((3,3)).astype(numpy.float32)
y = b.get()
dot_ab_gpu = cl_array.dot(a,b).get()
print(dot_ab_gpu)