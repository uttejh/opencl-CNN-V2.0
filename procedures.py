import numpy
import pyopencl as cl 

# Creating a dictionary
# It consists of our filter weights. Like an array of 3D arrays 
filters1 = dict()
filters2 = dict()
b1 = 1 
LENGTH = 2352

class Procedures:
	def __init__(self):
		self.bla = []

	# Layer 1 filters
	@staticmethod
	def initFilters1(filternum,n_in,n_out):
		for x in range(filternum):
			filters1[x] = []
			w_bound = numpy.sqrt(6./(n_in+n_out))
			filters1[x] = numpy.random.uniform(-w_bound,w_bound,(3,3,3))

	# Layer 2 filters
	def initFilters2(filternum,n_in,n_out):
		for x in range(filternum):
			filters2[x] = []
			w_bound = numpy.sqrt(6./(n_in+n_out))
			filters2[x] = numpy.random.uniform(-w_bound,w_bound,(3,3,3))

	@staticmethod
	def convolution(x):
		
		kernelsource = """
		__kernel void convolute(
		    __global float* a,
		    __global float* b,
		    __global float* c,
		    const unsigned int count)
		{
		    int i = get_global_id(0);
		    if (i < count)
		    {
		    	int j=0;
		    	if(j<3)
		    	{
		    		c[i] = a[i] + b[i][j][0];
		    		j++;
		    	}
		    }	
		        
		}
		"""

		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()
		# h_a = filters1
		h_a = numpy.random.rand(LENGTH).astype(numpy.float32)
		h_b = x
		# h_c = b1
		h_d = numpy.empty(9).astype(numpy.float32)

		d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
		d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
		# d_c = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_c)
		d_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_d.nbytes)

		convolute = program.convolute
		convolute.set_scalar_arg_dtypes([None, None, None, numpy.uint32])
		convolute(queue, h_a.shape, None, d_a, d_b, d_d, 9)
		queue.finish()
		cl.enqueue_copy(queue, h_d, d_d)
		print(h_a)
		print(x[:9])

		return h_d

