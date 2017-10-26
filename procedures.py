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
			filters1[x] = numpy.random.uniform(-w_bound,w_bound,(3,3))

	# Layer 2 filters
	def initFilters2(filternum,n_in,n_out):
		for x in range(filternum):
			filters2[x] = []
			w_bound = numpy.sqrt(6./(n_in+n_out))
			filters2[x] = numpy.random.uniform(-w_bound,w_bound,(3,3))

	@staticmethod
	def convolution(x):
		kernelsource = """
		__kernel void convolute(
		    __global float* a,
		    __global float* b,
		    __global float* c,
		    const unsigned int M,
		    const unsigned int N)
		{
		    int i = get_global_id(0); 
		    int j = get_global_id(1); 

		    int l;
		    int p = 0;
		    int q = 0;
		    int k;
		    int g;
		    float temp=0.0;
			
			if(i < (M-N+1))
			{
				p = 0;
				
				if(j < (M-N+1))
				{
					g = i;
					temp = 0.0;
					for(k=0;k<N;k++)
					{
						q = 0;
						for(l=p;l<N+p;l++)
						{
							temp += a[g*M + l] * b[g*N + q];
							q += 1;
						}
						g = g+1;
					}
					
					c[i*(M-N+1) + j] = temp;
					p=p+1;
				}

			}


		    
		}
		"""

		# kernelsource="""
		# 	__kernel void convolute(__global float4* a_vec, __global float4* b_vec,
		# 	__global float* output, __local float4* partial_dot) {

		# 		int gid = get_global_id(0);
		# 		int lid = get_local_id(0);
		# 		int group_size = get_local_size(0);

		# 		partial_dot[lid] = a_vec[gid] * b_vec[gid];
		# 		barrier(CLK_LOCAL_MEM_FENCE);

		# 		for(int i = group_size/2; i>0; i >>= 1) {
		# 			if(lid < i) {
		# 				partial_dot[lid] += partial_dot[lid + i];
		# 			}
		# 			barrier(CLK_LOCAL_MEM_FENCE);
		# 		}

		# 		if(lid == 0) {
		# 			output[get_group_id(0)] = dot(partial_dot[0], (float4)(1.0f));
		# 		}
		# 	}
		# """

		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()
		
		# for iteration in range()

		# h_a = filters1[0]
		h_a = numpy.ones((9,9)).astype(numpy.float32)
		h_b = numpy.ones((3,3)).astype(numpy.float32)
		# h_c = numpy.empty((3,3)).astype(numpy.float32)
		h_d = numpy.empty((7,7)).astype(numpy.float32)

		d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
		d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
		# d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_d.nbytes)
		d_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_d.nbytes)
		
		convolute = program.convolute
		convolute.set_scalar_arg_dtypes([None, None, None, numpy.uint32, numpy.uint32])
		convolute(queue, (9,9), None, d_a, d_b, d_d, 9, 3)
		queue.finish()
		cl.enqueue_copy(queue, h_d, d_d)
		

		return h_d
	def relu(x):
		kernelsource = """
		    __kernel void relu(
		    __global float* a,
		    __global float* b,
		     const unsigned int n)
		    {
		 	int i = get_global_id(0);
			if ( i < n )
			{
				if (x[i] < 0)
					x[i] = 0;
				else
					continue;

			}


		     }



		    """
		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()
		h_a =  numpy.random.rand((3,3)).astype(numpy.float32)
		h_b = numpy.empty((3,3)).astype(numpy.float32)

		d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
		d_b =cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

		relu = program.relu
		relu.set_scalar_arg_dtypes([None,None])
		relu(queue, h_a.shape, None, d_a, d_b,9)
		queue.finish()
		cl.enqueue_copy(queue, h_b, d_b)
		return h_b
