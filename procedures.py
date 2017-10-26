import numpy
import pyopencl as cl 
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# Creating a dictionary
# It consists of our filter weights. Like an array of 3D arrays 
filters1 = []
filters2 = []

class Procedures:
	def __init__(self):
		self.bla = []

	# Layer 1 filters
	@staticmethod
	def initFilters1(filternum,n_in,n_out):
		for x in range(filternum):
			# filters1[x] = []
			w_bound = numpy.sqrt(6./(n_in+n_out))
			filters1.append(numpy.random.uniform(-w_bound,w_bound,(3,3)))

	# Layer 2 filters
	def initFilters2(filternum,n_in,n_out):
		for x in range(filternum):
			# filters2[x] = []
			w_bound = numpy.sqrt(6./(n_in+n_out))
			filters2.append(numpy.random.uniform(-w_bound,w_bound,(3,3)))

	@staticmethod
	def convolution(x, w, bias):
		kernelsource = """
		__kernel void convolute(
		    __global float* a,
		    __global float* b,
		    __global float* c,
		    const unsigned int M,
		    const unsigned int N,
		    float bias)
		{
		    int row = get_global_id(0); 
		    int col = get_global_id(1); 

		    int receptive_col;
		    int fil_col = 0;
		    int k;
		    int receptive_row;
		    int fil_row;
		    float temp=0.0;
			
			/* Each row must end at M-N+1
			e.g - for 5*5 i/p with 3*3 filter.
			The filter must stop before M-N+1 = 3 rd so that from there (3rd) it will increment N times resulting
			in [(M-N+1)  + N ]= M + 1 (An array starts from 0 so we add 1).
			Going From TOP to BOTTOM*/

			if(row < (M-N+1))
			{		
				// Applying it from LEFT TO RIGHT		
				if(col < (M-N+1))
				{

					// Receptive Field's row. Dimensions same as filters = N*N
					receptive_row = row;

					// Filter's row. Dim = N*N
					fil_row = 0;
					temp = 0.0;

					// Looping N times so that we can move from TOP to BOTTOM 
					for(k=0;k<N;k++)
					{
						// Looping N times LEFT to RIGHT
						fil_col = 0;
						for(receptive_col=col;receptive_col<N+col;receptive_col++)
						{
							// a consists of N*N Receptive Field and b - Filter - N*N
							// adding the multiplied values with each iteration until N*N times and
							// then reinitializing temp to 0
							temp += a[receptive_row*M + receptive_col] * b[fil_row*N + fil_col];
							fil_col += 1;
					
						}
						fil_row = fil_row + 1;
						receptive_row = receptive_row+1;
					}
					// assign dot product(receptive field, filter) to C
					c[row*(M-N+1) + col] = temp + bias;
				}

			}


		    
		}
		"""

		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()
		
		# h_b = numpy.ones((3,3)).astype(numpy.float32)
		h_a = x
		# h_c = numpy.empty((3,3)).astype(numpy.float32)
		h_d = numpy.empty((26,26)).astype(numpy.float32)


		d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
		# d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_d.nbytes)
		d_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_d.nbytes)

		convolute = program.convolute
		convolute.set_scalar_arg_dtypes([None, None, None, numpy.uint32, numpy.uint32, numpy.float32])
		out = []

		noOffilters = len(w)
		# Convoluting Image with each filter
		for filt in range(noOffilters):
			# Passing each filter
			h_b = w[filt]
			d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
			convolute(queue, (28,28), None, d_a, d_b, d_d, 28, 3, bias)
			queue.finish()
			cl.enqueue_copy(queue, h_d, d_d)

			# appending output of convolution with each filter
			out.append(h_d)		
		
		return out

	@staticmethod
	def relu(x):
		kernelsource = """
		    __kernel void relu(
		    __global float* A,
		    __global float* B,
		     const unsigned int N)
		    {
			    int i = get_global_id(0);
			    int j = get_global_id(1);

			 	if ((i < N) && (j < N))
			    {
			           
				    if(A[i*N+j] < 0)
				    {
				    	B[i*N+j] = 0; // If negative then substitute with 0
				    }
				    else{
				    	B[i*N+j] = A[i*N+j]; // else - then positive. So, append same value.
				    }
			        
			    }


		    }
		    """

		shape = len(x[0])

		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()
		# h_a =  numpy.random.uniform(-1,1,(3,3)).astype(numpy.float32)
		h_b = numpy.empty((shape,shape)).astype(numpy.float32)

		
		d_b =cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

		relu = program.relu
		relu.set_scalar_arg_dtypes([None,None,numpy.uint32])
		relu_out = []

		# For each convoluted array
		for it in range(len(x)):
			h_a = x[it]
			d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
			relu(queue, h_a.shape, None, d_a, d_b,shape)
			queue.finish()
			cl.enqueue_copy(queue, h_b, d_b)

			relu_out.append(h_b)

		return relu_out

	@staticmethod
	def test(w):
		for x in range(len(w)):
			print(w[x])