import numpy
import pyopencl as cl 
import os
from numpy import array
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

from numpy import array 
# Creating a dictionary
# It consists of our filter weights. Like an array of 3D arrays 
# filters1 = []
# filters2 = []

class Procedures:
	def __init__(self):
		self.bla = []

	# Layer 1 filters
	@staticmethod
	def initFilters1(filternum,n_in,n_out,fsize):
		filters1 = []
		for x in range(filternum):
			
			w_bound = numpy.sqrt(6./float(n_in+n_out))
			filters1.append(numpy.random.uniform(-w_bound,w_bound,(fsize,fsize)))
		return filters1

	# Layer 2 filters
	@staticmethod
	def initFilters2(filternum,n_in,n_out,fsize):
		filters2 = []
		for x in range(filternum):
			# filters2[x] = []
			w_bound = numpy.sqrt(6./float(n_in+n_out))
			filters2.append(numpy.random.uniform(-w_bound,w_bound,(fsize,fsize))) #removed astype
		return filters2

	@staticmethod
	def convolution(x, w, bias, num, order):
		kernelsource = """
		__kernel void convolute(
		    __global double* a,
		    __global double* b,
		    __global double* c,
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
					// SIGMA(W*X) + B
					//if(isnan(temp) || isinf(temp)){
					//	c[row*(M-N+1) + col] = 1.0;
					//}else{
					//	c[row*(M-N+1) + col] = temp + bias;
					//}
					c[row*(M-N+1) + col] = temp + bias;
				}

			}


		    
		}
		"""
		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()
		
		F_order = 3
		
		out_order = (order - F_order + 1)

		

		convolute = program.convolute
		convolute.set_scalar_arg_dtypes([None, None, None, numpy.uint32, numpy.uint32, numpy.float32])
		out = []

		noOffilters = len(w)

		for img in range(num):
			if (num == 1):
				h_a = x
			else:
				h_a = x[img]

			d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
			# Convoluting Image with each filter
			for filt in range(noOffilters):
				# Passing each filter
				h_b = w[filt]
				# h_b = numpy.ones((3,3)).astype(numpy.float32)
				d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

				h_d = numpy.empty((out_order,out_order))
				d_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_d.nbytes)

				convolute(queue, (order,order), None, d_a, d_b, d_d, order, F_order, bias)
				queue.finish()
				cl.enqueue_copy(queue, h_d, d_d)

				# appending output of convolution with each filter
				out.append(h_d)		
		
		return out

	@staticmethod
	def relu(x, num, order):
		kernelsource = """
		    __kernel void relu(
		    __global double* A,
		    __global double* B,
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

		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()

		relu = program.relu
		relu.set_scalar_arg_dtypes([None,None,numpy.uint32])
		relu_out = []

		# For each convoluted array
		for it in range(num):
			h_a = x[it]
			d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)

			h_b = numpy.empty((order,order))	
			d_b =cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

			relu(queue, h_a.shape, None, d_a, d_b,order)
			queue.finish()
			cl.enqueue_copy(queue, h_b, d_b)

			relu_out.append(h_b)

		return relu_out

	# @staticmethod
	# def pooling(x, num, order):
	# 	kernelsource = """
	# 		__kernel void pool(
	# 	    __global double* A,
	# 	    __global double* B,
	# 	    __global double* C,
	# 	    const unsigned int N)
	# 	    {
	# 			int i = get_global_id(0);
	# 		    int j = get_global_id(1);

	# 		    int index1;
	# 		    int index2;
				
	# 			double t1,t2,t3,t4,t5,t6;
	# 		    if ((i < N-1) && (i%2 == 0))
	# 		    {
	# 				if ((j < N-1) && (j%2 == 0))
	# 			    {
	# 					t1 = A[i*N + j];
	# 					t2 = A[i*N + j+1];
	# 					t3 = A[(i+1)*N + j];
	# 					t4 = A[(i+1)*N + j+1];
	# 					if(t1>t2)
	# 					{
	# 						t5 = t1;
	# 						index1 = i*N + j;
	# 					}
	# 					else{
	# 						t5 = t2;
	# 						index1 = i*N + j + 1;
	# 					}

	# 					if(t3>t4)
	# 					{
	# 						t6 = t3;
	# 						index2 = (i+1)*N + j;
	# 					}
	# 					else{
	# 						t6 = t4;
	# 						index2 = (i+1)*N + j+1;
	# 					}
	# 					int x = (i/2);
	# 					int y = (j/2);
	# 					if(t5>t6)
	# 					{
	# 						B[x*(N/2) + y] = t5;
	# 						C[x*(N/2) + y] = index1;
	# 					}else{
	# 						B[x*(N/2) + y] = t6;
	# 						C[x*(N/2) + y] = index2;
	# 					}
	# 			    }
	# 		    }
	# 	    }
	# 	"""

	# 	context = cl.create_some_context()
	# 	queue = cl.CommandQueue(context)
	# 	program = cl.Program(context, kernelsource).build()

	# 	out_order = (order/2)
	# 	# h_a =  numpy.random.uniform(0,1,(400,400)).astype(numpy.float32)

	# 	pool = program.pool
	# 	pool.set_scalar_arg_dtypes([None,None,None,numpy.uint32])

	# 	pool_out = []
	# 	index = []

	# 	for it in range(num):
	# 		h_a = x[it]
	# 		d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)

	# 		h_b = numpy.empty((out_order,out_order))
	# 		d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

	# 		h_c = numpy.empty((out_order,out_order))
	# 		d_c = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_c)

	# 		pool(queue, (order, order), None, d_a, d_b, d_c, order)
	# 		queue.finish()
	# 		cl.enqueue_copy(queue, h_b, d_b)
	# 		cl.enqueue_copy(queue, h_c, d_c)

	# 		pool_out.append(h_b)
	# 		index.append(h_c)

	# 	return pool_out,index

	@staticmethod
	def pooling(x, num, order):
		kernelsource = """
			__kernel void pool(
		    __global double* A,
		    __global double* B,
		    const unsigned int N)
		    {
				int i = get_global_id(0);
			    int j = get_global_id(1);

			
				
				double t1,t2,t3,t4,t5,t6;
			    if ((i < N-1) && (i%2 == 0))
			    {
					if ((j < N-1) && (j%2 == 0))
				    {
						t1 = A[i*N + j];
						t2 = A[i*N + j+1];
						t3 = A[(i+1)*N + j];
						t4 = A[(i+1)*N + j+1];
						if(t1>t2)
						{
							t5 = t1;
							
						}
						else{
							t5 = t2;
							
						}

						if(t3>t4)
						{
							t6 = t3;
							
						}
						else{
							t6 = t4;
							
						}
						int x = (i/2);
						int y = (j/2);
						if(t5>t6)
						{
							B[x*(N/2) + y] = t5;
							
						}else{
							B[x*(N/2) + y] = t6;
							
						}
				    }
			    }
		    }
		"""

		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()

		out_order = (order/2)
		# h_a =  numpy.random.uniform(0,1,(400,400)).astype(numpy.float32)

		pool = program.pool
		pool.set_scalar_arg_dtypes([None,None,numpy.uint32])

		pool_out = []

		for it in range(num):
			h_a = x[it]
			d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)

			h_b = numpy.empty((out_order,out_order))
			d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)


			pool(queue, (order, order), None, d_a, d_b, order)
			queue.finish()
			cl.enqueue_copy(queue, h_b, d_b)

			pool_out.append(h_b)

		return pool_out


	@staticmethod
	def pad(x, num, order):
		kernelsource = """
			__kernel void pad(
		    __global double* A,
		    __global double* B,
		    const unsigned int M)
		    {
				int i = get_global_id(0);
			    int j = get_global_id(1);
				
				if((i<M) && (j<M))
				{
					
					if((j == 0) || (j == M-1) || (i == 0) || (i == M-1))
					{
						B[i*M + j] = 0;
					}else{
						B[i*M + j] = A[(i-1)*(M-2) + j-1];
					}			
				}
		    }
		"""
		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()

		out_order = order + 2

		pad = program.pad
		pad.set_scalar_arg_dtypes([None, None, numpy.uint32])
		pad_out = []

		for it in range(num):
			if (num == 1):
				h_a = x
			else:
				h_a = x[it]
	
			d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)

			h_b = numpy.empty((out_order,out_order))
			d_b = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_b.nbytes)

			pad(queue, h_b.shape, None, d_a, d_b, out_order)
			queue.finish()
			cl.enqueue_copy(queue, h_b, d_b)
			pad_out.append(h_b)

		return pad_out


	# @staticmethod
	# def derivative(x):
	# 	kernelsource = """
	# 		__kernel void derivative(
	# 	    __global double* A,
	# 	    __global double* B,
	# 	     const unsigned int N)
	# 	    {
	# 		    int i = get_global_id(0);

	# 		 	if (i < N)
	# 		    {
			           
	# 			    if(A[i] < 0)
	# 			    {
	# 			    	B[i] = 0.0; // If negative then substitute with 0
	# 			    }
	# 			    else{
	# 			    	B[i] = 1.0; // else - then positive. So, append 1
	# 			    }
			        
	# 		    }


	# 	    }
	# 	"""

	# 	context = cl.create_some_context()
	# 	queue = cl.CommandQueue(context)
	# 	program = cl.Program(context, kernelsource).build()
	# 	order = len(x)
	# 	h_a = x
	# 	d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
	# 	h_b = numpy.empty((order))	
		# # write read nbytes
	# 	d_b =cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

	# 	derivative = program.derivative
	# 	derivative.set_scalar_arg_dtypes([None,None,numpy.uint32])

	# 	derivative(queue, h_a.shape, None, d_a, d_b,order)
	# 	queue.finish()
	# 	cl.enqueue_copy(queue, h_b, d_b)
	# 	return h_b


	@staticmethod
	def BP_FC_to_HL(m, n, error, hl, w, alpha):
		kernelsource = """
			__kernel void bp(
		    __global double* a,
		    __global double* b,
		    __global double* c,
		    __global double* d,
		    __global double* e,
		    const unsigned int M,
		    const unsigned int N,
		    float alpha)
		    {
				int i = get_global_id(0); 
		    	int j = get_global_id(1); 
				
				float err = 0;
				float dw = 0;
				float derivative =0.0; 
				float weight = 0.0;
		    	
		    	if(i < M)
		    	{
		    		err = a[i];
		    		
		    		if(b[i]<0)
					{
						derivative = 0.0 ;
					}else{
						derivative = 1.0;
					}

					if(j < N)
					{
						
						weight = c[i*M + j];
						//E*f`(x)*w 
						dw = err*derivative*weight;
									
						// -ve * -ve = +ve
						d[i*N + j] = weight - alpha*dw ;

						e[i*N + j] = dw;
					}
		    	}

		    }
		"""

		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()

		h_a = array(error)
		d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
		h_b = hl
		d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
		h_c = array(w)
		d_c = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_c)

		h_d = numpy.empty(h_c.shape)
		d_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_d.nbytes)

		h_e = numpy.empty(h_c.shape)
		d_e = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_e.nbytes)

		bp = program.bp
		bp.set_scalar_arg_dtypes([None,None,None,None,None, numpy.uint32, numpy.uint32, numpy.float32])

		bp(queue, h_c.shape, None, d_a, d_b, d_c, d_d, d_e, m, n, alpha)
		queue.finish()
		cl.enqueue_copy(queue, h_d, d_d)
		cl.enqueue_copy(queue, h_e, d_e)
		return h_d,h_e


	# @staticmethod
	# def BP_conv(w):
	# 	kernelsource = """
	# 		__kernel void bpconv(
	# 		__global double* a,
	# 	    __global double* b,
	# 	    __global double* c,
	# 	    const unsigned int N,
	# 	    float alpha)
	# 	    {
	# 	    	int i = get_global_id(0);
	# 	    	int j = get_global_id(1);

	# 	    	if(i<N)
	# 	    	{
		    		
	# 	    	}

	# 	    }
	# 	"""

	# 	context = cl.create_some_context()
	# 	queue = cl.CommandQueue(context)
	# 	program = cl.Program(context, kernelsource).build()





	@staticmethod
	def globalError(m, n, err):
		kernelsource = """
			__kernel void adderr(
		    __global double* a,
		    __global double* b,
		    const unsigned int M,
		    const unsigned int N)
		    {
		    	int i = get_global_id(0);
		    	int j = get_global_id(1);

				
		    	if(i<M)
		    	{
		    		if(j<N)
		    		{
		    			b[i] += a[i*N + j];
		    			barrier(CLK_GLOBAL_MEM_FENCE);
		    		}
					
		    	}

		    }
		"""

		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()

		
		h_a = err
		d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
		h_b = numpy.zeros((m))
		d_b = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_b.nbytes)

		adderr = program.adderr
		adderr.set_scalar_arg_dtypes([None, None, numpy.uint32, numpy.uint32])
		adderr(queue, h_a.shape, None, d_a, d_b, m, n)
		queue.finish()
		cl.enqueue_copy(queue, h_b, d_b)

		return h_b



	@staticmethod
	def test(w):
		for x in range(len(w)):
			print(w[x])

