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
					// SIGMA(W*X) + B
					c[row*(M-N+1) + col] = temp + bias;
				}

			}


		    
		}
		"""