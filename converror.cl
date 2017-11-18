kernelsource = """
			__kernel void converror(
		    __global double* error,
		    __global double* index,
		    __global double* filter,
		    __global double* out,
		    const unsigned int M,
		    const unsigned int N)
		    {
		    	int row = get_global_id(0);
		    	int col = get_global_id(1);

		    	int index_row;
		    	int index_col;
		    	int receptive_row;
		    	int receptive_col;
		    	int fil_col;
		    	int l;
		    	int k;
		    	double temp;

				if(row < (M-N+1))
				{
					index_row = row/2;
					//printf("-%d-",index_row);
					if(col < (M-N+1))
					{
						receptive_row = row;
						index_col = col/2;
						temp = 0.0;
						for(k=0;k<N;k++)
						{ 
							l=0;
							fil_col = 0;
							for(receptive_col=col;receptive_col<N+col;receptive_col++){
								//barrier(CLK_GLOBAL_MEM_FENCE);
								if(receptive_row*M+receptive_col == (int)index[index_row*((M-N+1)/2)+index_col+l]){
									
									temp = out[receptive_row*M + receptive_col];
									out[receptive_row*M + receptive_col] = temp + error[index_row*((M-N+1)/2)+index_col+l]*filter[k*N+fil_col];
				
									l += 1;
									//break;
								}
								fil_col += 1;
							}
							receptive_row += 1;
							
						}
						//out[row*(M-N+1) + col] = temp;
					}				
				}

		    }
		"""