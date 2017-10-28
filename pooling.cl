//kernel void forwardNaive(const int inputSize, global const float *input, global float *output) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    int pool_size  = x; //must be a prime number
    int i,j,RowOffset, ColOffset;
    
    float maxValue = 0, thisValue;
    
    if(row % pool_size == 0 && col % pool_size == 0){
    
      for (int SelectedRow = 0; SelectedRow < pool_size; SelectedRow++) {
          for (int SelectedCol = 0; SelectedCol < pool_size;  SelectedCol++) {
                  thisValue = input[ SelectedRow * input_size + SelectedCol ];
                  if (thisValue > maxValue) {
                      maxValue = thisValue;
                  }else if (!thisValue){
                    return;
                  }
              }
          }
      }
      RowOffset = row/pool_size;  
      ColOffset = col/pool_size;  
    
      select = RowOffset*input_size/pool_size + ColOffset;
      output[ select ] = maxValue;
//    selectors[ globalId ] = selector;
//    selectors[globalId] = 123;
  }
}
