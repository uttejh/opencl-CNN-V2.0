//kernel void forwardNaive(const int inputSize, global const float *input, global float *output) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    int pool_size  = x; //must be a prime number
    int i,j,RowOffset, ColOffset, selectedIndex;
    
    // so we are saving index values of max pixel values at Index array.
    int Index[]; //
    // ?? size of the array?
    float maxValue = 0, thisValue;
    
    if(row % pool_size == 0 && col % pool_size == 0){
    
      for (int SelectedRow = 0; SelectedRow < pool_size; SelectedRow++) {
          for (int SelectedCol = 0; SelectedCol < pool_size;  SelectedCol++) {
                  thisValue = input[ SelectedRow * input_size + SelectedCol ];
                  if (thisValue > maxValue) {
                      maxValue = thisValue;
                       selectedIndex =  SelectedRow * input_size + SelectedCol
                  }else if (!thisValue){
                    return;
                  }
              }
          }
      }
      RowOffset = row/pool_size;  
      ColOffset = col/pool_size;  
    
      select = RowOffset*(input_size/pool_size) + ColOffset;
      output[ select ] = maxValue;
     
      // so we have to save the selected Index of max value in each pooling layer, right? We are saving it in a 1D array for each pooling layer.  
      Index[row] = selectedInded;
      

    //a global variable is necessary to save the two variables X and Y.

//    selectors[ globalId ] = selector;
//    selectors[globalId] = 123;
  }
}
