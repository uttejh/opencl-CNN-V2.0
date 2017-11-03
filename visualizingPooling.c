//NOT DONE YET. I'm Missing some part of logic.
#include<stdio.h>

int main(){
    const int input[36];
    const int input_size = 6; //5*5 matrix
    /*
    const int row;//= get_global_id(0);
    const int col = row*m+j; //= get_global_id(1);
    */

    int pool_size  = 2; //must be a prime number
    int i,j,RowOffset, ColOffset, selectedIndex;
    
    // so we are saving index values of max pixel values at Index array.
    int Index[]; //
    // ?? size of the array?
    float maxValue = 0, thisValue;

    while(!input[row] && !input[row*input_size + col]){
        // the if statement enters only when we encounter at first element of pool. 
        if(row % pool_size == 0 && col % pool_size == 0){  
            printf("row: %d, col: %d",row,col)
            for (int SelectedRow = row; SelectedRow < row + pool_size; SelectedRow++) {
                for (int SelectedCol = row*input_size + col ; SelectedCol < row*input_size + col+ pool_size;  SelectedCol++) {
                  thisValue = input[ SelectedCol ];
                  if (thisValue > maxValue) {
                      maxValue = thisValue;
                      selectedIndex =  SelectedRow * input_size + SelectedCol;
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
row++;
col++;  
}
}

}
