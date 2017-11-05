#include<stdio.h>

int input_size = 6;

int main(){
int i = 0;
int j = 0;
//consider 6*6 matrix
int pool_size = 3; //prime number
int k,m,SelectedRow, SelectedCol, selectedIndex;
float thisValue,maxValue=0;
int input[input_size*input_size];

//taking an ID array. this can be visualised as 2D array but it is in 1D format.
for( k=0; k<input_size; k++){
    for( m=0; m<input_size; m++){
        scanf("%d",&input[k*input_size + m]);
    }
}

    while(input[i]){
        while(input[i*input_size + j]){

        // the if statement enters only when we encounter at first element of pool. It indicates that we are ready for pooling
            if(i % pool_size == 0 && j % pool_size == 0){
                    printf("\n Entering two for loops at i=% and j=%d", i,j);
                //printf("row: %d, col: %d",row,col)
                for (SelectedRow = i; SelectedRow < pool_size; SelectedRow++) {
                    for (SelectedCol = j ; SelectedCol < pool_size;  SelectedCol++) {
                        thisValue = input[ SelectedRow*input_size + SelectedCol ];
                        if (thisValue > maxValue) {
                            maxValue = thisValue;
                            selectedIndex =  SelectedRow * input_size + SelectedCol;
                        }else if (!thisValue){
                            break;
                            printf("\n This value doesn't exist at %d", selectedIndex);
                        }
                    }
                }
                printf("\n Selected index from input is at %d and its corresponding value", selectedIndex, input[selectedIndex]);
            }
            j++;
        }
        i++;
    }
return 0;
}




/*
//NOT DONE YET. I'm Missing some part of logic.
#include<stdio.h>

int main(){
    int i=0,j=0;
    int row = i;
    int col = i*input_size + j;  //assuming arrays are stored in row major format.
    while(!input[row] && !input[row*input_size + col])
    const int input[36];
    const int input_size = 6; //6*6 matrix
    
    //const int row;//= get_global_id(0);
    //const int col = row*input_size + j; //= get_global_id(1);
    

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

}*/
