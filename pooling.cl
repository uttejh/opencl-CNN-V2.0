//there are some global variables which are to be understood. For now this code will work!



kernel void forwardNaive(const int batchSize, global const float *input, global int *selectors, global float *output) {
    const int globalId = get_global_id(0);

// **Notable global variables**
// gOutputSizeSquared, gOutputSize, 
    const int intraImageOffset = globalId % gOutputSizeSquared;
    const int outputRow = intraImageOffset / gOutputSize;
    const int outputCol = intraImageOffset % gOutputSize;

    const int image2dIdx = globalId / gOutputSizeSquared;
    const int plane = image2dIdx % gNumPlanes;
    const int n = image2dIdx / gNumPlanes;

    if (n >= batchSize) {
        return;
    }

    const int inputRow = outputRow * gPoolingSize;
    const int inputCol = outputCol * gPoolingSize;
    const int inputImageOffset = (n * gNumPlanes + plane) * gInputSizeSquared;
    int selector = 0;
    int poolInputOffset = inputImageOffset + inputRow * gInputSize + inputCol;
    float maxValue = input[ poolInputOffset ];
    for (int dRow = 0; dRow < gPoolingSize; dRow++) {
        for (int dCol = 0; dCol < gPoolingSize; dCol++) {
            bool process = (inputRow + dRow < gInputSize) && (inputCol + dCol < gInputSize);
            if (process) {
                float thisValue = input[ poolInputOffset + dRow * gInputSize + dCol ];
                if (thisValue > maxValue) {
                    maxValue = thisValue;
                    selector = dRow * gPoolingSize + dCol;
                }
            }
        }
    }
    output[ globalId ] = maxValue;
    selectors[ globalId ] = selector;
//    selectors[globalId] = 123;
}
