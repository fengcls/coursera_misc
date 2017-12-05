///////////////////////////////////
// not correct yet

// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
cudaError_t err = stmt;                                               \
if (err != cudaSuccess) {                                             \
wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
return -1;                                                        \
}                                                                     \
} while(0)


__global__ void scan1(float * input,float *output, float *S, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
    __shared__ float XY[2*BLOCK_SIZE];
    
    unsigned int t = threadIdx.x+ blockIdx.x*blockDim.x;
    if (t<len)
    XY[threadIdx.x]=input[t];
    //else
    //  XY[threadIdx.x]=0;
    
    for (int stride = 1;stride <= blockDim.x;stride *= 2)
    {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < blockDim.x)
        XY[index] += XY[index-stride];
        //__syncthreads();
    }
    
    for (int stride = BLOCK_SIZE/2; stride > 0;stride /= 2)
    {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index+stride < BLOCK_SIZE)
        {
            XY[index + stride] += XY[index];
        }
    }
    
    __syncthreads();
    //if (t < len)
    output[t] = XY[threadIdx.x];
    
    __syncthreads();
    if (threadIdx.x==0) {
        S[blockIdx.x]=XY[BLOCK_SIZE*2-1];
    }
}

__global__ void scan2(float * input, float * output, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
    __shared__ float XY[2*BLOCK_SIZE];
    
    unsigned int t = threadIdx.x+ blockIdx.x*blockDim.x;
    if (t<len)
    XY[threadIdx.x]=input[t];
    //else
    //  XY[threadIdx.x]=0;
    
    for (int stride = 1;stride <= blockDim.x;stride *= 2)
    {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < blockDim.x)
        XY[index] += XY[index-stride];
        //__syncthreads();
    }
    
    for (int stride = BLOCK_SIZE/2; stride > 0;stride /= 2)
    {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index+stride < BLOCK_SIZE)
        {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    //    if (t < len)
    output[t] = XY[threadIdx.x];
}

__global__ void scan3(float * input, float * output,int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
   	if (blockIdx.x)
    output[t] += input[blockIdx.x];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list
    
    args = wbArg_read(argc, argv);
    
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    wbLog(TRACE, "The number of input elements in the input is ", numElements);
    
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");
    
    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");
    
    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(BLOCK_SIZE,1,1);
    dim3 dimBlock(ceil(numElements/BLOCK_SIZE/2),1,1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    float * S_d;
    float * S_d_2;
    
    float * S_h;
    S_h = (float*) malloc(numElements/2/BLOCK_SIZE * sizeof(float));
    cudaMalloc((void**)&S_d, numElements/2/BLOCK_SIZE*sizeof(float));
    cudaMalloc((void**)&S_d_2, numElements/2/BLOCK_SIZE*sizeof(float));
    cudaMemcpy(S_d, S_h, numElements/2/BLOCK_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(S_d_2, S_h, numElements/2/BLOCK_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    
    scan1<<<dimGrid,dimBlock>>>(deviceInput,deviceOutput,S_d,numElements);
    
    //    cudaMemcpy(S_h, S_d, numElements/2/BLOCK_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    //    cudaMemcpy(S_d, S_h, numElements/2/BLOCK_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    
    //    cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost);
    //    cudaMemcpy(deviceOutput, hostOutput, numElements*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    scan2<<<dim3(1,1,1),dimBlock>>>(S_d,S_d_2,numElements/2/BLOCK_SIZE);
    
    //    cudaMemcpy(S_h, S_d, numElements/2/BLOCK_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    //    cudaMemcpy(S_d, S_h, numElements/2/BLOCK_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();
    scan3<<<dimGrid,dimBlock>>>(S_d_2,deviceOutput,numElements);
    
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");
    
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(S_d);
    cudaFree(S_d_2);
    
    wbTime_stop(GPU, "Freeing GPU Memory");
    
    wbSolution(args, hostOutput, numElements);
    
    free(hostInput);
    free(hostOutput);
    free(S_h);
    
    return 0;
}

