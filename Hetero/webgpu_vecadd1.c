// MP 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int size = len * sizeof(float);
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **) &d_A, size);
  cudaMemcpy(d_A, in1, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_B, size);
  cudaMemcpy(d_B, in2, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_C, size);
  vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);
  cudaMemcpy(out, d_C, size, cudaMemcpyDeviceToHost);
  // Free device memory for A, B, C
  cudaFree(d_Ad); cudaFree(d_B); cudaFree (d_C);
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = ( float * )malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  //cudaError_t err = cudaMalloc((void **) &d_A, size);
  //if (error != cudaSuccess) {
	//printf(“%s in %s at line %d\n”, cudaGetErrorString( err), __FILE__, __LINE__);
	//exit(EXIT_FAILURE);
  //}

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
