#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define MASK_WIDTH  5
#define MASK_RADIUS MASK_WIDTH/2

#define O_TILE_WIDTH 12
#define BLOCK_WIDTH 16

//@@ INSERT CODE HERE

//   maskWidth := 5
//    maskRadius := maskWidth/2 // this is integer division, so the result is 2

// typedef struct {
// int width; int height; int pitch; int channels; float* data;
// } * wbImage_t;



__global__ void convolution_2D_kernel(float *P,float *N,
									  int height,int width, int channels,
									  const float * __restrict__ M) {
// index = (yIndex*width + xIndex)*channels + channelIndex;
  //float clamp(float,float,float);
// float clamp(float x, float start, float end)
//{
//	return min(max(x, start), end);
//};
	
  //__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
  for (int k=0;k<channels;k++){
	__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
	
  int row_o = blockIdx.y*O_TILE_WIDTH + ty;
  int col_o = blockIdx.x*O_TILE_WIDTH + tx;

  int row_i = row_o - MASK_RADIUS;
  int col_i = col_o - MASK_RADIUS;

  if((row_i >= 0)&& (row_i < height)
	 && (col_i >= 0) && (col_i < width)){
    Ns[ty][tx] = N[(row_i*width + col_i)*channels+k];
  }
  else{
    Ns[ty][tx] = 0.0f;
  }

  __syncthreads();
	
  float output = 0.0f;
  if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH){
    for(int y = 0; y < MASK_WIDTH; y++) {
      for(int x = 0; x < MASK_WIDTH; x++) {
		  // output += M[j*MASK_WIDTH+i]*Ns[tz][i+ty][j+tx];
		  output += M[y*MASK_WIDTH+x]*Ns[y+ty][x+tx];
	  }
//		__syncthreads();
	}
//	__syncthreads();
    if(row_o < height && col_o < width)
	{
		P[(row_o*width + col_o)*channels+k]=fmin(fmax(output,0.0f),1.0f);
//    __syncthreads();
	}
//	  __syncthreads();
	}
	  __syncthreads();
  }
	
  // it seems I put __syncthreads in the wrong place, and it will also cause problem, two are needed.
  // the other problem is that I once put the assigning output outside if(ty <...) and cause a problem
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
	// assigned as a 2d matrix
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
	
	// allocate memory to device variables
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
	
	// host to device
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH,1);
	//dim3 dimBlock(O_TILE_WIDTH,O_TILE_WIDTH,1);
	//dim3 dimGrid(ceil(imageWidth/O_TILE_WIDTH),ceil(imageHeight/O_TILE_WIDTH),1);
	// dim3 dimGrid(ceil(imageWidth/BLOCK_WIDTH),ceil(imageHeight/BLOCK_WIDTH),1);
	//dim3 dimGrid(ceil(imageWidth),ceil(imageHeight),1);
	dim3 dimGrid( 1+ (imageWidth-1)/O_TILE_WIDTH, 1+ (imageHeight-1)/O_TILE_WIDTH,1);
	convolution_2D_kernel<<<dimGrid,dimBlock>>>
		(deviceOutputImageData,deviceInputImageData,imageHeight,imageWidth,imageChannels,deviceMaskData);
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
	
	// device to host
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);
	// free cuda memory
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}


