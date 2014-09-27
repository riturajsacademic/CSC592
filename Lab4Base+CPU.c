#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)


	void matrixMultiply_cpu(float *A, float *B, float *C,
                                      int numARows, int numAColumns,
                                      int numBRows, int numBColumns,
                                      int numCRows, int numCColumns){
	/*
	*	Here, C is the out matrix, in which we write values of C
	*/	
		#define inputA(i,j) A[i*numAColumns+j]
		#define inputB(i,j) B[i*numBColumns+j]
		#define outputC(i,j) C[i*numCColumns+j]
		
		printf("\nComming in Matrix Multiple CPU numAR  %d,NumAC  %d,NumBR   %d,NumBc %d",numARows,numAColumns,numBRows,numBColumns);
		
		int width = numARows;
		int height = numBColumns;
		
		int iteratorWidth = 0;
		int iteratorHeight = 0;
		
		int iteratorWidthResult=0;
		float sum=0;
		
		for(iteratorWidth=0;iteratorWidth<width;iteratorWidth++){
			for(iteratorHeight=0;iteratorHeight<height;iteratorHeight++){
				sum = 0.0;
				for(iteratorWidthResult = 0; iteratorWidthResult<numBRows;iteratorWidthResult++){
					sum = sum +	inputA(iteratorWidth,iteratorWidthResult)*inputB(iteratorWidthResult,iteratorHeight);
				}
				outputC(iteratorWidth,iteratorHeight) = sum;
			}
			
		}
		
		printf("Outer Loop: %d, Inner Loop: %d, Extra Outer loop: %d ",width,height,numBRows);
	}






// Compute C = A * B
__global__ void matrixMultiply_kernel(float *A, float *B, float *C,
                                      int numARows, int numAColumns,
                                      int numBRows, int numBColumns,
                                      int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to perform register tiling for this MP
}

static void matrixMultiply(float *A, float *B, float *C, int numARows,
                           int numAColumns, int numBRows, int numBColumns,
                           int numCRows, int numCColumns) {
  //@@ Insert code to launch matrix multiplication
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  numCRows = numARows;
  numCColumns = numBColumns;
  hostC = ( float * )malloc(sizeof(float) * numCRows * numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
	
	wbTime_start(GPU,"Running CPU computation")	;
	matrixMultiply_cpu(hostA, hostB, hostC, numARows, numAColumns, numBRows,
                 numBColumns, numCRows, numCColumns);
	wbTime_stop(GPU, "Stopping CPU Computation");
/*
  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void**) &deviceA, sizeof(float) * numARows * numAColumns);
  cudaMalloc((void**) &deviceB, sizeof(float) * numBRows * numBColumns);
  cudaMalloc((void**) &deviceC, sizeof(float) * numCRows * numCColumns);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns,
             cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiply(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows,
                 numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns,
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");
	*/
  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
