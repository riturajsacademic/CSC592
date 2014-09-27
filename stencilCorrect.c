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


__host__ __device__ float Clamp(float val,float start,float end) {

			return max(min(val, end), start);
	}

	

	void stencil_cpu(float *_out, float *_in, int width, int height, int depth) {

		#define out(i, j, k) _out[(( i )*width + (j)) * depth + (k)]
		#define in(i, j, k) _in[(( i )*width + (j)) * depth + (k)]

		float result;
		for (int i = 1; i < height - 1; ++i) {
			for (int j = 1; j < width - 1; ++j) {
				for (int k = 1; k < depth - 1; ++k) {
					result = in(i, j, k + 1) + in(i, j, k - 1) + in(i, j + 1, k) +
					in(i, j - 1, k) + in(i + 1, j, k) + in(i - 1, j, k) -
					6 * in(i, j, k);
					
					out(i,j,k)= Clamp(result, 0, 255);
					/*
					if(j==1&&k==1&&i==2046){
						printf("\n*********CPU*******");
						printf("Bottom: %f\t Current: %f\t Top: %f",in(i - 1, j, k),in(i, j, k),in(i+1, j, k));
						printf("\nX-1: %f \t X+1: %f",in(i, j-1, k),in(i , j+1, k));
						printf("\ny-1: %f \t y+1: %f",in(i, j, k-1),in(i , j, k+1));
						printf("\nOut value is%d,%d,%d: %f\t result: %f of %d",i,j,k,out(i,j,k),result,j);
					}
					*/
					/* out(i, j, k) = in(i, j, k + 1) + in(i, j, k - 1) + in(i, j + 1, k) +
					in(i, j - 1, k) + in(i + 1, j, k) + in(i - 1, j, k) -
					6 * in(i, j, k);*/
				}
			}
		}
		/*
		*	Debugging For CPU
		*/
		
		
		#undef out
		#undef in
	}

	__global__ void stencil(float *output, float *input, int width, int height,
	int depth) {

		const int TILE_WIDTH = 32;
		__shared__ float shared_array[TILE_WIDTH][TILE_WIDTH];
		
		#define out(i, j, k) output[(( i )*width + (j)) * depth + (k)]
		#define in(i, j, k) input[(( i )*width + (j)) * depth + (k)]
		/*To write main code here.*/
		/*
		*	Declare Variables
		*/
		int j=0;
		float result = 0;
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int idz = threadIdx.y + blockIdx.y * blockDim.y;
		
		/*
		*	implementation for i width, j height, k depth
		*/
		
		float bottom = in(0,idx,idz);
		
		float current = in(1,idx,idz);
		float top = in(2,idx,idz);
		
		shared_array[threadIdx.y][threadIdx.x] = current;
		float x0,x2,y0,y2;
		
		/*Running loop for Y-memory coarsning and x-z memory sharing.*/
		int count =0;
		
		__syncthreads();
		
		for(j=1;j<height-1;j++){
			/*
			*	If Boundary Conditions we come in IF, otherwise ELSE
			*/
			if((idx==0 && idz==0)||(idx==0 && idz==depth-1)||(idx==width-1 && idz == 0)||(idx==width-1 && idz==depth-1)){
				
				//printf("\tJ value is %d, depth %d height %d",j,depth,height);
				shared_array[threadIdx.y][threadIdx.x] = in(j,idx,idz);
			}
			else if((threadIdx.x>width-1)||(threadIdx.y>depth-1)){
				//this is invalid thread Index. Do nothing
			}
			else {
				/*
				*	This is the issue, we access the shared Array for thread ID 0, which value is not there at all.
				*/
				__syncthreads();
				x0 = (threadIdx.x>0)?shared_array[threadIdx.y][threadIdx.x-1]:(idx==0)?0:in(j,idx-1,idz);
				x2 = (threadIdx.x<blockDim.x-1)?shared_array[threadIdx.y][threadIdx.x+1]:(idx>=width-1)?0:in(j,idx+1,idz);
				y0 = (threadIdx.y>0)?shared_array[threadIdx.y-1][threadIdx.x]:(idz==0)?0:in(j,idx,idz-1);
				y2 = (threadIdx.y<blockDim.y-1)?shared_array[threadIdx.y+1][threadIdx.x]:(idz>=depth-1)?0:in(j,idx,idz+1);
				result = top+bottom+ x0+ x2+ y0+ y2- 6*current;
				
				
				
				
				__syncthreads();
				/*reintialize top,bottom,current variables*/
				if(idx==1&&idz==1){
					//printf("\nBefore update");	
				}
				
				bottom = current;
				current = top;
				shared_array[threadIdx.y][threadIdx.x] = current;
				
				__syncthreads();
				if(j+1<height-1){
					top = in(j+2,idx,idz);
					
				}
				//out(idx,j,idz) = result;
				__syncthreads();

				if((idx>0&&idx<(width-1))&&(idz>0&&idz<(depth-1))&&(j>0&&j<(height-1)))
					out(j,idx,idz) = Clamp(result, 0, 255);
			
				__syncthreads();
			}
		
		}
	
		#undef out
		#undef in
		

	}

	
	static void launch_stencil(float *deviceOutputData, float *deviceInputData,
	int width, int height, int depth) {
		//@@ INSERT CODE HERE
			dim3 DimGrid(ceil((width-1)/32.0),ceil((depth-1)/32.0),1);
			dim3 DimBlock(32,32,1);
		
			stencil<<<DimGrid,DimBlock>>>(deviceOutputData,deviceInputData,width,height,depth);
	}

	int main(int argc, char *argv[]) {
		
		
		wbArg_t arg;
		int width;
		int height;
		int depth;
		char *inputFile;
		wbImage_t input;
		wbImage_t output;
		wbImage_t output2;
		float *hostInputData;
		float *hostOutputData;
		float *deviceInputData;
		float *deviceOutputData;
		float *hostCPUOutputData;
		arg = wbArg_read(argc, argv);

		inputFile = wbArg_getInputFile(arg, 0);

		input = wbImport(inputFile);

		width = wbImage_getWidth(input);
		height = wbImage_getHeight(input);
		depth = wbImage_getChannels(input);

		output = wbImage_new(width, height, depth);
		output2 = wbImage_new(width, height, depth);
		hostInputData = wbImage_getData(input);
		hostOutputData = wbImage_getData(output);
		hostCPUOutputData = wbImage_getData(output2);
		/*Code for Serilized Input.*/
		
			wbLog(TRACE, "Doing CPU ");
			wbTime_start(GPU, "Performing CPU Scatter computation");
			stencil_cpu(hostCPUOutputData, hostInputData, width, height, depth);
			
			wbTime_stop(GPU,"Done with CPU run");
		
			//wbSolution(arg, output);

		/*End of code*/
		
		wbTime_start(GPU, "Doing GPU memory allocation");
		cudaMalloc(( void ** )&deviceInputData,
		width * height * depth * sizeof(float));
		cudaMalloc(( void ** )&deviceOutputData,
		width * height * depth * sizeof(float));
		wbTime_stop(GPU, "Doing GPU memory allocation");

		wbTime_start(Copy, "Copying data to the GPU");
		cudaMemcpy(deviceInputData, hostInputData,
		width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
		wbTime_stop(Copy, "Copying data to the GPU");

		wbTime_start(Compute, "Doing the computation on the GPU");

		//stencil_cpu(hostOutputData, hostInputData, width, height, depth);
		launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
		wbTime_stop(Compute, "Doing the computation on the GPU");

		wbTime_start(Copy, "Copying data from the GPU");
		wbCheck(cudaMemcpy(hostOutputData, deviceOutputData,
		width * height * depth * sizeof(float), cudaMemcpyDeviceToHost));
		wbTime_stop(Copy, "Copying data from the GPU");
		/*
		*	Debugging For GPU
		*/
		/*
		
		wbLog(TRACE, "The value of x = ", hostOutputData[(width*2+2)*depth+(2)]);
		
		wbLog(TRACE, "The value of  current= ", hostInputData[(width+1)*depth+1]);
		wbLog(TRACE, "The value of top = ", hostInputData[((width*2)+1)*depth+1]);
		wbLog(TRACE, "The value of vottom = ", hostInputData[(1)*depth+1]);
		wbLog(TRACE, "The value of x+1= ", hostInputData[(width+2)*depth+1]);
		wbLog(TRACE, "The value of x-1 = ", hostInputData[(width)*depth+1]);
		wbLog(TRACE, "The value of z-1 = ", hostInputData[(width+1)*depth]);
		wbLog(TRACE, "The value of z+1 = ", hostInputData[(width+1)*depth+2]);
		*/
		wbSolution(arg, output);

		cudaFree(deviceInputData);
		cudaFree(deviceOutputData);
		
		wbImage_delete(output);
		wbImage_delete(input);

		return 0;
	}
