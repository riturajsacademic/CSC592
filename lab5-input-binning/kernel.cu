/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// For simplicity, fix #bins=1024 so scan can use a single block and no padding
#define NUM_BINS 1024

/******************************************************************************
 GPU main computation kernels
*******************************************************************************/

__global__ void gpu_normal_kernel(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in) {

    // INSERT KERNEL CODE HERE
	//taking the outIndex to use
	unsigned int outIdx  = threadIdx.x + blockIdx.x*blockDim.x;
	float inval2;
	float dist;
	float result;
	//going from 0 to num_in if the outIdx is less than grid size
	if(outIdx<grid_size)
	{
		for(unsigned int inIdx = 0;inIdx<num_in;++inIdx){
			inval2 = in_val[inIdx]*in_val[inIdx];
			dist = in_pos[inIdx]-(float)outIdx;
			result +=inval2/(dist*dist);
		}
		out[outIdx] = result;
	}

}

__global__ void gpu_cutoff_kernel(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in, float cutoff2) {

    // INSERT KERNEL CODE HERE


	unsigned int outIdx  = threadIdx.x + blockIdx.x*blockDim.x;
	float inval2;
	float dist;
	float result;
	float dist2;
	//going from 0 to num_in if the outIdx is less than grid size
	if(outIdx<grid_size)
	{
		for(unsigned int inIdx = 0;inIdx<num_in;++inIdx){
			
			dist = in_pos[inIdx]-(float)outIdx;
			dist2 = dist*dist;
			if(dist2<cutoff2)
			{
				inval2 = in_val[inIdx]*in_val[inIdx];
				result +=inval2/(dist*dist);
			}
		}
		out[outIdx] = result;
	}


}

__global__ void gpu_cutoff_binned_kernel(unsigned int* binPtrs,
    float* in_val_sorted, float* in_pos_sorted, float* out,
    unsigned int grid_size, float cutoff2) {

    // INSERT KERNEL CODE HERE
	//in this 

	//find the input bin for the specific point.
	unsigned int outIdx  = threadIdx.x + blockIdx.x*blockDim.x;
	
	float dist;
	float result=0;
	
	
	
	unsigned int iterator;
	float dist2;
	if(outIdx<grid_size)
	{
		
        for(unsigned int bin = 0; bin < NUM_BINS; ++bin) {
            unsigned int startOfBin = binPtrs[bin];
            unsigned int endOfBin = binPtrs[bin+1];
            dist = in_pos_sorted[startOfBin] - (float)outIdx;
            dist2 = dist * dist;
            if(dist2 <= cutoff2) {
			/*checking for if the bin starting point is less than the cutoff or not. */
				for(iterator = startOfBin; iterator < endOfBin; ++iterator) {
					float point_dist = in_pos_sorted[iterator] - (float)outIdx;
					float points_dist2 = point_dist * point_dist;
					if(points_dist2 <= cutoff2)
					result += (in_val_sorted[iterator] * in_val_sorted[iterator]) / points_dist2;
				}
	    }
	}
	out[outIdx] = result;
		
	}


}

/******************************************************************************
 Main computation functions
*******************************************************************************/

void cpu_normal(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in) {

    for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
        const float in_val2 = in_val[inIdx]*in_val[inIdx];
        for(unsigned int outIdx = 0; outIdx < grid_size; ++outIdx) {
            const float dist = in_pos[inIdx] - (float) outIdx;
            out[outIdx] += in_val2/(dist*dist);
        }
    }

}

void gpu_normal(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in) {

    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (grid_size - 1)/numThreadsPerBlock + 1;
    gpu_normal_kernel <<< numBlocks , numThreadsPerBlock >>>
        (in_val, in_pos, out, grid_size, num_in);

}

void gpu_cutoff(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in, float cutoff2) {

    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (grid_size - 1)/numThreadsPerBlock + 1;
    gpu_cutoff_kernel <<< numBlocks , numThreadsPerBlock >>>
        (in_val, in_pos, out, grid_size, num_in, cutoff2);

}

void gpu_cutoff_binned(unsigned int* binPtrs, float* in_val_sorted,
    float* in_pos_sorted, float* out, unsigned int grid_size, float cutoff2) {

    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (grid_size - 1)/numThreadsPerBlock + 1;
    gpu_cutoff_binned_kernel <<< numBlocks , numThreadsPerBlock >>>
        (binPtrs, in_val_sorted, in_pos_sorted, out, grid_size, cutoff2);

}


/******************************************************************************
 Preprocessing kernels
*******************************************************************************/

__global__ void histogram(float* in_pos, unsigned int* binCounts,
    unsigned int num_in, unsigned int grid_size) {

    // INSERT KERNEL CODE HERE

	unsigned int index = threadIdx.x+blockIdx.x*blockDim.x;
	if(index<num_in){
		unsigned int binIdx = (unsigned int) ((in_pos[index]/grid_size)*NUM_BINS);
		atomicAdd(&binCounts[binIdx],1);
	}

}


__global__ void scan(unsigned int* binCounts, unsigned int* binPtrs) {

    // INSERT KERNEL CODE HERE
	__shared__ unsigned int temp[NUM_BINS];  // allocated on invocation  
		int thid = threadIdx.x;  
		
		int offset = 1;  
		int n = NUM_BINS;
		
		temp[2*thid] = binCounts[2*thid]; // load input into shared memory  
		temp[2*thid+1] = binCounts[2*thid+1];  
		int ai,bi;
		
		for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree  
		{   
		__syncthreads();  
		   if (thid < d)  
		   {  
		   
				ai = offset*(2*thid+1)-1;  
				bi = offset*(2*thid+2)-1;  
				temp[bi] += temp[ai];  
			}  
			offset *= 2;  
		}
		if (thid == 0) { temp[n - 1] = 0; } // clear the last element
		
			
		for (int d = 1; d < n; d *= 2) // traverse down tree & build scan  
		{  
			 offset >>= 1;  
			 __syncthreads();  
			 if (thid < d)                       
			 {  
					ai = offset*(2*thid+1)-1;  
					bi = offset*(2*thid+2)-1; 
					float t = temp[ai];  
					temp[ai] = temp[bi];  
					temp[bi] += t; 
			}
			  
		}  
		 
		 __syncthreads(); 
		binPtrs[2*thid] = temp[2*thid]; // write results to device memory  
		binPtrs[2*thid+1] = temp[2*thid+1];  
		if(thid==511)
			binPtrs[1024] = binPtrs[1023]+binCounts[1023];  
				
		

}

__global__ void sort(float* in_val, float* in_pos, float* in_val_sorted,
    float* in_pos_sorted, unsigned int grid_size, unsigned int num_in,
    unsigned int* binCounts, unsigned int* binPtrs) {

    // INSERT KERNEL CODE HERE
	unsigned int inIdx = threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int newIdx;
	if(inIdx<num_in){
			unsigned int binIdx = (unsigned int) ((in_pos[inIdx]/grid_size)*NUM_BINS);
			
			newIdx = binPtrs[binIdx + 1] - atomicSub(&binCounts[binIdx],1);
			in_val_sorted[newIdx] = in_val[inIdx];
			in_pos_sorted[newIdx] = in_pos[inIdx];
	}


}

/******************************************************************************
 Preprocessing functions
*******************************************************************************/

void cpu_preprocess(float* in_val, float* in_pos, float* in_val_sorted,
    float* in_pos_sorted, unsigned int grid_size, unsigned int num_in,
    unsigned int* binCounts, unsigned int* binPtrs) {

    // Histogram the input positions
    for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
        const unsigned int binIdx =
            (unsigned int) ((in_pos[inIdx]/grid_size)*NUM_BINS);
        ++binCounts[binIdx];
    }

    // Scan the histogram to get the bin pointers
    binPtrs[0] = 0;
    for(unsigned int binIdx = 0; binIdx < NUM_BINS; ++binIdx) {
        binPtrs[binIdx + 1] = binPtrs[binIdx] + binCounts[binIdx];
    }

    // Sort the inputs into the bins
    for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
        const unsigned int binIdx =
            (unsigned int) ((in_pos[inIdx]/grid_size)*NUM_BINS);
        const unsigned int newIdx = binPtrs[binIdx + 1] - binCounts[binIdx];
        --binCounts[binIdx];
        in_val_sorted[newIdx] = in_val[inIdx];
        in_pos_sorted[newIdx] = in_pos[inIdx];
    }

}

void gpu_preprocess(float* in_val, float* in_pos, float* in_val_sorted,
    float* in_pos_sorted, unsigned int grid_size, unsigned int num_in,
    unsigned int* binCounts, unsigned int* binPtrs) {

    const unsigned int numThreadsPerBlock = 512;

    // Histogram the input positions
    histogram <<< ((num_in-1)/numThreadsPerBlock)+1 , numThreadsPerBlock >>>
        (in_pos, binCounts, num_in, grid_size);

    // Scan the histogram to get the bin pointers
    if(NUM_BINS != 1024) FATAL("NUM_BINS must be 1024. Do not change.");
    scan <<< 1 , numThreadsPerBlock >>> (binCounts, binPtrs);

    // Sort the inputs into the bins
    sort <<< ((num_in-1)/numThreadsPerBlock)+1 , numThreadsPerBlock >>> (in_val, in_pos, in_val_sorted,
        in_pos_sorted, grid_size, num_in, binCounts, binPtrs);

}


