/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE/WARP_SIZE)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 128

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {
	
	
	unsigned int iterator;
	//unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int nodeInformation;
	unsigned int iteratorNode;
	unsigned int count;
	
  unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	for(unsigned int idx = tid; idx <  *numCurrLevelNodes; idx += blockDim.x*gridDim.x) {
	  // Do your computation here
	
  
  
  // INSERT KERNEL CODE HERE

	//for every node in the queue, we run this.
	count = 0;
	if(idx< *numCurrLevelNodes)
	{
		nodeInformation = currLevelNodes[idx];
		
		//loop over all the neighbours
		for(iterator=nodePtrs[nodeInformation];iterator<nodePtrs[nodeInformation+1];iterator++)
		{
			iteratorNode = nodeNeighbors[iterator];
			if(!atomicAdd(&nodeVisited[iteratorNode],1))
			{
				//if it has not been visited.
				nextLevelNodes[atomicAdd(&(*numNextLevelNodes),1)] = iteratorNode;
				count++;
			}
		}
	}
	}
	//__syncthreads();
}



__global__ void gpu_block_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
	//have a queue in local thread copy. When thread computation completed, 
	//Initialization of Variables
	
	__shared__ unsigned int localQueue[BQ_CAPACITY];
	__shared__ unsigned int totalCount;
	
	__shared__ unsigned int numberOfElementsToCopy;		
	__shared__ unsigned int queueStart;
	
	int iterator;
	
	unsigned int nodeInformation;
	unsigned int iteratorNode;

	
	int tid = threadIdx.x+blockIdx.x * blockDim.x;
	//Initializing queue count
	
	if(threadIdx.x == 0)
	{
		totalCount = 0;
	}
		
		
		
	__syncthreads();
	
	//sync all the threads at this point as 
	
	for(unsigned int idx = tid; idx <  *numCurrLevelNodes; idx += blockDim.x*gridDim.x) {
	//for every node in the queue, we run this.
	
	nodeInformation = currLevelNodes[idx];
		
		//loop over all the neighbours
		for(iterator=nodePtrs[nodeInformation];iterator<nodePtrs[nodeInformation+1];iterator++)
		{
			iteratorNode = nodeNeighbors[iterator];
			if(!atomicAdd(&nodeVisited[iteratorNode],1))
			{
				//if it has not been visited.
				int previousNodeVal = atomicAdd(&(totalCount),1);
				
				if (previousNodeVal < BQ_CAPACITY){
					//if the load is less than capacty. Load this in the local queue of thread
					localQueue[previousNodeVal] = iteratorNode;
				}
				else {
					//if block queue is saturated, we cannot place next level node in local queue.
					  nextLevelNodes[atomicAdd(&(*numNextLevelNodes),1)] = iteratorNode;
						totalCount = BQ_CAPACITY;	// if total count exceeds the capacity.
				}
				
				
				
			}
		}
	
	}
	
	//preprocessing per local thread complete. Now, syncing threads
	__syncthreads();
	
	//for first thread, increment the global pointer for the next block.
	if(blockIdx.x*blockDim.x < *numCurrLevelNodes)
	{
		//only let the first block to make changes globally.
		if(threadIdx.x == 0)
		{
			//make the pointers incremented to the next level. So global threads can write parallely.
			queueStart = atomicAdd(&(*numNextLevelNodes),totalCount);
			
			//how many number should be copied.
			numberOfElementsToCopy = ((totalCount-1)/BLOCK_SIZE +1);
		}
		
		__syncthreads();
		
		
		//copy elements on a per thread basis.
		for(unsigned int iter = 0;iter < numberOfElementsToCopy; iter++)
		{
			int index = threadIdx.x*numberOfElementsToCopy;
			if( index + iter < totalCount)
			{
				//no element more than this should be copied.
				nextLevelNodes[queueStart + index + iter] = localQueue[index+iter];
			}
		
		}
	}

}

__global__ void gpu_warp_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE



}

/******************************************************************************
 Functions
*******************************************************************************/

void cpu_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  // Loop over all nodes in the curent level
  for(unsigned int idx = 0; idx < *numCurrLevelNodes; ++idx) {
    unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for(unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
      ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      if(!nodeVisited[neighbor]) {
        // Mark it and add it to the queue
        nodeVisited[neighbor] = 1;
        nextLevelNodes[*numNextLevelNodes] = neighbor;
        ++(*numNextLevelNodes);
      }
    }
  }

}

void gpu_global_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

void gpu_block_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

void gpu_warp_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

