/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    // Variables
    float *in_val_h;
    float *in_pos_h;
    float *out_h;
    float *in_val_d;
    float *in_pos_d;
    float *out_d;
    unsigned int grid_size, num_in;
    cudaError_t cuda_ret;

    // Constants
    const unsigned int maxVal = 1; // Maximum input value
    const float cutoff = 3000.0f; // Cutoff distance for optimized computation
    const float cutoff2 = cutoff*cutoff;

   // Extras needed for input binning
   unsigned int* binCounts_h;
   unsigned int* binPtrs_h;
   float* in_val_sorted_h;
   float* in_pos_sorted_h;
   unsigned int* binCounts_d;
   unsigned int* binPtrs_d;
   float* in_val_sorted_d;
   float* in_pos_sorted_d;

    enum Mode {CPU_NORMAL = 1, GPU_NORMAL, GPU_CUTOFF,
        GPU_BINNED_CPU_PREPROCESSING, GPU_BINNED_GPU_PREPROCESSING};
    Mode mode;

    if(argc == 2) {
        mode = (Mode) atoi(argv[1]);
        grid_size = 20000;
        num_in = 60000;
    } else if(argc == 3) {
        mode = (Mode) atoi(argv[1]);
        grid_size = atoi(argv[2]);
        num_in = 3*grid_size;
    } else if(argc == 4) {
        mode = (Mode) atoi(argv[1]);
        grid_size = atoi(argv[2]);
        num_in = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters."
        "\n"
        "\n    Usage: ./binning <m>          # Mode: m, Grid: 20,000, Input: 60,000"
        "\n           ./binning <m> <M>      # Mode: m, Grid:      M, Input:    3*M"
        "\n           ./binning <m> <M> <N>  # Mode: m, Grid:      M, Input:      N"
        "\n"
        "\n    Modes: 1 = CPU normal execution"
        "\n           2 = GPU normal execution"
        "\n           3 = GPU with cutoff"
        "\n           4 = GPU with cutoff and binned input (CPU preprocessing)"
        "\n           5 = GPU with cutoff and binned input (GPU preprocessing)"
        "\n\n");
        exit(0);
    }

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    initVector(&in_val_h, num_in, maxVal);

    initVector(&in_pos_h, num_in, grid_size);

    out_h = (float*) malloc(grid_size*sizeof(float));
    memset((void*) out_h, 0, grid_size*sizeof(float));

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Grid size = %u\n    Input size = %u\n", grid_size, num_in);

    // CPU Preprocessing ------------------------------------------------------

    if(mode == GPU_BINNED_CPU_PREPROCESSING) {

        printf("Preprocessing data on the CPU..."); fflush(stdout);
        startTime(&timer);

        // Data structures needed to preprocess the bins on the CPU
        binCounts_h = (unsigned int*) malloc(NUM_BINS*sizeof(unsigned int));
        memset((void*) out_h, 0, grid_size*sizeof(float));
        binPtrs_h = (unsigned int*) malloc((NUM_BINS + 1)*sizeof(unsigned int));
        in_val_sorted_h = (float*) malloc(num_in*sizeof(float));
        in_pos_sorted_h = (float*) malloc(num_in*sizeof(float));

        cpu_preprocess(in_val_h, in_pos_h, in_val_sorted_h, in_pos_sorted_h,
            grid_size, num_in, binCounts_h, binPtrs_h);

        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }

    // Allocate device variables ----------------------------------------------

    if(mode != CPU_NORMAL) {
        printf("Allocating device variables..."); fflush(stdout);
        startTime(&timer);

        // If preprocessing on the CPU, GPU doesn't need the unsorted arrays
        if(mode != GPU_BINNED_CPU_PREPROCESSING) {
            cuda_ret = cudaMalloc((void**)&in_val_d, num_in * sizeof(float));
            if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
            cuda_ret = cudaMalloc((void**)&in_pos_d, num_in * sizeof(float));
            if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");
        }

        // All modes need the output array
        cuda_ret = cudaMalloc((void**)&out_d, grid_size * sizeof(float));
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

        // Only binning modes need binning information
        if(mode == GPU_BINNED_CPU_PREPROCESSING ||
            mode == GPU_BINNED_GPU_PREPROCESSING) {

            cuda_ret = cudaMalloc((void**)&in_val_sorted_d,
                num_in*sizeof(float));
            if(cuda_ret != cudaSuccess) {
                FATAL("Unable to allocate device memory");
            }

            cuda_ret = cudaMalloc((void**)&in_pos_sorted_d,
                num_in*sizeof(float));
            if(cuda_ret != cudaSuccess) {
                FATAL("Unable to allocate device memory");
            }

            cuda_ret = cudaMalloc((void**)&binPtrs_d,
                (NUM_BINS + 1)*sizeof(unsigned int));
            if(cuda_ret != cudaSuccess) {
                FATAL("Unable to allocate device memory");
            }

            if(mode == GPU_BINNED_GPU_PREPROCESSING) {
                // Only used in preprocessing but not the actual computation
                cuda_ret = cudaMalloc((void**)&binCounts_d,
                    NUM_BINS*sizeof(unsigned int));
                if(cuda_ret != cudaSuccess) {
                    FATAL("Unable to allocate device memory");
                }
            }

        }

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }

    // Copy host variables to device ------------------------------------------

    if(mode != CPU_NORMAL) {
        printf("Copying data from host to device..."); fflush(stdout);
        startTime(&timer);

        // If preprocessing on the CPU, GPU doesn't need the unsorted arrays
        if(mode != GPU_BINNED_CPU_PREPROCESSING) {
            cuda_ret = cudaMemcpy(in_val_d, in_val_h, num_in * sizeof(float),
                cudaMemcpyHostToDevice);
            if(cuda_ret != cudaSuccess) {
                FATAL("Unable to copy memory to the device");
            }

            cuda_ret = cudaMemcpy(in_pos_d, in_pos_h, num_in * sizeof(float),
                cudaMemcpyHostToDevice);
            if(cuda_ret != cudaSuccess) {
                FATAL("Unable to copy memory to the device");
            }
        }

        // All modes need the output array
        cuda_ret = cudaMemset(out_d, 0, grid_size * sizeof(float));
        if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

        if(mode == GPU_BINNED_CPU_PREPROCESSING) {

            cuda_ret = cudaMemcpy(in_val_sorted_d, in_val_sorted_h,
                num_in * sizeof(float), cudaMemcpyHostToDevice);
            if(cuda_ret != cudaSuccess) {
                FATAL("Unable to copy memory to the device");
            }

            cuda_ret = cudaMemcpy(in_pos_sorted_d, in_pos_sorted_h,
                num_in * sizeof(float), cudaMemcpyHostToDevice);
            if(cuda_ret != cudaSuccess) {
                FATAL("Unable to copy memory to the device");
            }

            cuda_ret = cudaMemcpy(binPtrs_d, binPtrs_h,
                (NUM_BINS + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
            if(cuda_ret != cudaSuccess) {
                FATAL("Unable to copy memory to the device");
            }

        } else if(mode == GPU_BINNED_GPU_PREPROCESSING) {
            // If preprocessing on the GPU, bin counts need to be initialized
            //  and nothing needs to be copied
            cuda_ret=cudaMemset(binCounts_d, 0, NUM_BINS*sizeof(unsigned int));
            if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");
        }

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }

    // GPU Preprocessing ------------------------------------------------------

    if(mode == GPU_BINNED_GPU_PREPROCESSING) {

        printf("Preprocessing data on the GPU..."); fflush(stdout);
        startTime(&timer);

        gpu_preprocess(in_val_d, in_pos_d, in_val_sorted_d, in_pos_sorted_d,
            grid_size, num_in, binCounts_d, binPtrs_d);

        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    }

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel ");

    if(mode == CPU_NORMAL) {
        printf("(CPU normal version)...");fflush(stdout);
        startTime(&timer);
        cpu_normal(in_val_h, in_pos_h, out_h, grid_size, num_in);
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else if(mode == GPU_NORMAL) {
        printf("(GPU normal version)...");fflush(stdout);
        startTime(&timer);
        gpu_normal(in_val_d, in_pos_d, out_d, grid_size, num_in);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else if(mode == GPU_CUTOFF) {
        printf("(GPU with cuttoff)...");fflush(stdout);
        startTime(&timer);
        gpu_cutoff(in_val_d, in_pos_d, out_d, grid_size, num_in, cutoff2);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else if(mode == GPU_BINNED_CPU_PREPROCESSING ||
        mode == GPU_BINNED_GPU_PREPROCESSING) {
        printf("(GPU with input binning and cutoff)...");fflush(stdout);
        startTime(&timer);
        gpu_cutoff_binned(binPtrs_d, in_val_sorted_d, in_pos_sorted_d, out_d,
            grid_size, cutoff2);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else {
        printf("Invalid mode!\n");
        exit(0);
    }


    // Copy device variables from host ----------------------------------------

    if(mode != CPU_NORMAL) {

        printf("Copying data from device to host..."); fflush(stdout);
        startTime(&timer);

        cuda_ret = cudaMemcpy(out_h, out_d, grid_size * sizeof(float),
            cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    }

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    if(mode == CPU_NORMAL || mode == GPU_NORMAL) {
        verify(in_val_h, in_pos_h, out_h, grid_size, num_in);
    } else {
        verify_cutoff(in_val_h, in_pos_h, out_h, grid_size, num_in, cutoff2);
    }

    // Free memory ------------------------------------------------------------

    free(in_val_h); free(in_pos_h); free(out_h);
    if(mode == GPU_BINNED_CPU_PREPROCESSING) {
        free(binCounts_h); free(binPtrs_h);
        free(in_val_sorted_h); free(in_pos_sorted_h);
    }
    if(mode != CPU_NORMAL) {
        if(mode != GPU_BINNED_CPU_PREPROCESSING) {
            cudaFree(in_val_d); cudaFree(in_pos_d);
        }
        cudaFree(out_d);
        if(mode == GPU_BINNED_CPU_PREPROCESSING ||
            mode == GPU_BINNED_GPU_PREPROCESSING) {
            cudaFree(in_val_sorted_d); cudaFree(in_pos_sorted_d);
            cudaFree(binPtrs_d);
            if(mode == GPU_BINNED_GPU_PREPROCESSING) {
                cudaFree(binCounts_d);
            }

        }
    }

    return 0;
}

