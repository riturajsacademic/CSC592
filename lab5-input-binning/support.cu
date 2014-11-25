/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void initVector(float **vec_h, unsigned int size, unsigned int max)
{
    *vec_h = (float*)malloc(size*sizeof(float));

    if(*vec_h == NULL) {
        FATAL("Unable to allocate host");
    }

    for (unsigned int i=0; i < size; i++) {
        (*vec_h)[i] = (((float)rand())/RAND_MAX)*max;
    }

}

void verify(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in) {

  // Initialize reference
  float* out_ref = (float*)malloc(grid_size*sizeof(unsigned int));
  for(unsigned int outIdx = 0; outIdx < grid_size; ++outIdx) {
      out_ref[outIdx] = 0.0f;
  }

  // Compute reference out
  for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
      const float in_val2 = in_val[inIdx]*in_val[inIdx];
      for(unsigned int outIdx = 0; outIdx < grid_size; ++outIdx) {
          const float dist = in_pos[inIdx] - (float) outIdx;
          out_ref[outIdx] += in_val2/(dist*dist);
      }
  }

  // Compare to reference out
  float tolerance = 1e-3;
  for(unsigned int outIdx = 0; outIdx < grid_size; ++outIdx) {
      const float diff = (out[outIdx] - out_ref[outIdx])/out_ref[outIdx];
      if(diff > tolerance || diff < -tolerance) {
        printf("TEST FAILED at output index %u, reference = %f, computed = %f"
          "\n\n", outIdx, out_ref[outIdx], out[outIdx]);
        exit(0);
      }
  }
  printf("TEST PASSED\n\n");

  free(out_ref);

}

void verify_cutoff(float* in_val, float* in_pos, float* out,
    unsigned int grid_size, unsigned int num_in, float cutoff2) {

  // Initialize reference
  float* out_ref = (float*)malloc(grid_size*sizeof(unsigned int));
  for(unsigned int outIdx = 0; outIdx < grid_size; ++outIdx) {
      out_ref[outIdx] = 0.0f;
  }

  // Compute reference out
  for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
      const float in_val2 = in_val[inIdx]*in_val[inIdx];
      for(unsigned int outIdx = 0; outIdx < grid_size; ++outIdx) {
          const float dist = in_pos[inIdx] - (float) outIdx;
          const float dist2 = dist*dist;
          if(dist2 < cutoff2) {
              out_ref[outIdx] += in_val2/(dist*dist);
          }
      }
  }

  // Compare to reference out
  float tolerance = 1e-3;
  for(unsigned int outIdx = 0; outIdx < grid_size; ++outIdx) {
      const float diff = (out[outIdx] - out_ref[outIdx])/out_ref[outIdx];
      if(diff > tolerance || diff < -tolerance) {
        printf("TEST FAILED at output index %u, reference = %f, computed = %f"
          "\n\n", outIdx, out_ref[outIdx], out[outIdx]);
        exit(0);
      }
  }
  printf("TEST PASSED\n\n");

  free(out_ref);

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

