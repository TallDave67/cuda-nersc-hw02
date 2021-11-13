#include <stdio.h>
#include <algorithm>

using namespace std;

// common utilities
#include "utility.h"

#define N 8192
#define RADIUS 5
#define BLOCK_SIZE 32

__global__ void stencil_1d(int *in, int *out) {
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS]; // includes 2 halos of size RADIUS, one for left side and one for right side
    // move each index past the halo by adding RADIUS
    int gindex = RADIUS + threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = RADIUS + threadIdx.x;

    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        // these threads responsible for reading in data for halo locations memory
        temp[lindex - RADIUS] = in[gindex - RADIUS]; // left halo
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE]; // right halo
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();


    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) // window of size 2 * RADIUS + 1
      result += temp[lindex + offset]; // window is centered at temp[lindex]

    // Store the result
    out[gindex] = result;

}

void fill_ints(int *x, int n) {
  fill_n(x, n, 1);
}

int main_stencil_1d(void) {
    // *******************************************
    // ********** ALLOCATE & INITIALIZE **********
    // *************** HOST MEMORY ***************
    // *******************************************

    int *in, *out; // host copies of a, b, c
    int size = (N + 2*RADIUS) * sizeof(int);
    //printf("sizeof(int)=%zd, size=%d\n", sizeof(int), size);
    in = (int *)malloc(size);
    out = (int *)malloc(size);
    //in = new int[N + 2*RADIUS];
    //out = new int[N + 2*RADIUS];   
    //
    fill_ints(in, N + 2*RADIUS);
    fill_ints(out, N + 2*RADIUS);

    // *******************************************
    // ***************** ALLOCATE ****************
    // ************** DEVICE MEMORY **************
    // *******************************************

    int *d_in, *d_out; // device copies of a, b, c
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaCheckErrors("cudaMalloc failure");

    // *******************************************
    // ****************** STEP 1 *****************
    // ************* COPY HOST MEMORY ************
    // ************* TO DEVICE MEMORY ************
    // *******************************************

    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // *******************************************
    // ****************** STEP 2 *****************
    // ****************** EXECUTE ****************
    // **************** DEVICE CODE **************
    // *******************************************

    // Launch stencil_1d() kernel on GPU
    stencil_1d<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in, d_out);
    cudaCheckErrors("kernel launch failure");

    // *******************************************
    // ****************** STEP 3 *****************
    // ************ COPY DEVICE MEMORY ***********
    // ************** TO HOST MEMORY *************
    // *******************************************

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    // *******************************************
    // **************** VERIFY AND ***************
    // ************** REPORT RESULTS *************
    // *******************************************

    constexpr int expected_ones = 2 * RADIUS;
    constexpr int expected_window_values = N;
    constexpr int window_value = 1 + 2 * RADIUS;
    printf("window_value=%d\n", window_value);
    int actual_ones = 0;
    int actual_window_values = 0;
    for (int i = 0; i < N + 2*RADIUS; i++) {
        if (i<RADIUS || i>=N+RADIUS){
            if (out[i] != 1)
            {
                printf("Output mismatch in halo at index %d, was: %d, should be: %d\n", i, out[i], 1);
            }
            else
            {
                actual_ones++;
            }
        } else {
            if (out[i] != window_value)
            {
                printf("Output mismatch in window at index %d, was: %d, should be: %d\n", i, out[i], window_value);
            }
            else
            {
                actual_window_values++;
            }
        }
    }
    printf("output => actual_ones : expected_ones (%d:%d), actual_window_values : expected_window_values (%d:%d)\n", actual_ones, expected_ones, actual_window_values, expected_window_values);

    // *******************************************
    // ******************* FREE ******************
    // ************** DEVICE MEMORY **************
    // *******************************************

    cudaFree(d_in);
    cudaFree(d_out);
    cudaCheckErrors("cudaFree failure");

    // *******************************************
    // ******************* FREE ******************
    // *************** HOST MEMORY ***************
    // *******************************************
  
    free(in);
    free(out);
    //delete[] in;
    //delete[] out;
  
    bool success = (actual_ones == expected_ones && actual_window_values == expected_window_values);
    printf("%s\n", success ? "Success" : "Failure");
    return 0;
}
