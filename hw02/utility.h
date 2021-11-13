#pragma once
#include <stdio.h>
#include <iostream>
#include <iomanip>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// double output
inline auto output_double(std::ostream& stream, double& value) -> std::ostream& {
    return stream << std::setprecision(8) << value;
}
