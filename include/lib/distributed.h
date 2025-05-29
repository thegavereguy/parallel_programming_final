#include <lib/common.h>
#include <openmpi-x86_64/mpi.h>

void sequential_explicit(float *input, float *output, int local_nx, int rank,
                         int size, Conditions);
int run(int, char **);
