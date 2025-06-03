#include <lib/common.h>
#include <mpi.h>

void mpi_ftcs(float *input, float *output, int local_nx, int rank, int size,
              float L, float alpha, float t_final, int n_x_global,
              int n_t_steps);

void run(int argc, char **argv, float L, float alpha, float t_final,
         int n_x_global, int n_t_steps);
