#include <lib/common.h>
#include <mpi.h>

void sequential_explicit(float *input, float *output, int local_nx, int rank,
                         int size, Conditions);
int run(int, char **);

void heat_equation_mpi_corrected(float *input, float *output, int local_nx,
                                 int rank, int size, float L, float alpha,
                                 float t_final, int n_x_global, int n_t_steps);

void run_mpi_corrected(int argc, char **argv, float L, float alpha,
                       float t_final, int n_x_global, int n_t_steps);
