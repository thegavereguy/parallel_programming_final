#include <fmt/base.h>
#include <lib/distributed.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

int main(int argc, char **argv) {
  if (argc < 6) {
    // Stampato solo da rank 0 per evitare output multipli
    // if (MPI_Comm_rank(MPI_COMM_WORLD, &argc) == 0) {
    fprintf(stderr, "Usage: %s <L> <alpha> <t_final> <n_x> <n_t>\n", argv[0]);
    //}
    MPI_Finalize();

    return 1;
  }

  const float L_val       = std::stof(argv[1]);
  const float alpha_val   = std::stof(argv[2]);
  const float t_final_val = std::stof(argv[3]);
  const int n_x_val       = std::stoi(argv[4]);
  const int n_t_val       = std::stoi(argv[5]);

  run(argc, argv, L_val, alpha_val, t_final_val, n_x_val,
                    n_t_val);

  return 0;
}
