#include <fmt/base.h>
#include <lib/distributed.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <numeric>
#include <string>

int main(int argc, char **argv) {
  if (argc > 2) {
    if (argc != 6) {
      fprintf(stderr, "Usage: %s <L> <alpha> <t_final> <n_x> <n_t>\n", argv[0]);
      return 1;
    } else {
      const float L_val       = std::stof(argv[1]);
      const float alpha_val   = std::stof(argv[2]);
      const float t_final_val = std::stof(argv[3]);
      const int n_x_val       = std::stoi(argv[4]);
      const int n_t_val       = std::stoi(argv[5]);

      // se i parametri sono passati chiamare singolarmente, altrimenti loopare
      // i testcases e far stampare tutto
      run_single(argc, argv, L_val, alpha_val, t_final_val, n_x_val, n_t_val);
    }
  } else {
    if (argc != 2) {
      fprintf(stderr, "Usage: %s <n_samples>\n", argv[0]);
      return 2;
    }

    int rank, size;
    int n_samples      = std::stoi(argv[1]);
    float current_mean = 0.0f;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) fmt::print("NAME,NX,NT,MEAN,MINT,MAXT,ITER\n");

    const std::vector<std::pair<Conditions, std::string>> t = {
        {{1, .5, 0.01, 16, 30}, "test"}};

    for (auto c : target_cases) {
      float L_val       = c.first.L;
      float alpha_val   = c.first.alpha;
      float t_final_val = c.first.t_final;
      int n_x_val       = c.first.n_x;
      int n_t_val       = c.first.n_t;

      std::vector<double> timings_ms;
      // if (rank == 0) {
      //   timings_ms.reserve(n_samples);
      // }

      if (rank == 0) {
        // fmt::print(
        //     "Running with L = {}, alpha = {}, t_final = {}, n_x = {}, n_t = "
        //     "{}\n",
        //     L_val, alpha_val, t_final_val, n_x_val, n_t_val);
      }

      for (int i = 0; i < n_samples; ++i) {
        float time =
            run_multiple(L_val, alpha_val, t_final_val, n_x_val, n_t_val);
        timings_ms.push_back(time);
        // ideally implemnt the check if the time is withing a confidence
        // interval, TODO
      }
      if (rank == 0) {
        double sum = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0);
        double mean_time = sum / timings_ms.size();
        double min_time =
            *std::min_element(timings_ms.begin(), timings_ms.end());
        double max_time =
            *std::max_element(timings_ms.begin(), timings_ms.end());
        fmt::print("{},{},{},{:.4f},{:.4f},{:.4f},{}\n", c.second, n_x_val,
                   n_t_val, mean_time, min_time, max_time, 1);
      }
    }
    MPI_Finalize();
  }

  return 0;
}
