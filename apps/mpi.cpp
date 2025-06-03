#include <fmt/base.h>
#include <lib/common.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <utility>

void heat_equation_mpi_corrected(float *input, float *output, int local_nx,
                                 int rank, int size, float L, float alpha,
                                 float t_final, int n_x_global, int n_t_steps) {
  float dt     = t_final / (static_cast<float>(n_t_steps) - 1.0f);
  float dx     = L / (static_cast<float>(n_x_global) - 1.0f);
  float factor = alpha * dt / (dx * dx);

  MPI_Request request[4];
  MPI_Status status[4];

  for (int t = 0; t < n_t_steps; t++) {
    if (rank > 0) {
      MPI_Isend(&input[1], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD,
                &request[0]);
      MPI_Irecv(&input[0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD,
                &request[1]);
    }
    if (rank < size - 1) {
      MPI_Isend(&input[local_nx], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD,
                &request[2]);
      MPI_Irecv(&input[local_nx + 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD,
                &request[3]);
    }

    for (int i = 2; i < local_nx; i++) {  // Indici da 2 a local_nx - 1
      output[i] =
          input[i] + factor * (input[i - 1] - 2.0f * input[i] + input[i + 1]);
    }

    if (rank > 0) {
      MPI_Waitall(2, request, status);
    }
    if (rank < size - 1) {
      MPI_Waitall(2, &request[2], status);
    }

    if (local_nx > 0) {
      float left_ghost = (rank == 0) ? input[1] : input[0];

      output[1] = input[1] + factor * (left_ghost - 2.0f * input[1] + input[2]);
    }

    if (local_nx > 1) {
      float right_ghost =
          (rank == size - 1) ? input[local_nx] : input[local_nx + 1];
      output[local_nx] =
          input[local_nx] +
          factor * (input[local_nx - 1] - 2.0f * input[local_nx] + right_ghost);
    }

    if (rank == 0) {
      if (local_nx > 0)
        output[1] = 100.0f;  // Valore fisso al bordo sinistro globale
    }
    if (rank == size - 1) {
      if (local_nx > 0)
        output[local_nx] = 200.0f;  // Valore fisso al bordo destro globale
    }

    std::swap(input, output);
  }
}

void run_mpi_corrected(int argc, char **argv, float L, float alpha,
                       float t_final, int n_x_global, int n_t_steps) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int local_nx  = n_x_global / size;
  int remainder = n_x_global % size;
  if (rank < remainder) {
    local_nx++;
  }

  float *input  = nullptr;
  float *output = nullptr;

  if (local_nx > 0) {
    input  = new float[local_nx + 2];
    output = new float[local_nx + 2];

    int my_global_idx_start = 0;
    for (int r_iter = 0; r_iter < rank; ++r_iter) {
      my_global_idx_start += (n_x_global / size) + (r_iter < remainder ? 1 : 0);
    }

    for (int i = 1; i <= local_nx; i++) {
      int current_point_global_idx =
          my_global_idx_start + (i - 1);  // Indice globale 0-based

      if (current_point_global_idx == 0) {
        input[i] = 100.0f;
      } else if (current_point_global_idx == n_x_global - 1) {
        input[i] = 200.0f;
      } else {
        input[i] = 0.0f;
      }
    }

    input[0]            = 0.0f;  // Ghost cell sinistro
    input[local_nx + 1] = 0.0f;  // Ghost cell destro

    // Copia l'input iniziale nell'output per il primo swap se heat_equation si
    // aspetta output come buffer di lavoro Tuttavia, con lo swap all'inizio del
    // ciclo in heat_equation (o alla fine), output non necessita di essere
    // inizializzato con i dati di input qui. Basterebbe inizializzare output a
    // zero o qualsiasi valore. Per sicurezza, possiamo copiare input in output
    // prima della chiamata, o assicurare che heat_equation_mpi_corrected
    // gestisca correttamente il primo swap. Lo swap alla fine del ciclo in
    // heat_equation_mpi_corrected è standard.
    for (int i = 0; i < local_nx + 2; ++i) {
      output[i] = 0.0f;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  if (local_nx > 0) {
    heat_equation_mpi_corrected(input, output, local_nx, rank, size, L, alpha,
                                t_final, n_x_global, n_t_steps);
  }
  MPI_Barrier(
      MPI_COMM_WORLD);  // Sincronizza prima di fermare il cronometraggio
  double end_time = MPI_Wtime();

  if (rank == 0) {
    // Se si volesse raccogliere l'array completo e local_nx varia:
    // 1. Allocare float* global_result = new float[n_x_global];
    // 2. Preparare int* recvcounts = new int[size]; (con i local_nx di ogni
    // rank)
    // 3. Preparare int* displs = new int[size]; (con gli offset di ogni rank)
    // 4. Chiamare MPI_Gatherv(&input[1] (o output[1]), local_nx, MPI_FLOAT,
    // global_result, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD); Qui,
    // per semplicità e coerenza con l'output richiesto dallo script di
    // benchmark, stampiamo solo le informazioni sul tempo. Il `MPI_Gather`
    // originale raccoglieva local_nx (del rank 0) da ogni processo, il che non
    // è corretto per ricostruire l'array. Non lo includiamo qui per evitare
    // confusione, dato che lo script di benchmark si basa sulla stampa del
    // tempo.

    float millis = static_cast<float>((end_time - start_time) * 1000.0);
    fmt::print("{},{},{:.4f},{:.4f},{:.4f},{}\n", n_x_global, n_t_steps, millis,
               millis, millis, 1);
  }
  for (int i = 0; i < size; ++i) {
    if (rank == i) {
      fmt::print("Rank {}: local_nx = {}\n", rank, local_nx);
      for (int j = 0; j < local_nx; ++j) {
        fmt::print("Rank {}: output[{}] = {:.4f}\n", rank, j + 1,
                   output[j + 1]);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);  // Sincronizza per evitare output misti
  }

  if (local_nx > 0) {
    delete[] input;
    delete[] output;
  }
  MPI_Finalize();
}

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

  run_mpi_corrected(argc, argv, L_val, alpha_val, t_final_val, n_x_val,
                    n_t_val);

  return 0;
}
