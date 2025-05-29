#include <lib/shared.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#include "fmt/base.h"

void sequential_explicit(Conditions conditions, float* input, float* output) {
  output[0]                  = input[0];
  output[conditions.n_x - 1] = input[conditions.n_x - 1];
  float dt                   = conditions.t_final / (conditions.n_t - 1);
  float dx                   = conditions.L / (conditions.n_x - 1);

  for (int i = 0; i < conditions.n_t; i++) {
    for (int j = 1; j < conditions.n_x - 1; j++) {
      output[j] =
          input[j] + conditions.alpha * (dt / (dx * dx)) *
                         (input[j + 1] - 2 * input[j] + input[j - 1]);  // d^2u
    }
    std::swap(input, output);
  }
}

void parallel2_explicit(Conditions conditions, float* input, float* output) {
  output[0]                  = input[0];
  output[conditions.n_x - 1] = input[conditions.n_x - 1];
  float dt                   = conditions.t_final / (conditions.n_t - 1);
  float dx                   = conditions.L / (conditions.n_x - 1);

  for (int i = 0; i < conditions.n_t; i++) {
#pragma omp parallel for num_threads(2)
    for (int j = 1; j < conditions.n_x - 1; j++) {
      output[j] =
          input[j] + conditions.alpha * (dt / (dx * dx)) *
                         (input[j + 1] - 2 * input[j] + input[j - 1]);  // d^2u
    }
    std::swap(input, output);
  }
}
void parallel4_explicit(Conditions conditions, float* input, float* output) {
  output[0]                  = input[0];
  output[conditions.n_x - 1] = input[conditions.n_x - 1];
  float dt                   = conditions.t_final / (conditions.n_t - 1);
  float dx                   = conditions.L / (conditions.n_x - 1);

  for (int i = 0; i < conditions.n_t; i++) {
#pragma omp parallel for num_threads(4)
    for (int j = 1; j < conditions.n_x - 1; j++) {
      output[j] =
          input[j] + conditions.alpha * (dt / (dx * dx)) *
                         (input[j + 1] - 2 * input[j] + input[j - 1]);  // d^2u
    }
    std::swap(input, output);
  }
}
void parallel8_explicit(Conditions conditions, float* input, float* output) {
  output[0]                  = input[0];
  output[conditions.n_x - 1] = input[conditions.n_x - 1];
  float dt                   = conditions.t_final / (conditions.n_t - 1);
  float dx                   = conditions.L / (conditions.n_x - 1);

  for (int i = 0; i < conditions.n_t; i++) {
#pragma omp parallel for num_threads(8)
    for (int j = 1; j < conditions.n_x - 1; j++) {
      output[j] =
          input[j] + conditions.alpha * (dt / (dx * dx)) *
                         (input[j + 1] - 2 * input[j] + input[j - 1]);  // d^2u
    }
    std::swap(input, output);
  }
}

void sequential_unroll_explicit(Conditions conditions, float* input,
                                float* output) {
  output[0]                  = input[0];
  output[conditions.n_x - 1] = input[conditions.n_x - 1];
  float dt                   = conditions.t_final / (conditions.n_t - 1);
  float dx                   = conditions.L / (conditions.n_x - 1);

  for (int i = 0; i < conditions.n_t; i++) {
#pragma omp unroll partial
    for (int j = 1; j < conditions.n_x - 1; j += 1) {
      output[j] =
          input[j] + conditions.alpha * (dt / (dx * dx)) *
                         (input[j + 1] - 2 * input[j] + input[j - 1]);  // d^2u
    }
    std::swap(input, output);
  }
}

void parallel4_alligned_explicit(Conditions conditions, float* input,
                                 float* output) {
  float dt = conditions.t_final / (conditions.n_t - 1);
  float dx = conditions.L / (conditions.n_x - 1);

  float* tmp_in  = (float*)aligned_alloc(32, conditions.n_x * sizeof(float));
  float* tmp_out = (float*)aligned_alloc(32, conditions.n_x * sizeof(float));
  for (int i = 0; i < conditions.n_x; i++) {
    tmp_in[i]  = input[i];
    tmp_out[i] = 0;
  }

  tmp_out[0]                  = tmp_in[0];
  tmp_out[conditions.n_x - 1] = tmp_in[conditions.n_x - 1];

  for (int t = 0; t < conditions.n_t; t++) {
#pragma omp parallel num_threads(4)
    {
#pragma omp for simd aligned(tmp_in, tmp_out : 32) schedule(static)
      for (int j = 1; j < conditions.n_x - 1; j += 1) {
        tmp_out[j] = tmp_in[j] + conditions.alpha * (dt / (dx * dx)) *
                                     (tmp_in[j + 1] - 2 * tmp_in[j] +
                                      tmp_in[j - 1]);  // d^2u
      }
    }

    std::swap(tmp_in, tmp_out);
  }
  for (int i = 0; i < conditions.n_x; i++) {
    output[i] = tmp_in[i];
  }
  free(tmp_out);
  free(tmp_in);
}
void sequential_alligned_explicit(Conditions conditions, float* input,
                                  float* output) {
  float dt = conditions.t_final / (conditions.n_t - 1);
  float dx = conditions.L / (conditions.n_x - 1);

  float* tmp_in  = (float*)aligned_alloc(32, conditions.n_x * sizeof(float));
  float* tmp_out = (float*)aligned_alloc(32, conditions.n_x * sizeof(float));
  for (int i = 0; i < conditions.n_x; i++) {
    tmp_in[i]  = input[i];
    tmp_out[i] = 0;
  }

  tmp_out[0]                  = tmp_in[0];
  tmp_out[conditions.n_x - 1] = tmp_in[conditions.n_x - 1];

  for (int t = 0; t < conditions.n_t; t++) {
    {
#pragma omp for simd aligned(tmp_in, tmp_out : 32) schedule(static)
      for (int j = 1; j < conditions.n_x - 1; j += 1) {
        tmp_out[j] = tmp_in[j] + conditions.alpha * (dt / (dx * dx)) *
                                     (tmp_in[j + 1] - 2 * tmp_in[j] +
                                      tmp_in[j - 1]);  // d^2u
      }
    }

    std::swap(tmp_in, tmp_out);
  }
  for (int i = 0; i < conditions.n_x; i++) {
    output[i] = tmp_in[i];
  }
  free(tmp_out);
  free(tmp_in);
}

void sequential_implicit(Conditions conditions, float* input, float* output) {
  float dt = conditions.t_final / (conditions.n_t - 1);
  float dx = conditions.L / (conditions.n_x - 1);

  // Coefficient for the scheme
  const double r = conditions.alpha * dt / (dx * dx);
  // fmt::print("r = {}\n", r);

  std::vector<double> a(conditions.n_x, -r);             // lower diagonal
  std::vector<double> b(conditions.n_x, 1.0 + 2.0 * r);  // main diagonal
  std::vector<double> c(conditions.n_x, -r);             // upper diagonal
  std::vector<double> d(conditions.n_x);                 // right-hand side

  std::vector<double> b_work(conditions.n_x);
  std::vector<double> d_work(conditions.n_x);

  // Set boundary conditions
  a[0] = 0.0;
  c[0] = 0.0;
  b[0] = 1.0;

  a[conditions.n_x - 1] = 0.0;
  c[conditions.n_x - 1] = 0.0;
  b[conditions.n_x - 1] = 1.0;

  for (int n = 0; n < conditions.n_t; ++n) {
    for (int i = 1; i < conditions.n_x - 1; ++i) {
      d[i] = input[i];
    }

    // Apply boundary conditions
    d[0]                  = input[0];
    d[conditions.n_x - 1] = input[conditions.n_x - 1];

    // Create working copies for Thomas algorithm
    for (int i = 0; i < conditions.n_x; ++i) {
      b_work[i] = b[i];
      d_work[i] = d[i];
    }

    // Forward sweep
    // Remeber to test with simd
    for (int i = 1; i < conditions.n_x; ++i) {
      double m = a[i] / b_work[i - 1];
      b_work[i] -= m * c[i - 1];
      d_work[i] -= m * d_work[i - 1];
    }

    // Backward substitution
    output[conditions.n_x - 1] =
        d_work[conditions.n_x - 1] / b_work[conditions.n_x - 1];
    for (int i = conditions.n_x - 2; i >= 0; --i) {
      output[i] = (d_work[i] - c[i] * output[i + 1]) / b_work[i];
    }

    // Update solution for next time step
    for (int i = 0; i < conditions.n_x; ++i) {
      input[i] = output[i];
    }
  }
}

void sequential_implicit_simd(Conditions conditions, float* input,
                              float* output) {
  float dt = conditions.t_final / (conditions.n_t - 1);
  float dx = conditions.L / (conditions.n_x - 1);

  // Coefficient for the scheme
  const double r = conditions.alpha * dt / (dx * dx);
  // fmt::print("r = {}\n", r);

  std::vector<double> a(conditions.n_x, -r);             // lower diagonal
  std::vector<double> b(conditions.n_x, 1.0 + 2.0 * r);  // main diagonal
  std::vector<double> c(conditions.n_x, -r);             // upper diagonal
  std::vector<double> d(conditions.n_x);                 // right-hand side

  std::vector<double> b_work(conditions.n_x);
  std::vector<double> d_work(conditions.n_x);

  // Set boundary conditions
  a[0] = 0.0;
  c[0] = 0.0;
  b[0] = 1.0;

  a[conditions.n_x - 1] = 0.0;
  c[conditions.n_x - 1] = 0.0;
  b[conditions.n_x - 1] = 1.0;

  for (int n = 0; n < conditions.n_t; ++n) {
#pragma omp simd
    for (int i = 1; i < conditions.n_x - 1; ++i) {
      d[i] = input[i];
    }

    // Apply boundary conditions
    d[0]                  = input[0];
    d[conditions.n_x - 1] = input[conditions.n_x - 1];

    // Create working copies for Thomas algorithm

#pragma omp simd
    for (int i = 0; i < conditions.n_x; ++i) {
      b_work[i] = b[i];
      d_work[i] = d[i];
    }

    // Forward sweep
    // Remeber to test with simd
#pragma omp simd
    for (int i = 1; i < conditions.n_x; ++i) {
      double m = a[i] / b_work[i - 1];
      b_work[i] -= m * c[i - 1];
      d_work[i] -= m * d_work[i - 1];
    }

    // Backward substitution
    output[conditions.n_x - 1] =
        d_work[conditions.n_x - 1] / b_work[conditions.n_x - 1];
#pragma omp simd
    for (int i = conditions.n_x - 2; i >= 0; --i) {
      output[i] = (d_work[i] - c[i] * output[i + 1]) / b_work[i];
    }

    // Update solution for next time step
#pragma omp simd
    for (int i = 0; i < conditions.n_x; ++i) {
      input[i] = output[i];
    }
  }
}
void parallel2_implicit(Conditions conditions, float* input, float* output) {
  float dt = conditions.t_final / (conditions.n_t - 1);
  float dx = conditions.L / (conditions.n_x - 1);

  // Coefficient for the scheme
  const double r = conditions.alpha * dt / (dx * dx);
  // fmt::print("r = {}\n", r);

  std::vector<double> a(conditions.n_x, -r);             // lower diagonal
  std::vector<double> b(conditions.n_x, 1.0 + 2.0 * r);  // main diagonal
  std::vector<double> c(conditions.n_x, -r);             // upper diagonal
  std::vector<double> d(conditions.n_x);                 // right-hand side

  std::vector<double> b_work(conditions.n_x);
  std::vector<double> d_work(conditions.n_x);

  // Set boundary conditions
  a[0] = 0.0;
  c[0] = 0.0;
  b[0] = 1.0;

  a[conditions.n_x - 1] = 0.0;
  c[conditions.n_x - 1] = 0.0;
  b[conditions.n_x - 1] = 1.0;

  for (int n = 0; n < conditions.n_t; ++n) {
#pragma omp parallel for num_threads(2)
    for (int i = 1; i < conditions.n_x - 1; ++i) {
      d[i] = input[i];
    }

    // Apply boundary conditions
    d[0]                  = input[0];
    d[conditions.n_x - 1] = input[conditions.n_x - 1];

    // Create working copies for Thomas algorithm
#pragma omp parallel for num_threads(2)
    for (int i = 0; i < conditions.n_x; ++i) {
      b_work[i] = b[i];
      d_work[i] = d[i];
    }

    // Forward sweep
    for (int i = 1; i < conditions.n_x; ++i) {
      double m = a[i] / b_work[i - 1];
      b_work[i] -= m * c[i - 1];
      d_work[i] -= m * d_work[i - 1];
    }

    // Backward substitution
    output[conditions.n_x - 1] =
        d_work[conditions.n_x - 1] / b_work[conditions.n_x - 1];
    for (int i = conditions.n_x - 2; i >= 0; --i) {
      output[i] = (d_work[i] - c[i] * output[i + 1]) / b_work[i];
    }

#pragma omp parallel for num_threads(2)
    // Update solution for next time step
    for (int i = 0; i < conditions.n_x; ++i) {
      input[i] = output[i];
    }
  }
}
void parallel4_implicit(Conditions conditions, float* input, float* output) {
  float dt = conditions.t_final / (conditions.n_t - 1);
  float dx = conditions.L / (conditions.n_x - 1);

  // Coefficient for the scheme
  const double r = conditions.alpha * dt / (dx * dx);
  // fmt::print("r = {}\n", r);

  std::vector<double> a(conditions.n_x, -r);             // lower diagonal
  std::vector<double> b(conditions.n_x, 1.0 + 2.0 * r);  // main diagonal
  std::vector<double> c(conditions.n_x, -r);             // upper diagonal
  std::vector<double> d(conditions.n_x);                 // right-hand side

  std::vector<double> b_work(conditions.n_x);
  std::vector<double> d_work(conditions.n_x);

  // Set boundary conditions
  a[0] = 0.0;
  c[0] = 0.0;
  b[0] = 1.0;

  a[conditions.n_x - 1] = 0.0;
  c[conditions.n_x - 1] = 0.0;
  b[conditions.n_x - 1] = 1.0;

  for (int n = 0; n < conditions.n_t; ++n) {
#pragma omp parallel for num_threads(4)
    for (int i = 1; i < conditions.n_x - 1; ++i) {
      d[i] = input[i];
    }

    // Apply boundary conditions
    d[0]                  = input[0];
    d[conditions.n_x - 1] = input[conditions.n_x - 1];

    // Create working copies for Thomas algorithm
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < conditions.n_x; ++i) {
      b_work[i] = b[i];
      d_work[i] = d[i];
    }

    // Forward sweep
    for (int i = 1; i < conditions.n_x; ++i) {
      double m = a[i] / b_work[i - 1];
      b_work[i] -= m * c[i - 1];
      d_work[i] -= m * d_work[i - 1];
    }

    // Backward substitution
    output[conditions.n_x - 1] =
        d_work[conditions.n_x - 1] / b_work[conditions.n_x - 1];
    for (int i = conditions.n_x - 2; i >= 0; --i) {
      output[i] = (d_work[i] - c[i] * output[i + 1]) / b_work[i];
    }

#pragma omp parallel for num_threads(4)
    // Update solution for next time step
    for (int i = 0; i < conditions.n_x; ++i) {
      input[i] = output[i];
    }
  }
}
void parallel8_implicit(Conditions conditions, float* input, float* output) {
  float dt = conditions.t_final / (conditions.n_t - 1);
  float dx = conditions.L / (conditions.n_x - 1);

  // Coefficient for the scheme
  const double r = conditions.alpha * dt / (dx * dx);
  // fmt::print("r = {}\n", r);

  std::vector<double> a(conditions.n_x, -r);             // lower diagonal
  std::vector<double> b(conditions.n_x, 1.0 + 2.0 * r);  // main diagonal
  std::vector<double> c(conditions.n_x, -r);             // upper diagonal
  std::vector<double> d(conditions.n_x);                 // right-hand side

  std::vector<double> b_work(conditions.n_x);
  std::vector<double> d_work(conditions.n_x);

  // Set boundary conditions
  a[0] = 0.0;
  c[0] = 0.0;
  b[0] = 1.0;

  a[conditions.n_x - 1] = 0.0;
  c[conditions.n_x - 1] = 0.0;
  b[conditions.n_x - 1] = 1.0;

  for (int n = 0; n < conditions.n_t; ++n) {
#pragma omp parallel for num_threads(8)
    for (int i = 1; i < conditions.n_x - 1; ++i) {
      d[i] = input[i];
    }

    // Apply boundary conditions
    d[0]                  = input[0];
    d[conditions.n_x - 1] = input[conditions.n_x - 1];

    // Create working copies for Thomas algorithm
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < conditions.n_x; ++i) {
      b_work[i] = b[i];
      d_work[i] = d[i];
    }

    // Forward sweep
    for (int i = 1; i < conditions.n_x; ++i) {
      double m = a[i] / b_work[i - 1];
      b_work[i] -= m * c[i - 1];
      d_work[i] -= m * d_work[i - 1];
    }

    // Backward substitution
    output[conditions.n_x - 1] =
        d_work[conditions.n_x - 1] / b_work[conditions.n_x - 1];
    for (int i = conditions.n_x - 2; i >= 0; --i) {
      output[i] = (d_work[i] - c[i] * output[i + 1]) / b_work[i];
    }

#pragma omp parallel for num_threads(8)
    // Update solution for next time step
    for (int i = 0; i < conditions.n_x; ++i) {
      input[i] = output[i];
    }
  }
}
// not working and generally less efficient for small systems
void sequential_implicit_pcr(Conditions conditions, float* input,
                             float* output) {
  const double dx = conditions.L / (conditions.n_x - 1);
  const double dt = conditions.t_final / (conditions.n_t - 1);
  const double r  = conditions.alpha * dt / (dx * dx);

  // Main time loop - note: n_t - 1 iterations
  for (int n = 0; n < conditions.n_t - 1; ++n) {
    // Set up the tridiagonal system for this timestep
    std::vector<double> a(conditions.n_x, -r);             // lower diagonal
    std::vector<double> b(conditions.n_x, 1.0 + 2.0 * r);  // main diagonal
    std::vector<double> c(conditions.n_x, -r);             // upper diagonal
    std::vector<double> d(conditions.n_x);                 // right-hand side

    // Prepare the right-hand side vector (from current solution)
    for (int i = 0; i < conditions.n_x; ++i) {
      d[i] = input[i];
    }

    // Apply boundary conditions to the matrix
    a[0] = 0.0;
    c[0] = 0.0;
    b[0] = 1.0;
    d[0] = input[0];  // Left boundary

    a[conditions.n_x - 1] = 0.0;
    c[conditions.n_x - 1] = 0.0;
    b[conditions.n_x - 1] = 1.0;
    d[conditions.n_x - 1] = input[conditions.n_x - 1];  // Right boundary

    // Parallel Cyclic Reduction (PCR)
    int n_steps = static_cast<int>(std::ceil(std::log2(conditions.n_x)));

    for (int step = 0; step < n_steps; ++step) {
      int stride = 1 << step;  // 2^step

      // Create temporary arrays for this step
      std::vector<double> new_a(conditions.n_x);
      std::vector<double> new_b(conditions.n_x);
      std::vector<double> new_c(conditions.n_x);
      std::vector<double> new_d(conditions.n_x);

      // Copy current arrays
      for (int i = 0; i < conditions.n_x; ++i) {
        new_a[i] = a[i];
        new_b[i] = b[i];
        new_c[i] = c[i];
        new_d[i] = d[i];
      }

      // Apply PCR reduction
      for (int i = stride; i < conditions.n_x - stride; i += 2 * stride) {
        // Eliminate a[i] using equation (i - stride)
        if (std::abs(b[i - stride]) > 1e-15) {
          double factor1 = a[i] / b[i - stride];
          new_a[i]       = -factor1 * a[i - stride];
          new_b[i]       = b[i] - factor1 * c[i - stride];
          new_c[i]       = c[i];
          new_d[i]       = d[i] - factor1 * d[i - stride];
        }

        // Eliminate c[i] using equation (i + stride)
        if (std::abs(b[i + stride]) > 1e-15) {
          double factor2 = c[i] / b[i + stride];
          new_a[i]       = new_a[i];
          new_b[i]       = new_b[i] - factor2 * a[i + stride];
          new_c[i]       = -factor2 * c[i + stride];
          new_d[i]       = new_d[i] - factor2 * d[i + stride];
        }
      }

      // Update arrays
      a = new_a;
      b = new_b;
      c = new_c;
      d = new_d;
    }

    // Solve the reduced system (should be nearly diagonal now)
    for (int i = 0; i < conditions.n_x; ++i) {
      if (std::abs(b[i]) > 1e-15) {
        output[i] = d[i] / b[i];
      } else {
        output[i] = 0.0;  // Fallback for numerical issues
      }
    }

    // Copy solution back to input for next iteration
    for (int i = 0; i < conditions.n_x; ++i) {
      input[i] = output[i];
    }
  }
}
