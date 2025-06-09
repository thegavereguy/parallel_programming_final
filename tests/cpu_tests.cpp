#include <fmt/base.h>
#include <lib/shared.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_chronometer.hpp>
#include <catch2/benchmark/catch_clock.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>

TEST_CASE("Sequential solution explicit - Test", "[seq_ex]") {
  Conditions conditions = {1, 0.5, 0.01, 16, 30};
  float* input          = new float[conditions.n_x];
  float* output         = new float[conditions.n_x];
  initialize_array(input, conditions.n_x);
  initialize_array(output, conditions.n_x);

  input[0]                  = 100;
  input[conditions.n_x - 1] = 200;

  sequential_explicit(conditions, input, output);
  for (int i = 0; i < conditions.n_x; i++) {
    REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected[i], 0.001));
  }
  delete[] input;
  delete[] output;
}
TEST_CASE("Sequential solution implicit - Test", "[seq_im]") {
  Conditions conditions = {1, 0.5, 0.01, 16, 30};
  float* input          = new float[conditions.n_x];
  float* output         = new float[conditions.n_x];
  initialize_array(input, conditions.n_x);
  initialize_array(output, conditions.n_x);

  input[0]                  = 100;
  input[conditions.n_x - 1] = 200;

  sequential_implicit(conditions, input, output);
  for (int i = 0; i < conditions.n_x; i++) {
    // fmt::print("{} - {}\n", output[i], expected[i]);
    REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected[i], 0.001));
  }
  delete[] input;
  delete[] output;
}
TEST_CASE("Sequential solution implicit PCR- Test", "[seq_im_pcr]") {
  Conditions conditions = {1, 0.5, 0.01, 16, 30};
  float* input          = new float[conditions.n_x];
  float* output         = new float[conditions.n_x];
  initialize_array(input, conditions.n_x);
  initialize_array(output, conditions.n_x);

  input[0]                  = 100;
  input[conditions.n_x - 1] = 200;

  sequential_implicit_pcr(conditions, input, output);
  for (int i = 0; i < conditions.n_x; i++) {
    fmt::print("{} - {}\n", output[i], expected[i]);
    REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected[i], 0.1));
  }
  delete[] input;
  delete[] output;
}
TEST_CASE("Parallel solution - 2 threads - explicit - Test", "[par2_ex]") {
  Conditions conditions = {1, 0.5, 0.01, 16, 30};
  float* input          = new float[conditions.n_x];
  float* output         = new float[conditions.n_x];
  initialize_array(input, conditions.n_x);
  initialize_array(output, conditions.n_x);

  input[0]                  = 100;
  input[conditions.n_x - 1] = 200;

  for (int i = 0; i < conditions.n_x; i++) {
  }
  parallel_variable_explicit(conditions, input, output, 2);
  for (int i = 0; i < conditions.n_x; i++) {
    REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected[i], 0.001));
  }
  delete[] input;
  delete[] output;
}
TEST_CASE("Parallel solution - 4 threads - explicit - Test", "[par4_ex]") {
  Conditions conditions = {1, 0.5, 0.01, 16, 30};
  float* input          = new float[conditions.n_x];
  float* output         = new float[conditions.n_x];
  initialize_array(input, conditions.n_x);
  initialize_array(output, conditions.n_x);
  input[0]                  = 100;
  input[conditions.n_x - 1] = 200;

  // fmt::print("par4 solution\n");
  parallel_variable_explicit(conditions, input, output, 4);
  for (int i = 0; i < conditions.n_x; i++) {
    REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected[i], 0.001));
  }

  delete[] input;
  delete[] output;
}

TEST_CASE("Parallel solution - 8 threads - explicit - Test", "[par8_ex]") {
  Conditions conditions = {1, 0.5, 0.01, 16, 30};
  float* input          = new float[conditions.n_x];
  float* output         = new float[conditions.n_x];
  initialize_array(input, conditions.n_x);
  initialize_array(output, conditions.n_x);
  input[0]                  = 100;
  input[conditions.n_x - 1] = 200;

  parallel_variable_explicit(conditions, input, output, 8);
  for (int i = 0; i < conditions.n_x; i++) {
    REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected[i], 0.001));
  }
  delete[] input;
  delete[] output;
}

// TEST_CASE("Parallel 4 alligned solution - Test", "[par4_all_ex]") {
//   Conditions conditions = {1, 0.5, 0.01, 16, 30};
//   float* input          = new float[conditions.n_x];
//   float* output         = new float[conditions.n_x];
//   initialize_array(input, conditions.n_x);
//   initialize_array(output, conditions.n_x);
//   input[0]                  = 100;
//   input[conditions.n_x - 1] = 200;
//   parallel4_alligned_explicit(conditions, input, output);
//   for (int i = 0; i < conditions.n_x; i++) {
//     REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected[i], 0.001));
//   }
//   delete[] input;
//   delete[] output;
// }

TEST_CASE("Sequential solution - loop unroll - explicit - Test",
          "[seq_unr_ex]") {
  Conditions conditions = {1, 0.5, 0.01, 16, 30};
  float* input          = new float[conditions.n_x];
  float* output         = new float[conditions.n_x];
  initialize_array(input, conditions.n_x);
  initialize_array(output, conditions.n_x);
  input[0]                  = 100;
  input[conditions.n_x - 1] = 200;
  sequential_unroll_explicit(conditions, input, output);
  for (int i = 0; i < conditions.n_x; i++) {
    // fmt::print("{} ", output[i]);
    REQUIRE_THAT(output[i], Catch::Matchers::WithinAbs(expected[i], 0.001));
  }
  delete[] input;
  delete[] output;
}
