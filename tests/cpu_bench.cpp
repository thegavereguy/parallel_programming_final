#include <fmt/base.h>
#include <lib/shared.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_chronometer.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>

TEST_CASE("sequential - explicit - BENCH", "[omp_1_ex]") {
  char* name = new char[100];

  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);
    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return sequential_explicit(conditions.first, input, output);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("sequential - implicit - BENCH", "[omp_1_im]") {
  char* name = new char[100];

  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);
    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return sequential_implicit(conditions.first, input, output);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("sequential alligned - explicit - BENCH", "[omp_1_ex_all]") {
  char* name = new char[100];

  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);
    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return sequential_explicit_alligned(conditions.first, input, output);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("sequential SIMD - explicit - BENCH", "[omp_1_ex_simd]") {
  char* name = new char[100];

  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);
    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return sequential_explicit_simd(conditions.first, input, output);
      });

      delete[] input;
      delete[] output;
    };
  }
}

TEST_CASE("sequential SIMD - implicit - BENCH", "[omp_1_im_simd]") {
  char* name = new char[100];

  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);
    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return sequential_implicit_simd(conditions.first, input, output);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("sequential PCR - implicit - BENCH", "[omp_1_im_pcr]") {
  char* name = new char[100];

  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);
    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return sequential_implicit_pcr(conditions.first, input, output);
      });

      delete[] input;
      delete[] output;
    };
  }
}

TEST_CASE("parallel - 2 threads - explicit - BENCH", "[omp_2_ex]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return parallel_variable_explicit(conditions.first, input, output, 2);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("parallel - 2 threads - implicit - BENCH", "[omp_2_im]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return parallel_variable_implicit(conditions.first, input, output, 2);
      });

      delete[] input;
      delete[] output;
    };
  }
}

TEST_CASE("parallel - 4 threads - explicit - BENCH", "[omp_4_ex]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return parallel_variable_explicit(conditions.first, input, output, 4);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("parallel - 4 threads - implicit - BENCH", "[omp_4_im]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return parallel_variable_implicit(conditions.first, input, output, 4);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("parallel - 8 threads - explicit - BENCH", "[omp_8_ex]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return parallel_variable_explicit(conditions.first, input, output, 8);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("parallel - 8 threads - implicit - BENCH", "[omp_8_im]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return parallel_variable_implicit(conditions.first, input, output, 8);
      });

      delete[] input;
      delete[] output;
    };
  }
}

TEST_CASE("parallel - 16 threads - explicit - BENCH", "[omp_16_ex]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return parallel_variable_explicit(conditions.first, input, output, 16);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("parallel - 16 threads - implicit - BENCH", "[omp_16_im]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return parallel_variable_implicit(conditions.first, input, output, 16);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("parallel - 32 threads - explicit - BENCH", "[omp_32_ex]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return parallel_variable_explicit(conditions.first, input, output, 32);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("parallel - 32 threads - implicit - BENCH", "[omp_32_im]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return parallel_variable_implicit(conditions.first, input, output, 32);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("parallel alligned - 4 threads - explicit - BENCH",
          "[omp_4_ex_all]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return parallel4_alligned_explicit(conditions.first, input, output);
      });

      delete[] input;
      delete[] output;
    };
  }
}
TEST_CASE("sequential unroll - explicit - BENCH", "[omp_1_ex_unr]") {
  char* name = new char[100];
  for (auto conditions : target_cases) {
    sprintf(name, "%s,%ld,%ld", conditions.second.data(),
            (long)conditions.first.n_x, (long)conditions.first.n_t);

    BENCHMARK_ADVANCED(name)(Catch::Benchmark::Chronometer meter) {
      float* input                    = new float[conditions.first.n_x];
      input[0]                        = 100;
      input[conditions.first.n_x - 1] = 200;
      float* output                   = new float[conditions.first.n_x];

      meter.measure([conditions, input, output] {
        return sequential_explicit_unroll(conditions.first, input, output);
      });

      delete[] input;
      delete[] output;
    };
  }
}

int main(int argc, char* argv[]) {
  int result = Catch::Session().run(argc, argv);

  return result;
}

class PartialCSVReporter : public Catch::StreamingReporterBase {
 public:
  using StreamingReporterBase::StreamingReporterBase;

  static std::string getDescription() {
    return "Reporter for benchmarks in CSV format";
  }

  void testCasePartialStarting(Catch::TestCaseInfo const& testInfo,
                               uint64_t partNumber) override {
    // std::cout << "TestCase: " << testInfo.name << '#' << partNumber << '\n';
    // std::cout << "DIMENSION,MEAN,MINT,MAXT,ITER" << '\n';
    fmt::print("NAME,NX,NT,MEAN,MINT,MAXT,ITER\n");
  }

  void testCasePartialEnded(Catch::TestCaseStats const& testCaseStats,
                            uint64_t partNumber) override {
    // std::cout << "TestCaseEnded: " << testCaseStats.testInfo->name << '#' <<
    // partNumber << '\n';
  }

  void benchmarkEnded(Catch::BenchmarkStats<> const& stats) override {
    // std::cout << stats.info.name << "," << stats.mean.point.count() / 1e6 <<
    // ","
    //           << stats.mean.lower_bound.count() / 1e6 << ","
    //           << stats.mean.upper_bound.count() / 1e6 << ","
    //           << stats.info.iterations << '\n';
    // print the first 4 decimal places
    fmt::print("{},{:.4f},{:.4f},{:.4f},{}\n", stats.info.name,
               stats.mean.point.count() / 1e6,
               stats.mean.lower_bound.count() / 1e6,
               stats.mean.upper_bound.count() / 1e6, stats.info.iterations);
  }
};
CATCH_REGISTER_REPORTER("csv", PartialCSVReporter)
