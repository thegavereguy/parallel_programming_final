#include <string>
#include <utility>
#include <vector>

struct Conditions {
  const double L;
  const double alpha;
  const double t_final;
  const int n_x;
  const int n_t;
};
void initialize_array(float* array, int size);

const float expected[16] = {100.0000,  50.5701,   19.2164,   5.61374,
                            1.29573,   0.242096,  0.0374409, 0.00589091,
                            0.0101906, 0.0747319, 0.484179,  2.59147,
                            11.2275,   38.4327,   101.14,    200.0000};

const std::vector<std::pair<Conditions, std::string>> target_cases = {
    {{1, .1, 1.0, 50, 100}, "Stability_Test_Large_dt"},
    {{1, .1, 1.0, 50, 50}, "Stability_Test_Very_Large_dt"},
    {{1, .05, 1.0, 100, 200}, "Standard_Diffusion"},
    {{1, .05, 1.0, 200, 400}, "Fine_Spatial_Resolution"},
    {{1, 1, 1.0, 100, 1000}, "Long_Time_Simulation"},
    {{1, .1, 1.0, 50, 2000}, "Fine_Time_Stepping"},
    {{1, .1, 1.0, 5000, 2000}, "Very_High_Spatial_Resolution"},
    {{1, .1, 1.0, 500, 20000}, "Very_High_Time_Resolution"},
    {{1, .05, 1.0, 10000, 400}, "Massive_Spatial_Resolution_1"},
    {{1, .05, 1.0, 50000, 400}, "Massive_Spatial_Resolution_2"},
    {{1, .05, 1.0, 100000, 400}, "Massive_Spatial_Resolution_3"},
    {{1, .05, 10.0, 8192, 10000}, "Very_Large_Problem_1"},
    {{1, .05, 10.0, 16384, 20000}, "Very_Large_Problem_2"},
    {{1, .05, 10.0, 32768, 40000}, "Very_Large_Problem_3"},
    {{1, 0.5, 1.0, 100, 100}, "High_Alpha_Explicit_Stability_Limit"},
    {{1, 1.0, 1.0, 100, 100}, "Very_High_Alpha_Implicit_Advantage"},
};
