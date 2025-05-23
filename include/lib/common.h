#include <string>
#include <utility>
#include <vector>
struct Conditions {
  const double L;        // Length of the rod
  const double alpha;    // Thermal diffusivity
  const double t_final;  // Final time
  const int n_x;         // Number of spatial points
  const int n_t;         // Number of time steps
};
void initialize_array(float* array, int size);

// const float expected[16]{100,       83.93145,  68.75215,  55.306667,
//                          44.362194, 36.59398,  32.58662,  32.84042,
//                          37.768017, 47.669025, 62.678932, 82.70068,
//                          107.33814, 135.85602, 167.18753, 200};
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
    {{10, .05, 1.0, 100, 200}, "High_Diffusivity"},
    {{.1, .05, 1.0, 100, 500}, "Low_Diffusivity"},
    {{1, .5, 10.0, 100, 200}, "Large_Domain"},
    {{1, .1, 1.0, 50, 2000}, "Fine_Time_Stepping"},
};
