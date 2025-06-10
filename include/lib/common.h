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

const double L_PER_UNIT = 1.0;
const double ALPHA      = 0.01;
const double T_FINAL    = 0.1;
const int NX_INTERVALS  = 2 << 13;  // Numero di intervalli per unità
const int NT            = 1000;

void initialize_array(float* array, int size);

const float expected[16] = {100.0000,  50.5701,   19.2164,   5.61374,
                            1.29573,   0.242096,  0.0374409, 0.00589091,
                            0.0101906, 0.0747319, 0.484179,  2.59147,
                            11.2275,   38.4327,   101.14,    200.0000};

const std::vector<std::pair<Conditions, std::string>> target_cases = {
    {{1, .1, 1.0, 50, 100}, "Stability_Test_Large_dt"},
    {{1, .1, 1.0, 50, 50}, "Stability_Test_Very_Large_dt"},
    {{1, 1, 1.0, 100, 1000}, "Long_Time_Simulation"},
    {{1, .1, 1.0, 50, 2000}, "Fine_Time_Stepping"},
    {{1, .1, 1.0, 5000, 2000}, "Very_High_Spatial_Resolution"},
    {{1, .1, 1.0, 500, 20000}, "Very_High_Time_Resolution"},
    {{1, .05, 1.0, 10000, 400}, "Massive_Spatial_Resolution_1"},
    {{1, .05, 1.0, 50000, 400}, "Massive_Spatial_Resolution_2"},
    {{1, .05, 1.0, 100000, 400}, "Massive_Spatial_Resolution_3"},
    {{1, .05, 10.0, 8192, 10000}, "Very_Large_Problem_1"},
    {{L_PER_UNIT * 1, ALPHA, T_FINAL, (NX_INTERVALS * 1) + 1, NT},
     "WeakScaling_P1"},
    {{L_PER_UNIT * 2, ALPHA, T_FINAL, (NX_INTERVALS * 2) + 1, NT},
     "WeakScaling_P2"},
    {{L_PER_UNIT * 4, ALPHA, T_FINAL, (NX_INTERVALS * 4) + 1, NT},
     "WeakScaling_P4"},
    {{L_PER_UNIT * 8, ALPHA, T_FINAL, (NX_INTERVALS * 8) + 1, NT},
     "WeakScaling_P8"},
    {{L_PER_UNIT * 16, ALPHA, T_FINAL, (NX_INTERVALS * 16) + 1, NT},
     "WeakScaling_P16"},
    {{L_PER_UNIT * 32, ALPHA, T_FINAL, (NX_INTERVALS * 32) + 1, NT},
     "WeakScaling_P32"},

    // temporarily disabled due to long execution time
    // {{1, .05, 10.0, 16384, 20000}, "Very_Large_Problem_2"},
    // {{1, .05, 10.0, 32768, 40000}, "Very_Large_Problem_3"},
};

const std::vector<std::pair<Conditions, std::string>> weak_scaling_cases = {
    {{L_PER_UNIT * 1, ALPHA, T_FINAL, (NX_INTERVALS * 1) + 1, NT},
     "WeakScaling_P1_Nx" + std::to_string((NX_INTERVALS * 1) + 1) + "_L" +
         std::to_string(L_PER_UNIT * 1)},
    {{L_PER_UNIT * 2, ALPHA, T_FINAL, (NX_INTERVALS * 2) + 1, NT},
     "WeakScaling_P2_Nx" + std::to_string((NX_INTERVALS * 2) + 1) + "_L" +
         std::to_string(L_PER_UNIT * 2)},
    {{L_PER_UNIT * 4, ALPHA, T_FINAL, (NX_INTERVALS * 4) + 1, NT},
     "WeakScaling_P4_Nx" + std::to_string((NX_INTERVALS * 4) + 1) + "_L" +
         std::to_string(L_PER_UNIT * 4)},
    {{L_PER_UNIT * 8, ALPHA, T_FINAL, (NX_INTERVALS * 8) + 1, NT},
     "WeakScaling_P8_Nx" + std::to_string((NX_INTERVALS * 8) + 1) + "_L" +
         std::to_string(L_PER_UNIT * 8)},
    {{L_PER_UNIT * 16, ALPHA, T_FINAL, (NX_INTERVALS * 16) + 1, NT},
     "WeakScaling_P16_Nx" + std::to_string((NX_INTERVALS * 16) + 1) + "_L" +
         std::to_string(L_PER_UNIT * 16)},
    // {
    //     {L_PER_UNIT_WS * 32,
    //      ALPHA_CONST_WS,
    //      T_FINAL_CONST_WS,
    //      (NX_INTERVALS_PER_UNIT_WS * 32) + 1,
    //      NT_CONST_WS},
    //     "WeakScaling_P32_Nx" + std::to_string((NX_INTERVALS_PER_UNIT_WS * 32)
    //     + 1) + "_L" + std::to_string(L_PER_UNIT_WS * 32)
    // }
};
