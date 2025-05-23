#include <lib/common.h>
void sequential_explicit(Conditions, float*, float*);
void parallel2_inner_explicit(Conditions, float*, float*);
void parallel4_inner_explicit(Conditions, float*, float*);
void parallel8_inner_explicit(Conditions, float*, float*);

void parallel2_outer_explicit(Conditions, float*, float*);

void sequential_unroll_explicit(Conditions, float*, float*);

void parallel2_collapse_explicit(Conditions, float*, float*);

void parallel4_alligned_explicit(Conditions, float*, float*);

void sequential_implicit(Conditions, float*, float*);
void sequential_implicit_pcr(Conditions, float*, float*);
