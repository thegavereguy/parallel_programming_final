#include <lib/common.h>
void sequential_explicit(Conditions, float*, float*);

void sequential_explicit_unroll(Conditions, float*, float*);
void sequential_explicit_alligned(Conditions, float*, float*);

void parallel4_alligned_explicit(Conditions, float*, float*);

void sequential_implicit(Conditions, float*, float*);
void sequential_implicit_pcr(Conditions, float*, float*);
void sequential_implicit_simd(Conditions, float*, float*);

void parallel_variable_explicit(Conditions, float*, float*, int);
void parallel_variable_implicit(Conditions, float*, float*, int);
