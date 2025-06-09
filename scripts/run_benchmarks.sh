#!/bin/bash

omp_tests=("seq_ex" "seq_im")

if [ -z "$BENCHMARK_SAMPLES" ]; then
	export BENCHMARK_SAMPLES=5
fi

if [ -z "$BENCHMARK_CONFIDENCE_INTERVAL" ]; then
	export BENCHMARK_CONFIDENCE_INTERVAL=0.30
fi

echo "Running $BENCHMARK_SAMPLES samples per benchmark with $BENCHMARK_CONFIDENCE_INTERVAL confidence interval"
# go to folder /build/tests relative to the project root
if [ -z "$PROJECT_ROOT" ]; then
	PROJECT_ROOT=$(git rev-parse --show-toplevel)
fi

cd $PROJECT_ROOT/build/tests

for i in ${omp_tests[@]}; do
	echo "Running benchmark for [$i]"
	eval "./cpu_bench" \"[$i]\" "-r csv" "--benchmark-samples=$BENCHMARK_SAMPLES" "--benchmark-confidence-interval=$BENCHMARK_CONFIDENCE_INTERVAL" >"$PROJECT_ROOT/results/$i.csv"
done

echo "Running MPI benchmarks"

n=(1 2 4)

for iter in "${n[@]}"; do
	echo "Running MPI benchmark for $iter processes"
	mpirun -np $iter ../apps/mpi >>"$PROJECT_ROOT/results/mpi_$iter.csv"
	# valutare se si riesce a spostare l'eseguibile in /build/tests
done
