export BENCHMARK_CONFIDENCE_INTERVAL=0.30
export BENCH_SAMPLES=2
export CLEAR_RESULTS=0

export PROJECT_ROOT=$(pwd)

# array of optiimization levels to try
optimization_levels=(O0 O2 O3)

######################################################

mkdir results >/dev/null

if [ ${CLEAR_RESULTS} == 1 ]; then
	echo "Clearing previous results"
	rm -rf results/*
else
	echo "Keeping previous results"
fi

# Print information about the cpu
lscpu >results/cpu_info.txt

# Creta a build directory if it does not exist
mkdir build >/dev/null
cd build

for OPTIMIZATION_LEVEL in "${optimization_levels[@]}"; do
	echo "Building with optimization level: ${OPTIMIZATION_LEVEL}"
	mkdir -p ${PROJECT_ROOT}/results/${OPTIMIZATION_LEVEL}
	export OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL}

	cmake -DOPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL} -DENABLE_OPTIMIZATION=ON ..

	make -j4
	sh $PROJECT_ROOT/scripts/run_benchmarks.sh
done
