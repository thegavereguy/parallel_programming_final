export BENCHMARK_CONFIDENCE_INTERVAL=0.30
export BENCH_SAMPLES=2
export BUILD_VULKAN=true
export BUILD_MPI=false
export BUILD_ACC=true
export DOWNLOAD_VULKAN_SDK=false
export CLEAR_RESULTS=0
export BUILD_OPTIMIZED=true

export PROJECT_ROOT=$(pwd)

######################################################

mkdir results > /dev/null

if [ ${CLEAR_RESULTS} == 1 ]; then
	echo "Clearing previous results";
	rm -rf results/*
else
  echo "Keeping previous results";
fi

# Print information about the cpu and gpu (which is somewhere in lspci)
lscpu > results/cpu_info.txt
lspci > results/gpu_info.txt

# Creta a build directory if it does not exist
mkdir build > /dev/null
cd build

cmake ..  -DBUILD_VULKAN=${BUILD_VULKAN} -DBUILD_MPI=${BUILD_MPI} -DBUILD_ACC=${BUILD_ACC} -DDOWNLOAD_VULKAN_SDK=${DOWNLOAD_VULKAN_SDK} -DBUILD_OPTIMIZED=${BUILD_OPTIMIZED}

# Remove the previous build
# make clean

make all -j4
cd ..

sh $PROJECT_ROOT/scripts/run_benchmarks.sh
