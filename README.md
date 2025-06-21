# Fundamentals of Parallel Programming - Final Project

## Parallelized Solution of 1D Heat Equation
## Index

<!--toc:start-->
- [Fundamentals of Parallel Programming - Final Project](#fundamentals-of-parallel-programming-final-project)
  - [Parallelized Solution of 1D Heat Equation](#parallelized-solution-of-1d-heat-equation)
  - [Index](#index)
  - [Project structure](#project-structure)
  - [Getting started](#getting-started)
    - [Cloning repository](#cloning-repository)
    - [Building](#building)
      - [Optimization (to be fixed)](#optimization-to-be-fixed)
      - [Manual building](#manual-building)
      - [Automatic building and test running](#automatic-building-and-test-running)
<!--toc:end-->

## Project structure

- `apps`
    - Source code for test executables and the MPI solution
- `include/lib`
    - Header files of the libraries
- `scripts`
    - Bash and python scripts to build, run benchmarks and visualize results
- `src`
    - Source code of the libraries
- `tests`
    - Source code for tests and benchmarks
- `third-party`
    - Git submodules of dependencies

## Getting started

### Cloning repository

```sh
git clone --recursive https://github.com/thegavereguy/parallel_programming_final
cd parallel_programming_final
```

### Building 

#### Optimization (to be fixed)

When building manually, compiler optimization can be toggled using:
```
export ENABLE_OPTIMIZATION=ON|OFF
```

The optimization level can be set with:
```
export OPTIMIZATION_LEVEL=O0|O1|O2|O3|Ofast
```

Keen in mind that these setting are used inside the build script (`assignment.sh`). When building manually, these should be passed as arguments when calling `cmake`.

#### Manual building

When not running the `assignment.sh` script or submitting the `assignment.pbs` job, the project can be manually built and run:

```sh
mkdir build
cd build
cmake .. # add settings if required, ex: -DENABLE_OPTIMIZATION=ON -DOPTIMIZATION_LEVEL=O2
make 
```

When building on the cluster, make sure to load the `cmake-3.15.4` and `openmpi-4.0.4` modules. 

#### Automatic building and test running

The build process, as well as the data collection process, can be executed by running `./scripts/assingment.sh`.
The various build settings can be changes modifying the exported variable at the top of the file. Previously exported variables are overwritten.

The results are saved into a `results` folder.

!!! When run with 10 benchmark samples and all the test cases, the execution takes around 30-40 mins !!!
