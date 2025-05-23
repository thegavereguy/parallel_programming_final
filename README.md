# Fundamentals of Parallel Programming - Final Project

## Parallelized Solution of 1D Heat Equation
## Index

<!--toc:start-->
- [Fundamentals of Parallel Programming - Final Project](#fundamentals-of-parallel-programming-final-project)
  - [Parallelized Solution of 1D Heat Equation](#parallelized-solution-of-1d-heat-equation)
  - [Index](#index)
  - [Project structure](#project-structure)
  - [Important notes](#important-notes)
  - [Getting started](#getting-started)
    - [Dependencies](#dependencies)
    - [Cloning repository](#cloning-repository)
    - [Building](#building)
      - [Parts](#parts)
      - [Optimization](#optimization)
      - [Manual building](#manual-building)
      - [Automatic building and test running](#automatic-building-and-test-running)
<!--toc:end-->

## Project structure

- `apps`
    - Source code for test executables and the MPI solution
- `framework`
    - Source code for the Vulkan framework
- `include/lib`
    - Header files of the libraries
- `scripts`
    - Bash and python scripts to build, run benchmarks and visualize results
- `shaders`
    - Source code for GLSL shaders
- `src`
    - Source code of the libraries
- `tests`
    - Source code for tests and benchmarks
- `third-party`
    - Git submodules of dependencies

## Getting started

### Cloning repository

```sh
git clone --recursive https://github.com/thgavereguy/parallel_programming_final
cd parallel_programming_final
```

### Building 

#### Parts
- OpenMP implementation '(required)'.
- MPI implementation (can be included by setting the ENV variable )
    - `export BUILD_MPI=true`

#### Optimization (to be fixed)

Compiler optimization can be toggled using:
```
export BUILD_OPTMIZED=true|false
```

#### Manual building

After selecting the required parts we can manually compile the project:


```sh
mkdir build
cd build
cmake .. # of /apps/cmake-3.20.3/bin/cmake on the cluster
make 
```

#### Automatic building and test running

The build process, as well as the data collection process, can be executed by running `./scripts/assingment.sh`.
The various build settings can be changes modifying the exported variable at the top of the file. Previously exported variables are overwritten.

The results are saved into a `results` folder.

