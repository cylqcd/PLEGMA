#!/bin/bash

module purge
module load HDF5/1.10.5-gompic-2019b Autotools/20180311-GCCcore-8.3.0 CMake/3.15.3-GCCcore-8.3.0 CUDA/10.1.243-GCC-8.3.0 OpenBLAS/0.3.7-GCC-8.3.0 OpenMPI/3.1.4-GCC-8.3.0 Eigen/3.3.7
module list

rm -rf build
mkdir build
cd build

cmake -DGPU_ARCH=sm_70 -DPLEGMA_NUCLEON_3PF_FIX_SINK=ON -DQUDA_HOME=$HOME/src/quda/build -DQUDA_SRC=$HOME/src/quda -DPLEGMA_OPENBLAS=ON -DPLEGMA_LIMEHOME=$HOME/src/c-lime/ -DPLEGMA_LIME_LIB=$HOME/src/c-lime/lib/liblime.a ../

make -j 
