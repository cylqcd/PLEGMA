#!/bin/bash

module purge
module load HDF5/1.10.5-gompic-2019b Autotools/20180311-GCCcore-8.3.0 CMake/3.15.3-GCCcore-8.3.0 CUDA/10.1.243-GCC-8.3.0 OpenBLAS/0.3.7-GCC-8.3.0 OpenMPI/3.1.4-gcccuda-2019b Python/3.7.4-GCCcore-8.3.0
module list
#which cmake
# Somehow, CC & CXX env variables are set, probably by anaconda.  These are not default bash env var but used by cmake.  I need to set them to appropriate values
# !! I have deactivated conda base environment !!
#export CC=/onyx/buildsets/eb_cyclamen/software/GCCcore/8.3.0/bin/cc
#export CXX=/onyx/buildsets/eb_cyclamen/software/GCCcore/8.3.0/bin/c++
#export /usr/lib64/libdl.so
export PLEGMA_HDF5HOME=/onyx/buildsets/eb_cyclamen/software/HDF5/1.10.5-gompic-2019b
echo $PATH
rm -rf build
mkdir build
cd build
cmake -DPLEGMA_NUCLEON_3PF_FIX_SINK=ON -DQUDA_HOME=$HOME/src/quda/build -DQUDA_SRC=$HOME/src/quda -DPLEGMA_OPENBLAS=ON -DPLEGMA_LIMEHOME=$HOME/src/c-lime/ -DPLEGMA_LIME_LIB=$HOME/src/c-lime/lib/liblime.a ../
make -j 
