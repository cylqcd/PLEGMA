#!/bin/bash
source ~/LoadModule.sh
module load CMake
rm -rf build
mkdir build
cd build
cmake -DPLEGMA_NUCLEON_3PF_FIX_SINK=ON -DQUDA_HOME=$HOME/src/quda/build -DQUDA_SRC=$HOME/src/quda -DPLEGMA_OPENBLAS=ON -DPLEGMA_LIMEHOME=$HOME/src/c-lime/ -DPLEGMA_LIME_LIB=$HOME/src/c-lime/lib/liblime.a ../
#/cyclamen/home/dnole/src/thiscmake/cmake-3.12.3/bin/cmake -DPLEGMA_NUCLEON_3PF_FIX_SINK=ON -DQUDA_HOME=$HOME/src/quda/build -DQUDA_SRC=$HOME/src/quda -DPLEGMA_OPENBLAS=ON -DPLEGMA_LIMEHOME=$HOME/src/c-lime/ -DPLEGMA_LIME_LIB=$HOME/src/c-lime/lib/liblime.a ../
make -j 
