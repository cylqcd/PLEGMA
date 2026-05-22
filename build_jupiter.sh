#!/usr/bin/env bash
# Build script for PLEGMA_wilson_flow on Jupiter (login node, no SBATCH)
#
# QUDA version: 9d98797464cbf1aed55d2daeb17677a656064ea4
#   branch: develop
#   repo:   $WORK/opt/quda_stack_latest/src/quda
#
# Usage:
#   bash build_jupiter.sh           # full reconfigure + build
#   bash build_jupiter.sh --build-only  # skip cmake configure, only ninja
#
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------- modules ---------------------------------------------------------
module --force purge
module load Stages/2026
module load GCC/14.3.0
module load CUDA/13
module load CMake/4.0.3
module load Ninja/1.13.0
module load OpenBLAS/0.3.30
module load OpenMPI/5.0.8
module load HDF5/1.14.6

# ---------- paths -----------------------------------------------------------
ROOT="$WORK/opt/quda_stack_latest"

# This script lives inside the source tree; derive SRC from its own location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLEGMA_WF_SRC="$SCRIPT_DIR"
PLEGMA_WF_BUILD="$ROOT/build/plegma_wilson_flow2"
QUDA_SRC="$ROOT/src/quda"

HDF5_DIR=/e/software/default/stages/2026/software/HDF5/1.14.6-gompi-2025b/lib/cmake/hdf5
OPENMPI_INCLUDE=/e/software/default/stages/2026/software/OpenMPI/5.0.8-GCC-14.3.0/include

# ---------- configure -------------------------------------------------------
BUILD_ONLY=false
[[ "${1:-}" == "--build-only" ]] && BUILD_ONLY=true

if ! $BUILD_ONLY; then
  rm -rf "$PLEGMA_WF_BUILD"
  mkdir -p "$PLEGMA_WF_BUILD"

  cmake -S "$PLEGMA_WF_SRC" -B "$PLEGMA_WF_BUILD" -G Ninja \
    -DGPU_ARCH=sm_90 \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DOpenMP_CXX_FLAGS="-fopenmp" \
    -DOpenMP_C_FLAGS="-fopenmp" \
    -DCMAKE_CUDA_FLAGS="-Xcompiler=-fopenmp" \
    -DMPI_HOME_MANUAL="$(dirname "$(dirname "$(which mpicxx)")")" \
    -DQUDA_HOME="$ROOT/build/quda" \
    -DQUDA_SRC="$QUDA_SRC" \
    -DHDF5_DIR="$HDF5_DIR" \
    -DHDF5_PREFER_PARALLEL=TRUE \
    -DPLEGMA_LIMEHOME="$ROOT/build/lime" \
    -DPLEGMA_OPENBLAS=ON \
    -DPLEGMA_OPENBLASHOME="/e/software/default/stages/2026/software/OpenBLAS/0.3.30-GCC-14.3.0" \
    -DPLEGMA_OPENBLAS_LIB="/e/software/default/stages/2026/software/OpenBLAS/0.3.30-GCC-14.3.0/lib/libopenblas.so" \
    -DPLEGMA_OPENBLAS_GOMP="/e/software/default/stages/2026/software/GCCcore/14.3.0/lib64/libgomp.so" \
    -DCUDA_nvToolsExt_LIBRARY=/e/software/default/stages/2026/software/CUDA/13/targets/sbsa-linux/lib/libnvtx3interop.so \
    -DPLEGMA_HIGHER_DERIV=ON \
    -DPLEGMA_SCATTERING_CONTRACTIONS=OFF \
    -DPLEGMA_NUCLEON_3PF_FIX_SINK=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

  # Workaround: CMake on Jupiter injects bogus MPI include paths
  # (-isystem /include -isystem /src) into CUDA compile rules in build.ninja.
  sed -i "s|-isystem /include -isystem /src|-isystem ${OPENMPI_INCLUDE}|g" \
      "$PLEGMA_WF_BUILD/build.ninja"
fi

# ---------- build -----------------------------------------------------------
cmake --build "$PLEGMA_WF_BUILD" -j"$(nproc)"

echo "==== PLEGMA_wilson_flow build completed ===="
echo "Build directory: $PLEGMA_WF_BUILD"
