#pragma once

// Optimized contractG5 / oneEnd_trick replacement.
//
// The stock PLEGMA path (PLEGMA_QLoops::oneEnd_trick) does:
//   1. contractG5(x_l, x_r)              [GPU kernel, ~0.5 ms]
//   2. unload()                          [D2H of all 16 gamma channels]
//   3. cBLAS::axpy / scal on host buffer [host BLAS over ~80 MB]
//
// On a V100 with 24^3 x 48 the per-call cost is dominated by (2)+(3)
// (~20 ms / 0.5 ms = ~40x slowdown over the kernel itself).
//
// This wrapper keeps the accumulator on the device.  Each call only
// runs (1) plus a fused scale-axpy kernel (a few hundred microseconds
// of memory-bound work).  A single D2H copy is performed at flush time.
//
// Usage:
//   PLEGMA_QLoops_fast<float> qLoops(BOTH);
//   qLoops.clearAccumDevice();
//   for (...) qLoops.oneEnd_trick_fast(x_l, x_r, val, accum);
//   qLoops.flushAccumToHost();   // populates H_loc() once
//
// After flushAccumToHost(), the existing PLEGMA dump pipeline
// (load() + ft->apply() + writeFile()) keeps working unchanged.

#include "PLEGMA_QLoops_patterns.h"
#include <cuda_runtime.h>
#include <quda_api.h>
#include <cstddef>
#include <vector>

namespace bhu {

// Free functions implemented in lib/kernels/contractG5_fast.cu.
// `n` is the number of real Float scalars (= field_length * total_length * 2).
template<typename Float>
void launch_axpy_real(Float* acc, const Float* src, Float val, std::size_t n);

template<typename Float>
void launch_scal_real(Float* dst, const Float* src, Float val, std::size_t n);

}  // namespace bhu


template<typename Float>
class PLEGMA_QLoops_fast : public PLEGMA_QLoops_patterns<Float> {
  using Base = PLEGMA_QLoops_patterns<Float>;

  Float* d_acc = nullptr;        // device-side local accumulator
  std::size_t n_real = 0;        // number of real Float scalars in d_acc
  std::size_t bytes_total = 0;

public:
  // Inherit all constructors from PLEGMA_QLoops_patterns.
  using Base::Base;

  void allocAccumDevice() {
    if (d_acc) return;
    n_real      = static_cast<std::size_t>(this->Field_length())
                * static_cast<std::size_t>(this->Total_length())
                * 2u;
    bytes_total = this->Bytes_total();
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&d_acc), bytes_total);
    if (e != cudaSuccess) {
      PLEGMA_error("PLEGMA_QLoops_fast: cudaMalloc(d_acc, %zu) failed: %s",
                   bytes_total, cudaGetErrorString(e));
    }
    cudaMemset(d_acc, 0, bytes_total);
  }

  void freeAccumDevice() {
    if (d_acc) {
      cudaFree(d_acc);
      d_acc = nullptr;
    }
  }

  ~PLEGMA_QLoops_fast() { freeAccumDevice(); }

  // Reset the device accumulator to zero.  Cheap; just one cudaMemsetAsync.
  void clearAccumDevice() {
    if (!d_acc) allocAccumDevice();
    cudaMemset(d_acc, 0, bytes_total);
  }

  // Optimized replacement for oneEnd_trick(x_l, x_r, val, accum).
  // Runs contractG5 on the device, then a fused scale-axpy kernel that
  // updates the device accumulator.  No PCIe traffic, no host BLAS.
  void oneEnd_trick_fast(PLEGMA_Vector<Float>& x_l,
                         PLEGMA_Vector<Float>& x_r,
                         Float val,
                         bool accum) {
    if (this->IsOneD() || this->IsTwoD())
      PLEGMA_error("oneEnd_trick_fast does not handle oneD / twoD; "
                   "use the stock PLEGMA path for derivatives.");
    if (!d_acc) allocAccumDevice();

    // Fills this->D_elem() with x_l^dag gamma_5 Gamma x_r (no scale).
    this->contractG5(x_l, x_r);

    if (accum)
      bhu::launch_axpy_real<Float>(d_acc, this->D_elem(), val, n_real);
    else
      bhu::launch_scal_real<Float>(d_acc, this->D_elem(), val, n_real);
  }

  // Multi-isc variant.  Sums Nsc disconnected-loop contributions
  //   d_acc = sum_i vals[i] * contractG5( x_l[i], x_r[i] )
  // entirely on the device (no D2H, no host BLAS).  After this
  // call the caller normally invokes batchStore() to FT and push.
  // x_l / x_r must have size Nsc >= 1; vals must match.
  void oneEnd_trick_fast_multi(
      const std::vector<PLEGMA_Vector<Float>*>& x_l,
      const std::vector<PLEGMA_Vector<Float>*>& x_r,
      const std::vector<Float>& vals)
  {
    if (this->IsOneD() || this->IsTwoD())
      PLEGMA_error("oneEnd_trick_fast_multi does not handle oneD / twoD");
    const std::size_t Nsc = x_l.size();
    if (Nsc == 0) PLEGMA_error("oneEnd_trick_fast_multi: empty isc list");
    if (x_r.size() != Nsc || vals.size() != Nsc)
      PLEGMA_error("oneEnd_trick_fast_multi: x_l/x_r/vals size mismatch");
    if (!d_acc) allocAccumDevice();

    for (std::size_t i = 0; i < Nsc; ++i) {
      this->contractG5(*x_l[i], *x_r[i]);
      if (i == 0)
        bhu::launch_scal_real<Float>(d_acc, this->D_elem(), vals[i], n_real);
      else
        bhu::launch_axpy_real<Float>(d_acc, this->D_elem(), vals[i], n_real);
    }
  }

  // Same name as base class so existing call sites that say
  //   qLoops.oneEnd_trick(x_l, x_r, val, accum)
  // automatically pick the fast path when the static type is
  // PLEGMA_QLoops_fast.  This intentionally hides (not overrides)
  // the non-virtual base method.
  void oneEnd_trick(PLEGMA_Vector<Float>& x_l,
                    PLEGMA_Vector<Float>& x_r,
                    Float val,
                    bool accum) {
    oneEnd_trick_fast(x_l, x_r, val, accum);
  }

  // Single D2H copy of the accumulator into the inherited host buffer
  // H_loc(), so the standard dumpLoops() pipeline keeps working.
  void flushAccumToHost() {
    if (!d_acc) return;
    cudaMemcpy(this->H_loc(), d_acc, bytes_total, cudaMemcpyDeviceToHost);
    qudaDeviceSynchronize();
  }

  // Optimized batchStore that mirrors PLEGMA_QLoops_patterns::batchStore
  // but feeds the device-side accumulator directly into the FT input
  // (D2D copy) instead of going through host (H2D from H_loc).
  void batchStore(PLEGMA_FT<Float>& ft, bool accumulate = false) {
    if (!d_acc) {
      // No fast path was used; fall back to base behaviour.
      Base::batchStore(ft, accumulate);
      return;
    }

    // Push d_acc into D_elem (device-to-device, ~ms on V100).
    cudaMemcpy(this->D_elem(), d_acc, bytes_total, cudaMemcpyDeviceToDevice);
    ft.apply(*this, FT_GEMV);

    std::vector<hsize_t> shape, lshape, start;
    ft.fill_H5_shapes(shape, lshape, start, 0);
    hsize_t n = 1;
    for (auto l : lshape) n *= l;

    if (accumulate) {
      for (auto& e : this->m_batch_) {
        if (e.group == this->pat_group && e.dataset == this->pat_dataset) {
          const Float* src = ft.H_elem();
          for (hsize_t j = 0; j < n; ++j) e.data[j] += src[j];
          return;
        }
      }
    }

    typename Base::BatchEntry_ e;
    e.group   = this->pat_group;
    e.dataset = this->pat_dataset;
    e.data.assign(ft.H_elem(), ft.H_elem() + n);
    this->m_batch_.push_back(std::move(e));
  }
};
