#pragma once

#include "PLEGMA_Correlator.h"
#include "PLEGMA_QLoops_patterns.h"
#include <PLEGMA_FT.h>

#include <array>
#include <string>
#include <vector>

using namespace plegma;

// enum Dir { X = 0, Y = 1, Z = 2, T = 3, TP = 4, TM = 5 };

// constexpr std::array<int, 5> DERIV_DIRS = {X, Y, Z, TP, TM};

const char *dir_name(int d);

bool cancel_time_pair(int a, int b);
void add_spatial_count(int d, int &cx, int &cy, int &cz);
int infer_gamma_from_counts(int cx, int cy, int cz);
bool is_valid_counts(int cx, int cy, int cz);
std::string fmt_derivs(const int *seq, int n);

GAMMAS infer_plegma_gamma_from_counts(int cx, int cy, int cz);
const char *plegma_gamma_name(GAMMAS g);

bool useful_prefix4(int d4, int cx4, int cy4, int cz4);

bool decode_deriv_dir(int d, short &dir, int &time_mode);

// 主包装函数
template<typename Float, typename FloatGauge>
void contractNucleonThrp_fiveD_patterns(
    PLEGMA_Propagator<Float> &bwd_prop,
    PLEGMA_Propagator<Float> &fwd_prop,
    PLEGMA_Gauge<FloatGauge> &gauge,
    CORR_SPACE corr_space,
    site src,
    int maxQsq,
    const std::string &filename,
    int max_order = 6);

// QLoops G5 contraction with higher-derivative patterns.
// Results are accumulated in qLoops batch (via batchStore).
// When accumulate=true, adds to existing batch entries (for spin-color dilution).
// Caller must call qLoops.batchFlush() afterwards to write HDF5.
//
// Templated on QLoopsT so it accepts either PLEGMA_QLoops_patterns<Float>
// (stock host accumulator) or PLEGMA_QLoops_fast<Float> (device accumulator).
template<typename Float, typename FloatGauge, typename QLoopsT>
void contractG5_fiveD_patterns(
    PLEGMA_Vector<Float> &x_l,
    PLEGMA_Vector<Float> &x_r,
    PLEGMA_Gauge<FloatGauge> &gauge,
    QLoopsT &qLoops,
    PLEGMA_FT<Float> &ft,
    Float val,
    bool accumulate = false,
    int maxOrder = 6);

// Multi-isc variant: takes Nsc copies of (x_l, x_r) and weights `vals`,
// accumulates per-pattern  sum_i vals[i]*contractG5(x_l[i], x_r[i])  on
// the device, then issues ONE FT + push per pattern.  Saves (Nsc-1) FTs
// per pattern relative to calling the single-isc version Nsc times.
// Only PLEGMA_QLoops_fast is supported (device accumulator required).
template<typename Float, typename FloatGauge, typename QLoopsT>
void contractG5_fiveD_patterns_multi(
    std::vector<PLEGMA_Vector<Float>*> &x_l,
    std::vector<PLEGMA_Vector<Float>*> &x_r,
    PLEGMA_Gauge<FloatGauge> &gauge,
    QLoopsT &qLoops,
    PLEGMA_FT<Float> &ft,
    const std::vector<Float> &vals,
    bool accumulate = false,
    int maxOrder = 6);