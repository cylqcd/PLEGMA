#include "higher_derivative_patterns.h"
#include "PLEGMA_QLoops_patterns.h"
#include "PLEGMA_QLoops_fast.h"
#include "PLEGMA_fiveD.h"
#include "PLEGMA_global.h"
#include "PLEGMA_oneD.cuh"

#include <mpi.h>
#include <vector>
#include <string>
#include <memory>

using namespace plegma;

// Print only at PLEGMA verbosity >= 2 (debug)
#define PRINT_PATTERN(...) do { if (HGC_verbosity >= 2) PLEGMA_printf(__VA_ARGS__); } while(0)

static std::vector<double> runtime;
#define TIME(fnc)                                                              \
  runtime.push_back(MPI_Wtime());                                              \
  fnc;                                                                         \
  PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back());   \
  runtime.pop_back()// Like TIME but only prints at PLEGMA verbosity >= 2 (debug)
#define TIME_V(fnc)                                                                                          \
  runtime.push_back(MPI_Wtime());                                                                            \
  fnc;                                                                                                       \
  if (HGC_verbosity >= 2)                                                                                    \
    PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back());                           \
  runtime.pop_back()
// Build derivative cache for a single spinor
template<typename Float, typename FloatGauge>
static bool build_deriv_cache_vec(PLEGMA_Vector<Float> &out,
                                  PLEGMA_Vector<Float> &in,
                                  PLEGMA_Gauge<FloatGauge> &gauge,
                                  int d) {
  short dir = -1;
  int time_mode = TIME_PLUS;
  if (!decode_deriv_dir(d, dir, time_mode)) return false;

  TIME_V(covariant_derivative(out, in, gauge, dir, time_mode));
  TIME_V(out.communicateGhost());
  return true;
}

bool useful_prefix4(int d, int cx, int cy, int cz);

static inline bool useful_prefix3(int d, int cx, int cy, int cz)
{
  return useful_prefix4(d, cx, cy, cz);
}

template<typename Float, typename QLoopsT>
static bool direct_contraction_from_cache_G5(
    int total_order, int cache_order, const int *derivs, int n,
    int cx, int cy, int cz,
    PLEGMA_Vector<Float> &x_l,
    PLEGMA_Vector<Float> &cache_vec,
    QLoopsT &qLoops,
    PLEGMA_FT<Float> &ft,
    Float val,
    bool accumulate)
{
  const int gamma_logic = infer_gamma_from_counts(cx, cy, cz);
  if (gamma_logic == -1) return false;

  const GAMMAS gamma_tag = infer_plegma_gamma_from_counts(cx, cy, cz);

  PRINT_PATTERN("[%d-direct-G5] %s ; direct contract from cache%d ; derivs = %s\n",
                total_order, plegma_gamma_name(gamma_tag),
                cache_order, fmt_derivs(derivs, n).c_str());

  ThrpPattern pat;
  pat.gamma = gamma_tag;
  pat.n_deriv = n;
  pat.oneD_conserved = false;
  pat.isZfac = false;
  pat.derivs.fill(-1);
  for (int i = 0; i < n && i < 5; ++i) pat.derivs[i] = derivs[n-1-i];

  TIME(contractG5_patterns(qLoops, x_l, cache_vec, val, false, pat));
  TIME(qLoops.batchStore(ft, accumulate));

  return true;
}

template<typename Float, typename FloatGauge, typename QLoopsT>
static bool contract_after_one_more_deriv_from_cache_G5(
    int total_order, int cache_order, const int *derivs, int n, int extra_d,
    int cx, int cy, int cz,
    PLEGMA_Vector<Float> &x_l,
    PLEGMA_Vector<Float> &cache_vec,
    PLEGMA_Vector<Float> &vecout,
    PLEGMA_Gauge<FloatGauge> &gauge,
    QLoopsT &qLoops,
    PLEGMA_FT<Float> &ft,
    Float val,
    bool accumulate)
{
  const int gamma_logic = infer_gamma_from_counts(cx, cy, cz);
  if (gamma_logic == -1) return false;

  const GAMMAS gamma_tag = infer_plegma_gamma_from_counts(cx, cy, cz);

  short dir = -1;
  int time_mode = TIME_PLUS;
  if (!decode_deriv_dir(extra_d, dir, time_mode)) return false;

  TIME_V(covariant_derivative(vecout, cache_vec, gauge, dir, time_mode));
  TIME_V(vecout.communicateGhost());

  PRINT_PATTERN("[%d-after-D-G5] %s ; from cache%d apply D_%s then contract ; derivs = %s\n",
                total_order, plegma_gamma_name(gamma_tag), cache_order,
                dir_name(extra_d), fmt_derivs(derivs, n).c_str());

  ThrpPattern pat;
  pat.gamma = gamma_tag;
  pat.n_deriv = n;
  pat.oneD_conserved = false;
  pat.isZfac = false;
  pat.derivs.fill(-1);
  for (int i = 0; i < n && i < 5; ++i) pat.derivs[i] = derivs[n-1-i];

  TIME(contractG5_patterns(qLoops, x_l, vecout, val, false, pat));
  TIME(qLoops.batchStore(ft, accumulate));

  return true;
}

template<typename Float, typename FloatGauge, typename QLoopsT>
void contractG5_fiveD_patterns(
    PLEGMA_Vector<Float> &x_l,
    PLEGMA_Vector<Float> &x_r,
    PLEGMA_Gauge<FloatGauge> &gauge,
    QLoopsT &qLoops,
    PLEGMA_FT<Float> &ft,
    Float val,
    bool accumulate,
    int maxOrder)
{
  PRINT_PATTERN("contractG5_fiveD_patterns: maxOrder = %d\n", maxOrder);
  PLEGMA_Vector<Float> vec_cache1(BOTH, FIRST_SIDE);
  PLEGMA_Vector<Float> vec_cache2(BOTH, FIRST_SIDE);
  PLEGMA_Vector<Float> vec_cache3(BOTH, FIRST_SIDE);
  PLEGMA_Vector<Float> vec_cache4(BOTH, FIRST_SIDE);
  PLEGMA_Vector<Float> vec_cache5(BOTH, FIRST_SIDE);

  int count1 = 0, count2 = 0, count3 = 0, count4 = 0, count5 = 0, count6 = 0;
  int seq[5] = {0, 0, 0, 0, 0};

  // order 0: local (no derivative)
  {
    ThrpPattern pat;
    pat.gamma = G4;
    pat.n_deriv = 0;
    pat.derivs.fill(-1);
    contractG5_patterns(qLoops, x_l, x_r, val, false, pat);
    qLoops.batchStore(ft, accumulate);
    ++count1;
    PRINT_PATTERN("[1-direct-G5] G4(gamma_t) ; direct contract from base ; derivs = (none)\n");
  }

  for (int d1 : DERIV_DIRS) {
    if (maxOrder < 2) break;
    seq[0] = d1;
    if (!build_deriv_cache_vec(vec_cache1, x_r, gauge, d1)) continue;

    int cx1 = 0, cy1 = 0, cz1 = 0;
    add_spatial_count(d1, cx1, cy1, cz1);

    if (direct_contraction_from_cache_G5(2, 1, seq, 1, cx1, cy1, cz1,
                                         x_l, vec_cache1, qLoops, ft,
                                         val, accumulate)) {
      ++count2;
    }

    for (int d2 : DERIV_DIRS) {
      if (maxOrder < 3) break;
      if (cancel_time_pair(d1, d2)) continue;
      seq[1] = d2;

      if (!build_deriv_cache_vec(vec_cache2, vec_cache1, gauge, d2)) continue;

      int cx2 = cx1, cy2 = cy1, cz2 = cz1;
      add_spatial_count(d2, cx2, cy2, cz2);

      if (direct_contraction_from_cache_G5(3, 2, seq, 2, cx2, cy2, cz2,
                                           x_l, vec_cache2, qLoops, ft,
                                           val, accumulate)) {
        ++count3;
      }

      for (int d3 : DERIV_DIRS) {
        if (maxOrder < 4) break;
        if (cancel_time_pair(d2, d3)) continue;
        seq[2] = d3;

        if (!build_deriv_cache_vec(vec_cache3, vec_cache2, gauge, d3)) continue;

        int cx3 = cx2, cy3 = cy2, cz3 = cz2;
        add_spatial_count(d3, cx3, cy3, cz3);

        if (direct_contraction_from_cache_G5(4, 3, seq, 3, cx3, cy3, cz3,
                                             x_l, vec_cache3, qLoops, ft,
                                             val, accumulate)) {
          ++count4;
        }

        for (int d4 : DERIV_DIRS) {
          if (maxOrder < 5) break;
          if (cancel_time_pair(d3, d4)) continue;
          seq[3] = d4;

          int cx4 = cx3, cy4 = cy3, cz4 = cz3;
          add_spatial_count(d4, cx4, cy4, cz4);

          if (!useful_prefix4(d4, cx4, cy4, cz4)) continue;
          if (!build_deriv_cache_vec(vec_cache4, vec_cache3, gauge, d4)) continue;

          if (direct_contraction_from_cache_G5(5, 4, seq, 4, cx4, cy4, cz4,
                                               x_l, vec_cache4, qLoops, ft,
                                               val, accumulate)) {
            ++count5;
          }

          for (int d5 : DERIV_DIRS) {
            if (maxOrder < 6) break;
            if (cancel_time_pair(d4, d5)) continue;
            seq[4] = d5;

            int cx5 = cx4, cy5 = cy4, cz5 = cz4;
            add_spatial_count(d5, cx5, cy5, cz5);

            if (contract_after_one_more_deriv_from_cache_G5(
                    6, 4, seq, 5, d5, cx5, cy5, cz5,
                    x_l, vec_cache4, vec_cache5, gauge, qLoops, ft,
                    val, accumulate)) {
              ++count6;
            }
          }
        }
      }
    }
  }

  PRINT_PATTERN("G5 patterns total order 1: %d\n", count1);
  PRINT_PATTERN("G5 patterns total order 2: %d\n", count2);
  PRINT_PATTERN("G5 patterns total order 3: %d\n", count3);
  PRINT_PATTERN("G5 patterns total order 4: %d\n", count4);
  PRINT_PATTERN("G5 patterns total order 5: %d\n", count5);
  PRINT_PATTERN("G5 patterns total order 6: %d\n", count6);
  PRINT_PATTERN("G5 batch entries so far: %zu\n", qLoops.batchSize());
}

// =====================================================================
// Multi-isc variants.  These take Nsc copies of (x_l, x_r) and an
// Nsc-length vector of weights `vals`, and (per pattern) accumulate
//   sum_i vals[i] * contractG5( x_l[i], x_r[i] )
// on the device into qLoops's d_acc, then issue a single FT + push.
// Compared to calling the single-isc version Nsc times this saves
// (Nsc - 1) FTs per pattern.
// =====================================================================

template<typename Float, typename FloatGauge>
static bool build_deriv_cache_vec_multi(
    std::vector<std::unique_ptr<PLEGMA_Vector<Float>>>& out,
    std::vector<std::unique_ptr<PLEGMA_Vector<Float>>>& in,
    PLEGMA_Gauge<FloatGauge>& gauge,
    int d)
{
  short dir = -1;
  int time_mode = TIME_PLUS;
  if (!decode_deriv_dir(d, dir, time_mode)) return false;
  const std::size_t Nsc = in.size();
  for (std::size_t i = 0; i < Nsc; ++i) {
    TIME_V(covariant_derivative(*out[i], *in[i], gauge, dir, time_mode));
    TIME_V(out[i]->communicateGhost());
  }
  return true;
}

template<typename Float, typename QLoopsT>
static bool direct_contraction_from_cache_G5_multi(
    int total_order, int cache_order, const int* derivs, int n,
    int cx, int cy, int cz,
    std::vector<PLEGMA_Vector<Float>*>& x_l,
    std::vector<std::unique_ptr<PLEGMA_Vector<Float>>>& cache_vec,
    QLoopsT& qLoops,
    PLEGMA_FT<Float>& ft,
    const std::vector<Float>& vals,
    bool accumulate)
{
  const int gamma_logic = infer_gamma_from_counts(cx, cy, cz);
  if (gamma_logic == -1) return false;
  const GAMMAS gamma_tag = infer_plegma_gamma_from_counts(cx, cy, cz);

  PRINT_PATTERN("[%d-direct-G5-multi] %s ; cache%d ; derivs = %s\n",
                total_order, plegma_gamma_name(gamma_tag),
                cache_order, fmt_derivs(derivs, n).c_str());

  ThrpPattern pat;
  pat.gamma = gamma_tag;
  pat.n_deriv = n;
  pat.oneD_conserved = false;
  pat.isZfac = false;
  pat.derivs.fill(-1);
  for (int i = 0; i < n && i < 5; ++i) pat.derivs[i] = derivs[n-1-i];

  const std::size_t Nsc = x_l.size();
  std::vector<PLEGMA_Vector<Float>*> cache_ptrs(Nsc);
  for (std::size_t i = 0; i < Nsc; ++i) cache_ptrs[i] = cache_vec[i].get();

  contractG5_patterns_multi(qLoops, x_l, cache_ptrs, vals, pat);
  qLoops.batchStore(ft, accumulate);
  return true;
}

template<typename Float, typename FloatGauge, typename QLoopsT>
static bool contract_after_one_more_deriv_from_cache_G5_multi(
    int total_order, int cache_order, const int* derivs, int n, int extra_d,
    int cx, int cy, int cz,
    std::vector<PLEGMA_Vector<Float>*>& x_l,
    std::vector<std::unique_ptr<PLEGMA_Vector<Float>>>& cache_vec,
    std::vector<std::unique_ptr<PLEGMA_Vector<Float>>>& vecout,
    PLEGMA_Gauge<FloatGauge>& gauge,
    QLoopsT& qLoops,
    PLEGMA_FT<Float>& ft,
    const std::vector<Float>& vals,
    bool accumulate)
{
  const int gamma_logic = infer_gamma_from_counts(cx, cy, cz);
  if (gamma_logic == -1) return false;
  const GAMMAS gamma_tag = infer_plegma_gamma_from_counts(cx, cy, cz);

  short dir = -1;
  int time_mode = TIME_PLUS;
  if (!decode_deriv_dir(extra_d, dir, time_mode)) return false;

  const std::size_t Nsc = cache_vec.size();
  for (std::size_t i = 0; i < Nsc; ++i) {
    TIME_V(covariant_derivative(*vecout[i], *cache_vec[i], gauge, dir, time_mode));
    // TIME_V(vecout[i]->communicateGhost());
  }

  PRINT_PATTERN("[%d-after-D-G5-multi] %s ; cache%d D_%s ; derivs = %s\n",
                total_order, plegma_gamma_name(gamma_tag), cache_order,
                dir_name(extra_d), fmt_derivs(derivs, n).c_str());

  ThrpPattern pat;
  pat.gamma = gamma_tag;
  pat.n_deriv = n;
  pat.oneD_conserved = false;
  pat.isZfac = false;
  pat.derivs.fill(-1);
  for (int i = 0; i < n && i < 5; ++i) pat.derivs[i] = derivs[n-1-i];

  std::vector<PLEGMA_Vector<Float>*> vecout_ptrs(Nsc);
  for (std::size_t i = 0; i < Nsc; ++i) vecout_ptrs[i] = vecout[i].get();

  contractG5_patterns_multi(qLoops, x_l, vecout_ptrs, vals, pat);
  qLoops.batchStore(ft, accumulate);
  return true;
}

template<typename Float, typename FloatGauge, typename QLoopsT>
void contractG5_fourD_patterns_multi(
    std::vector<PLEGMA_Vector<Float>*>& x_l,
    std::vector<PLEGMA_Vector<Float>*>& x_r,
    PLEGMA_Gauge<FloatGauge>& gauge,
    QLoopsT& qLoops,
    PLEGMA_FT<Float>& ft,
    const std::vector<Float>& vals,
    bool accumulate,
    int maxOrder)
{
  const std::size_t Nsc = x_l.size();
  if (Nsc == 0) PLEGMA_error("contractG5_fourD_patterns_multi: empty isc list");
  if (x_r.size() != Nsc || vals.size() != Nsc)
    PLEGMA_error("contractG5_fourD_patterns_multi: size mismatch (Nsc=%zu)", Nsc);

  PRINT_PATTERN("contractG5_fourD_patterns_multi: maxOrder=%d Nsc=%zu\n",
                maxOrder, Nsc);

  std::vector<std::unique_ptr<PLEGMA_Vector<Float>>>
      vec_cache1(Nsc), vec_cache2(Nsc), vec_cache3(Nsc), vec_cache4(Nsc);
  for (std::size_t i = 0; i < Nsc; ++i) {
    vec_cache1[i].reset(new PLEGMA_Vector<Float>(BOTH, FIRST_SIDE));
    vec_cache2[i].reset(new PLEGMA_Vector<Float>(BOTH, FIRST_SIDE));
    vec_cache3[i].reset(new PLEGMA_Vector<Float>(BOTH, FIRST_SIDE));
    vec_cache4[i].reset(new PLEGMA_Vector<Float>(BOTH, FIRST_SIDE));
  }

  int count1 = 0, count2 = 0, count3 = 0, count4 = 0, count5 = 0;
  int seq[4] = {0, 0, 0, 0};

  {
    ThrpPattern pat;
    pat.gamma = G4;
    pat.n_deriv = 0;
    pat.derivs.fill(-1);
    contractG5_patterns_multi(qLoops, x_l, x_r, vals, pat);
    qLoops.batchStore(ft, accumulate);
    ++count1;
  }

  for (int d1 : DERIV_DIRS) {
    if (maxOrder < 2) break;
    seq[0] = d1;
    {
      short dir = -1; int time_mode = TIME_PLUS;
      if (!decode_deriv_dir(d1, dir, time_mode)) continue;
      for (std::size_t i = 0; i < Nsc; ++i) {
        covariant_derivative(*vec_cache1[i], *x_r[i], gauge, dir, time_mode);
        vec_cache1[i]->communicateGhost();
      }
    }

    int cx1 = 0, cy1 = 0, cz1 = 0;
    add_spatial_count(d1, cx1, cy1, cz1);

    if (direct_contraction_from_cache_G5_multi(
            2, 1, seq, 1, cx1, cy1, cz1,
            x_l, vec_cache1, qLoops, ft, vals, accumulate)) {
      ++count2;
    }

    for (int d2 : DERIV_DIRS) {
      if (maxOrder < 3) break;
      if (cancel_time_pair(d1, d2)) continue;
      seq[1] = d2;
      if (!build_deriv_cache_vec_multi(vec_cache2, vec_cache1, gauge, d2))
        continue;

      int cx2 = cx1, cy2 = cy1, cz2 = cz1;
      add_spatial_count(d2, cx2, cy2, cz2);

      if (direct_contraction_from_cache_G5_multi(
              3, 2, seq, 2, cx2, cy2, cz2,
              x_l, vec_cache2, qLoops, ft, vals, accumulate)) {
        ++count3;
      }

      for (int d3 : DERIV_DIRS) {
        if (maxOrder < 4) break;
        if (cancel_time_pair(d2, d3)) continue;
        seq[2] = d3;

        int cx3 = cx2, cy3 = cy2, cz3 = cz2;
        add_spatial_count(d3, cx3, cy3, cz3);

        // For maxOrder=5, keep the four-derivative branch only when the
        // prefix can still lead to a valid contraction.
        if (!useful_prefix3(d3, cx3, cy3, cz3)) continue;
        if (!build_deriv_cache_vec_multi(vec_cache3, vec_cache2, gauge, d3))
          continue;

        if (direct_contraction_from_cache_G5_multi(
                4, 3, seq, 3, cx3, cy3, cz3,
                x_l, vec_cache3, qLoops, ft, vals, accumulate)) {
          ++count4;
        }

        for (int d4 : DERIV_DIRS) {
          if (maxOrder < 5) break;
          if (cancel_time_pair(d3, d4)) continue;
          seq[3] = d4;

          int cx4 = cx3, cy4 = cy3, cz4 = cz3;
          add_spatial_count(d4, cx4, cy4, cz4);

          if (contract_after_one_more_deriv_from_cache_G5_multi(
                  5, 3, seq, 4, d4, cx4, cy4, cz4,
                  x_l, vec_cache3, vec_cache4, gauge, qLoops, ft,
                  vals, accumulate)) {
            ++count5;
          }
        }
      }
    }
  }

  PRINT_PATTERN("G5 fourD multi patterns total order 1: %d\n", count1);
  PRINT_PATTERN("G5 fourD multi patterns total order 2: %d\n", count2);
  PRINT_PATTERN("G5 fourD multi patterns total order 3: %d\n", count3);
  PRINT_PATTERN("G5 fourD multi patterns total order 4: %d\n", count4);
  PRINT_PATTERN("G5 fourD multi patterns total order 5: %d\n", count5);
  PRINT_PATTERN("G5 batch entries so far: %zu\n", qLoops.batchSize());
}

template<typename Float, typename FloatGauge, typename QLoopsT>
void contractG5_fiveD_patterns_multi(
    std::vector<PLEGMA_Vector<Float>*>& x_l,
    std::vector<PLEGMA_Vector<Float>*>& x_r,
    PLEGMA_Gauge<FloatGauge>& gauge,
    QLoopsT& qLoops,
    PLEGMA_FT<Float>& ft,
    const std::vector<Float>& vals,
    bool accumulate,
    int maxOrder)
{
  const std::size_t Nsc = x_l.size();
  if (Nsc == 0) PLEGMA_error("contractG5_fiveD_patterns_multi: empty isc list");
  if (x_r.size() != Nsc || vals.size() != Nsc)
    PLEGMA_error("contractG5_fiveD_patterns_multi: size mismatch (Nsc=%zu)", Nsc);

  if (maxOrder == 5) {
    contractG5_fourD_patterns_multi(x_l, x_r, gauge, qLoops, ft, vals, accumulate, maxOrder);
    return;
  }

  PRINT_PATTERN("contractG5_fiveD_patterns_multi: maxOrder=%d Nsc=%zu\n",
                maxOrder, Nsc);

  // Per-isc cache vectors (unique_ptr because PLEGMA_Vector is non-copyable
  // and non-moveable, so it cannot live directly in std::vector).
  std::vector<std::unique_ptr<PLEGMA_Vector<Float>>>
      vec_cache1(Nsc), vec_cache2(Nsc), vec_cache3(Nsc),
      vec_cache4(Nsc), vec_cache5(Nsc);
  for (std::size_t i = 0; i < Nsc; ++i) {
    vec_cache1[i].reset(new PLEGMA_Vector<Float>(BOTH, FIRST_SIDE));
    vec_cache2[i].reset(new PLEGMA_Vector<Float>(BOTH, FIRST_SIDE));
    vec_cache3[i].reset(new PLEGMA_Vector<Float>(BOTH, FIRST_SIDE));
    vec_cache4[i].reset(new PLEGMA_Vector<Float>(BOTH, FIRST_SIDE));
    vec_cache5[i].reset(new PLEGMA_Vector<Float>(BOTH, FIRST_SIDE));
  }

  int count1 = 0, count2 = 0, count3 = 0, count4 = 0, count5 = 0, count6 = 0;
  int seq[5] = {0, 0, 0, 0, 0};

  // order 0: local
  {
    ThrpPattern pat;
    pat.gamma = G4;
    pat.n_deriv = 0;
    pat.derivs.fill(-1);
    contractG5_patterns_multi(qLoops, x_l, x_r, vals, pat);
    qLoops.batchStore(ft, accumulate);
    ++count1;
  }

  for (int d1 : DERIV_DIRS) {
    if (maxOrder < 2) break;
    seq[0] = d1;
    // Build first-level cache vec_cache1[i] = D_{d1} x_r[i] for each isc.
    {
      short dir = -1; int time_mode = TIME_PLUS;
      if (!decode_deriv_dir(d1, dir, time_mode)) continue;
      for (std::size_t i = 0; i < Nsc; ++i) {
        covariant_derivative(*vec_cache1[i], *x_r[i], gauge, dir, time_mode);
        vec_cache1[i]->communicateGhost();
      }
    }

    int cx1 = 0, cy1 = 0, cz1 = 0;
    add_spatial_count(d1, cx1, cy1, cz1);

    if (direct_contraction_from_cache_G5_multi(
            2, 1, seq, 1, cx1, cy1, cz1,
            x_l, vec_cache1, qLoops, ft, vals, accumulate)) {
      ++count2;
    }

    for (int d2 : DERIV_DIRS) {
      if (maxOrder < 3) break;
      if (cancel_time_pair(d1, d2)) continue;
      seq[1] = d2;
      if (!build_deriv_cache_vec_multi(vec_cache2, vec_cache1, gauge, d2))
        continue;

      int cx2 = cx1, cy2 = cy1, cz2 = cz1;
      add_spatial_count(d2, cx2, cy2, cz2);

      if (direct_contraction_from_cache_G5_multi(
              3, 2, seq, 2, cx2, cy2, cz2,
              x_l, vec_cache2, qLoops, ft, vals, accumulate)) {
        ++count3;
      }

      for (int d3 : DERIV_DIRS) {
        if (maxOrder < 4) break;
        if (cancel_time_pair(d2, d3)) continue;
        seq[2] = d3;
        if (!build_deriv_cache_vec_multi(vec_cache3, vec_cache2, gauge, d3))
          continue;

        int cx3 = cx2, cy3 = cy2, cz3 = cz2;
        add_spatial_count(d3, cx3, cy3, cz3);

        if (direct_contraction_from_cache_G5_multi(
                4, 3, seq, 3, cx3, cy3, cz3,
                x_l, vec_cache3, qLoops, ft, vals, accumulate)) {
          ++count4;
        }

        for (int d4 : DERIV_DIRS) {
          if (maxOrder < 5) break;
          if (cancel_time_pair(d3, d4)) continue;
          seq[3] = d4;

          int cx4 = cx3, cy4 = cy3, cz4 = cz3;
          add_spatial_count(d4, cx4, cy4, cz4);

          if (!useful_prefix4(d4, cx4, cy4, cz4)) continue;
          if (!build_deriv_cache_vec_multi(vec_cache4, vec_cache3, gauge, d4))
            continue;

          if (direct_contraction_from_cache_G5_multi(
                  5, 4, seq, 4, cx4, cy4, cz4,
                  x_l, vec_cache4, qLoops, ft, vals, accumulate)) {
            ++count5;
          }

          for (int d5 : DERIV_DIRS) {
            if (maxOrder < 6) break;
            if (cancel_time_pair(d4, d5)) continue;
            seq[4] = d5;

            int cx5 = cx4, cy5 = cy4, cz5 = cz4;
            add_spatial_count(d5, cx5, cy5, cz5);

            if (contract_after_one_more_deriv_from_cache_G5_multi(
                    6, 4, seq, 5, d5, cx5, cy5, cz5,
                    x_l, vec_cache4, vec_cache5, gauge, qLoops, ft,
                    vals, accumulate)) {
              ++count6;
            }
          }
        }
      }
    }
  }

  PRINT_PATTERN("G5 multi patterns total order 1: %d\n", count1);
  PRINT_PATTERN("G5 multi patterns total order 2: %d\n", count2);
  PRINT_PATTERN("G5 multi patterns total order 3: %d\n", count3);
  PRINT_PATTERN("G5 multi patterns total order 4: %d\n", count4);
  PRINT_PATTERN("G5 multi patterns total order 5: %d\n", count5);
  PRINT_PATTERN("G5 multi patterns total order 6: %d\n", count6);
  PRINT_PATTERN("G5 batch entries so far: %zu\n", qLoops.batchSize());
}

// Explicit template instantiations
// Float=float, FloatGauge=float
template void contractG5_fiveD_patterns<float, float, PLEGMA_QLoops_patterns<float>>(
    PLEGMA_Vector<float>&, PLEGMA_Vector<float>&, PLEGMA_Gauge<float>&,
    PLEGMA_QLoops_patterns<float>&, PLEGMA_FT<float>&, float,
    bool, int);

// Float=float, FloatGauge=double
template void contractG5_fiveD_patterns<float, double, PLEGMA_QLoops_patterns<float>>(
    PLEGMA_Vector<float>&, PLEGMA_Vector<float>&, PLEGMA_Gauge<double>&,
    PLEGMA_QLoops_patterns<float>&, PLEGMA_FT<float>&, float,
    bool, int);

// Float=double, FloatGauge=double
template void contractG5_fiveD_patterns<double, double, PLEGMA_QLoops_patterns<double>>(
    PLEGMA_Vector<double>&, PLEGMA_Vector<double>&, PLEGMA_Gauge<double>&,
    PLEGMA_QLoops_patterns<double>&, PLEGMA_FT<double>&, double,
    bool, int);

// Float=double, FloatGauge=float
template void contractG5_fiveD_patterns<double, float, PLEGMA_QLoops_patterns<double>>(
    PLEGMA_Vector<double>&, PLEGMA_Vector<double>&, PLEGMA_Gauge<float>&,
    PLEGMA_QLoops_patterns<double>&, PLEGMA_FT<double>&, double,
    bool, int);

// Fast variants (PLEGMA_QLoops_fast)
template void contractG5_fiveD_patterns<float, float, PLEGMA_QLoops_fast<float>>(
    PLEGMA_Vector<float>&, PLEGMA_Vector<float>&, PLEGMA_Gauge<float>&,
    PLEGMA_QLoops_fast<float>&, PLEGMA_FT<float>&, float,
    bool, int);

template void contractG5_fiveD_patterns<float, double, PLEGMA_QLoops_fast<float>>(
    PLEGMA_Vector<float>&, PLEGMA_Vector<float>&, PLEGMA_Gauge<double>&,
    PLEGMA_QLoops_fast<float>&, PLEGMA_FT<float>&, float,
    bool, int);

template void contractG5_fiveD_patterns<double, double, PLEGMA_QLoops_fast<double>>(
    PLEGMA_Vector<double>&, PLEGMA_Vector<double>&, PLEGMA_Gauge<double>&,
    PLEGMA_QLoops_fast<double>&, PLEGMA_FT<double>&, double,
    bool, int);

template void contractG5_fiveD_patterns<double, float, PLEGMA_QLoops_fast<double>>(
    PLEGMA_Vector<double>&, PLEGMA_Vector<double>&, PLEGMA_Gauge<float>&,
    PLEGMA_QLoops_fast<double>&, PLEGMA_FT<double>&, double,
    bool, int);

// Multi-isc instantiations (only the fast QLoops variant is supported,
// since multi accumulation requires the device-side accumulator).
template void contractG5_fiveD_patterns_multi<float, float, PLEGMA_QLoops_fast<float>>(
    std::vector<PLEGMA_Vector<float>*>&, std::vector<PLEGMA_Vector<float>*>&,
    PLEGMA_Gauge<float>&, PLEGMA_QLoops_fast<float>&, PLEGMA_FT<float>&,
    const std::vector<float>&, bool, int);

template void contractG5_fiveD_patterns_multi<float, double, PLEGMA_QLoops_fast<float>>(
    std::vector<PLEGMA_Vector<float>*>&, std::vector<PLEGMA_Vector<float>*>&,
    PLEGMA_Gauge<double>&, PLEGMA_QLoops_fast<float>&, PLEGMA_FT<float>&,
    const std::vector<float>&, bool, int);

template void contractG5_fiveD_patterns_multi<double, double, PLEGMA_QLoops_fast<double>>(
    std::vector<PLEGMA_Vector<double>*>&, std::vector<PLEGMA_Vector<double>*>&,
    PLEGMA_Gauge<double>&, PLEGMA_QLoops_fast<double>&, PLEGMA_FT<double>&,
    const std::vector<double>&, bool, int);

template void contractG5_fiveD_patterns_multi<double, float, PLEGMA_QLoops_fast<double>>(
    std::vector<PLEGMA_Vector<double>*>&, std::vector<PLEGMA_Vector<double>*>&,
    PLEGMA_Gauge<float>&, PLEGMA_QLoops_fast<double>&, PLEGMA_FT<double>&,
    const std::vector<double>&, bool, int);

// Multi-isc fourD instantiations.
template void contractG5_fourD_patterns_multi<float, float, PLEGMA_QLoops_fast<float>>(
  std::vector<PLEGMA_Vector<float>*>&, std::vector<PLEGMA_Vector<float>*>&,
  PLEGMA_Gauge<float>&, PLEGMA_QLoops_fast<float>&, PLEGMA_FT<float>&,
  const std::vector<float>&, bool, int);

template void contractG5_fourD_patterns_multi<float, double, PLEGMA_QLoops_fast<float>>(
  std::vector<PLEGMA_Vector<float>*>&, std::vector<PLEGMA_Vector<float>*>&,
  PLEGMA_Gauge<double>&, PLEGMA_QLoops_fast<float>&, PLEGMA_FT<float>&,
  const std::vector<float>&, bool, int);

template void contractG5_fourD_patterns_multi<double, double, PLEGMA_QLoops_fast<double>>(
  std::vector<PLEGMA_Vector<double>*>&, std::vector<PLEGMA_Vector<double>*>&,
  PLEGMA_Gauge<double>&, PLEGMA_QLoops_fast<double>&, PLEGMA_FT<double>&,
  const std::vector<double>&, bool, int);

template void contractG5_fourD_patterns_multi<double, float, PLEGMA_QLoops_fast<double>>(
  std::vector<PLEGMA_Vector<double>*>&, std::vector<PLEGMA_Vector<double>*>&,
  PLEGMA_Gauge<float>&, PLEGMA_QLoops_fast<double>&, PLEGMA_FT<double>&,
  const std::vector<double>&, bool, int);