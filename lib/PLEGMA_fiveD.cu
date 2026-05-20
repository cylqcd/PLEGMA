#include "higher_derivative_patterns.h"
#include "PLEGMA_Correlator_patterns.h"
#include "PLEGMA_global.h"
#include "PLEGMA_oneD.cuh"

#include <mpi.h>
#include <sstream>
#include <vector>
#include <string>

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
const char *dir_name(int d) {
  switch (d) {
    case X:  return "x";
    case Y:  return "y";
    case Z:  return "z";
    case T:  return "t";
    case TP: return "t+";
    case TM: return "t-";
    default: return "?";
  }
}

bool decode_deriv_dir(int d, short &dir, int &time_mode) {
  switch (d) {
    case X:  dir = 0; time_mode = TIME_PLUS;  return true;
    case Y:  dir = 1; time_mode = TIME_PLUS;  return true;
    case Z:  dir = 2; time_mode = TIME_PLUS;  return true;
    case TP: dir = 3; time_mode = TIME_PLUS;  return true;
    case TM: dir = 3; time_mode = TIME_MINUS; return true;
    default: return false;
  }
}

bool cancel_time_pair(int a, int b) {
  return (a == TP && b == TM) || (a == TM && b == TP);
}

void add_spatial_count(int d, int &cx, int &cy, int &cz) {
  if (d == X) ++cx;
  else if (d == Y) ++cy;
  else if (d == Z) ++cz;
}

int infer_gamma_from_counts(int cx, int cy, int cz) {
  const int px = cx % 2;
  const int py = cy % 2;
  const int pz = cz % 2;
  const int odd = px + py + pz;

  if (odd == 0) return T;
  if (odd == 1) {
    if (px) return X;
    if (py) return Y;
    return Z;
  }
  return -1;
}

bool is_valid_counts(int cx, int cy, int cz) {
  return infer_gamma_from_counts(cx, cy, cz) != -1;
}

std::string fmt_derivs(const int *seq, int n) {
  if (n == 0) return "(none)";
  std::string out;
  for (int i = 0; i < n; ++i) {
    if (i) out += " ";
    out += dir_name(seq[i]);
  }
  return out;
}

GAMMAS infer_plegma_gamma_from_counts(int cx, int cy, int cz) {
  const int gamma = infer_gamma_from_counts(cx, cy, cz);
  switch (gamma) {
    case X: return G1;
    case Y: return G2;
    case Z: return G3;
    case T: return G4;
    default: return ONE;
  }
}

const char *plegma_gamma_name(GAMMAS g) {
  switch (g) {
    case G1: return "G1(gamma_x)";
    case G2: return "G2(gamma_y)";
    case G3: return "G3(gamma_z)";
    case G4: return "G4(gamma_t)";
    default: return "INVALID";
  }
}

bool useful_prefix4(int d4, int cx4, int cy4, int cz4) {
  if (is_valid_counts(cx4, cy4, cz4)) return true;

  for (int d5 : DERIV_DIRS) {
    if (cancel_time_pair(d4, d5)) continue;
    int cx5 = cx4, cy5 = cy4, cz5 = cz4;
    add_spatial_count(d5, cx5, cy5, cz5);
    if (is_valid_counts(cx5, cy5, cz5)) return true;
  }
  return false;
}

template<typename Float, typename FloatGauge>
static bool build_deriv_cache(PLEGMA_Propagator<Float> &out,
                              PLEGMA_Propagator<Float> &in,
                              PLEGMA_Gauge<FloatGauge> &gauge,
                              int d) {
  short dir = -1;
  int time_mode = TIME_PLUS;
  if (!decode_deriv_dir(d, dir, time_mode)) return false;

  TIME_V(covariant_derivative(out, in, gauge, dir, time_mode));
  TIME(out.communicateSideGhost());
  return true;
}

static inline void fill_pattern(ThrpPattern &pat,
                                GAMMAS gamma_tag,
                                const int *derivs,
                                int n) {
  pat.gamma = gamma_tag;
  pat.n_deriv = n;
  pat.oneD_conserved = false;
  pat.isZfac = false;
  pat.derivs.fill(-1);
  // for (int i = 0; i < n && i < 5; ++i) pat.derivs[i] = derivs[i];
  // resverse order
  for (int i = 0; i < n && i < 5; ++i) pat.derivs[i] = derivs[n-1-i];
}

template<typename Float>
static bool direct_contraction_from_cache(
    int total_order, int cache_order, const int *derivs, int n,
    int cx, int cy, int cz,
    PLEGMA_Propagator<Float> &bwd_prop,
    PLEGMA_Propagator<Float> &cache_prop,
    PLEGMA_Correlator_patterns<Float> &corr)
{
  const int gamma_logic = infer_gamma_from_counts(cx, cy, cz);
  if (gamma_logic == -1) return false;

  const GAMMAS gamma_tag = infer_plegma_gamma_from_counts(cx, cy, cz);

  PRINT_PATTERN("[%d-direct] %s ; direct contract from cache%d ; derivs = %s\n",
                total_order, plegma_gamma_name(gamma_tag),
                cache_order, fmt_derivs(derivs, n).c_str());

  ThrpPattern pat;
  fill_pattern(pat, gamma_tag, derivs, n);

  TIME(contractNucleonThrp_patterns(corr, bwd_prop, cache_prop, 0, pat, false));
  corr.batchStore();

  return true;
}

template<typename Float, typename FloatGauge>
static bool contract_after_one_more_deriv_from_cache(
    int total_order, int cache_order, const int *derivs, int n, int extra_d,
    int cx, int cy, int cz,
    PLEGMA_Propagator<Float> &bwd_prop,
    PLEGMA_Propagator<Float> &cache_prop,
    PLEGMA_Propagator<Float> &propout,
    PLEGMA_Gauge<FloatGauge> &gauge,
    PLEGMA_Correlator_patterns<Float> &corr)
{
  const int gamma_logic = infer_gamma_from_counts(cx, cy, cz);
  if (gamma_logic == -1) return false;

  const GAMMAS gamma_tag = infer_plegma_gamma_from_counts(cx, cy, cz);

  short dir = -1;
  int time_mode = TIME_PLUS;
  if (!decode_deriv_dir(extra_d, dir, time_mode)) return false;

  TIME_V(covariant_derivative(propout, cache_prop, gauge, dir, time_mode));
  // No ghost communication needed on propout: it is only used for the
  // contraction below (threep_local), which is purely local.

  PRINT_PATTERN("[%d-after-D] %s ; from cache%d apply D_%s then contract ; derivs = %s\n",
                total_order, plegma_gamma_name(gamma_tag), cache_order,
                dir_name(extra_d), fmt_derivs(derivs, n).c_str());

  ThrpPattern pat;
  fill_pattern(pat, gamma_tag, derivs, n);

  TIME(contractNucleonThrp_patterns(corr, bwd_prop, propout, 0, pat, false));
  corr.batchStore();

  return true;
}

template<typename Float, typename FloatGauge>
void contractNucleonThrp_fiveD_patterns(
    PLEGMA_Propagator<Float> &bwd_prop,
    PLEGMA_Propagator<Float> &fwd_prop,
    PLEGMA_Gauge<FloatGauge> &gauge,
    CORR_SPACE corr_space,
    site src,
    int maxQsq,
    const std::string &filename,
    int max_order)
{
  PRINT_PATTERN("contractNucleonThrp_fiveD_patterns: max_order = %d\n", max_order);
  // Allocate derivative-cache propagators lazily based on max_order.
  // Each PLEGMA_Propagator<double>(BOTH,FIRST_SIDE) is ~4.6 GB on a 32^3x64 lattice;
  // allocating only the required caches avoids OOM on large lattices.
  std::unique_ptr<PLEGMA_Propagator<Float>> _pc1, _pc2, _pc3, _pc4, _pc5;
  if (max_order >= 2) _pc1 = std::make_unique<PLEGMA_Propagator<Float>>(BOTH, FIRST_SIDE);
  if (max_order >= 3) _pc2 = std::make_unique<PLEGMA_Propagator<Float>>(BOTH, FIRST_SIDE);
  if (max_order >= 4) _pc3 = std::make_unique<PLEGMA_Propagator<Float>>(BOTH, FIRST_SIDE);
  if (max_order >= 5) _pc4 = std::make_unique<PLEGMA_Propagator<Float>>(BOTH, FIRST_SIDE);
  if (max_order >= 6) _pc5 = std::make_unique<PLEGMA_Propagator<Float>>(BOTH, FIRST_SIDE);
  // Loop guards (if (max_order < N) break) ensure _pcN is never null when dereferenced.
#define prop_cache1 (*_pc1)
#define prop_cache2 (*_pc2)
#define prop_cache3 (*_pc3)
#define prop_cache4 (*_pc4)
#define prop_cache5 (*_pc5)

  // Single correlator object — results accumulated via batchStore()
  PLEGMA_Correlator_patterns<Float> corr(corr_space, src, maxQsq);

  int count1 = 0, count2 = 0, count3 = 0, count4 = 0, count5 = 0, count6 = 0;
  int seq[5] = {0, 0, 0, 0, 0};

  // Communicate gauge and fwd_prop ghost once before any derivative.
  // bwd_prop is only used in threep_local (purely local), no ghost needed.
  gauge.communicateSideGhost();
  fwd_prop.communicateSideGhost();

  // order 1
  {
    ThrpPattern pat;
    pat.gamma = G4;
    pat.n_deriv = 0;
    pat.derivs.fill(-1);
    TIME(contractNucleonThrp_patterns(corr, bwd_prop, fwd_prop, 0, pat, false));
    corr.batchStore();
    ++count1;
    PRINT_PATTERN("[1-direct] G4(gamma_t) ; direct contract from base ; derivs = (none)\n");
  }

  for (int d1 : DERIV_DIRS) {
    if (max_order < 2) break;
    seq[0] = d1;
    if (!build_deriv_cache(prop_cache1, fwd_prop, gauge, d1)) continue;

    int cx1 = 0, cy1 = 0, cz1 = 0;
    add_spatial_count(d1, cx1, cy1, cz1);

    if (direct_contraction_from_cache(2, 1, seq, 1, cx1, cy1, cz1,
                                      bwd_prop, prop_cache1, corr)) {
      ++count2;
    }

    for (int d2 : DERIV_DIRS) {
      if (max_order < 3) break;
      if (cancel_time_pair(d1, d2)) continue;
      seq[1] = d2;

      if (!build_deriv_cache(prop_cache2, prop_cache1, gauge, d2)) continue;

      int cx2 = cx1, cy2 = cy1, cz2 = cz1;
      add_spatial_count(d2, cx2, cy2, cz2);

      if (direct_contraction_from_cache(3, 2, seq, 2, cx2, cy2, cz2,
                                        bwd_prop, prop_cache2, corr)) {
        ++count3;
      }

      for (int d3 : DERIV_DIRS) {
        if (max_order < 4) break;
        if (cancel_time_pair(d2, d3)) continue;
        seq[2] = d3;

        if (!build_deriv_cache(prop_cache3, prop_cache2, gauge, d3)) continue;

        int cx3 = cx2, cy3 = cy2, cz3 = cz2;
        add_spatial_count(d3, cx3, cy3, cz3);

        if (direct_contraction_from_cache(4, 3, seq, 3, cx3, cy3, cz3,
                                          bwd_prop, prop_cache3, corr)) {
          ++count4;
        }

        for (int d4 : DERIV_DIRS) {
          if (max_order < 5) break;
          if (cancel_time_pair(d3, d4)) continue;
          seq[3] = d4;

          int cx4 = cx3, cy4 = cy3, cz4 = cz3;
          add_spatial_count(d4, cx4, cy4, cz4);

          if (!useful_prefix4(d4, cx4, cy4, cz4)) continue;
          if (!build_deriv_cache(prop_cache4, prop_cache3, gauge, d4)) continue;

          if (direct_contraction_from_cache(5, 4, seq, 4, cx4, cy4, cz4,
                                            bwd_prop, prop_cache4, corr)) {
            ++count5;
          }

          for (int d5 : DERIV_DIRS) {
            if (max_order < 6) break;
            if (cancel_time_pair(d4, d5)) continue;
            seq[4] = d5;

            int cx5 = cx4, cy5 = cy4, cz5 = cz4;
            add_spatial_count(d5, cx5, cy5, cz5);

            if (contract_after_one_more_deriv_from_cache(
                    6, 4, seq, 5, d5, cx5, cy5, cz5,
                    bwd_prop, prop_cache4, prop_cache5, gauge, corr)) {
              ++count6;
            }
          }
        }
      }
    }
  }

  PRINT_PATTERN("total order 1: %d\n", count1);
  PRINT_PATTERN("total order 2: %d\n", count2);
  PRINT_PATTERN("total order 3: %d\n", count3);
  PRINT_PATTERN("total order 4: %d\n", count4);
  PRINT_PATTERN("total order 5: %d\n", count5);
  PRINT_PATTERN("total order 6: %d\n", count6);
  PRINT_PATTERN("batch entries: %zu — flushing to HDF5\n", corr.batchSize());
  TIME(corr.batchFlush(filename));
#undef prop_cache1
#undef prop_cache2
#undef prop_cache3
#undef prop_cache4
#undef prop_cache5
}

// Explicit template instantiations
// Float=float, FloatGauge=float
template void contractNucleonThrp_fiveD_patterns<float, float>(
    PLEGMA_Propagator<float>&, PLEGMA_Propagator<float>&, PLEGMA_Gauge<float>&,
    CORR_SPACE, site, int,
    const std::string&, int);

// Float=float, FloatGauge=double
template void contractNucleonThrp_fiveD_patterns<float, double>(
    PLEGMA_Propagator<float>&, PLEGMA_Propagator<float>&, PLEGMA_Gauge<double>&,
    CORR_SPACE, site, int,
    const std::string&, int);

// Float=double, FloatGauge=double
template void contractNucleonThrp_fiveD_patterns<double, double>(
    PLEGMA_Propagator<double>&, PLEGMA_Propagator<double>&, PLEGMA_Gauge<double>&,
    CORR_SPACE, site, int,
    const std::string&, int);

// Float=double, FloatGauge=float
template void contractNucleonThrp_fiveD_patterns<double, float>(
    PLEGMA_Propagator<double>&, PLEGMA_Propagator<double>&, PLEGMA_Gauge<float>&,
    CORR_SPACE, site, int,
    const std::string&, int);
