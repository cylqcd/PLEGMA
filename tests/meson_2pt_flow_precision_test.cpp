// tests/meson_2pt_flow_precision_test.cpp
//
// Precision study for meson 2pt correlators with Wilson-flowed propagators.
//
// Two comparisons are performed:
//
//   (A) Spinor storage precision during flow:
//       Same double-precision Laplace kernel, but spinor stored as
//       float vs double.  The difference measures pure float rounding in
//       the intermediate spinor buffers.
//
//   (B) Contraction accumulation precision:
//       The float-flowed and double-flowed propagators are each contracted
//       with PLEGMA_Correlator<float> (float accumulators) and, if the
//       library is built with the double instantiation, also with
//       PLEGMA_Correlator<double>.
//
// Usage (same flags as meson_2pt_3pt_flowed):
//   mpirun -n 1 ./meson_2pt_flow_prec --load-gauge <file> \
//          --flow-steps 500 --flow-epsilon 0.01 \
//          [--src-filename <file>]   # optional: PLEGMA source-list file
//
// Output: HDF5 correlator files in the working directory.

#include "PLEGMA_global.h"
#include "QUDA_wilson_flow.h"
#include "PLEGMA_fiveD.h"
#include "quda.h"

#include <PLEGMA.h>
#include <PLEGMA_utils.h>

#include <mpi.h>
#include <memory>
#include <string>
#include <vector>

std::vector<std::thread> threads;
using namespace plegma;
using namespace quda;

static std::vector<double> runtime;
#define TIME(fnc)                                                              \
  runtime.push_back(MPI_Wtime());                                              \
  fnc;                                                                         \
  PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back()); \
  runtime.pop_back()

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct PropMetrics {
  double norm_ref  = 0.0;
  double norm_test = 0.0;
  double norm_diff = 0.0;
  double rel_diff  = 0.0;
};

// Compare two float propagators column by column and report max relative diff.
static PropMetrics compare_props(PLEGMA_Propagator<float> &ref,
                                 PLEGMA_Propagator<float> &test)
{
  PropMetrics m;
  for (int isc = 0; isc < 12; ++isc) {
    int spin = isc / 3, col = isc % 3;

    PLEGMA_Vector<float> va, vb, vd;
    va.absorb(ref,  spin, col);
    vb.absorb(test, spin, col);
    vd.copy(va);
    vd.add(vb, std::complex<float>(-1.0f, 0.0f));

    const double na = va.norm();
    const double nd = vd.norm();
    const double rd = (na > 0.0) ? nd / na : nd;

    m.norm_ref  += na * na;
    m.norm_test += vb.norm() * vb.norm();
    m.norm_diff += nd * nd;
    PLEGMA_printf("  col (%d,%d): |ref|=%.6e  |test|=%.6e  rel_diff=%.6e\n",
                  spin, col, na, vb.norm(), rd);
  }
  m.norm_ref  = std::sqrt(m.norm_ref);
  m.norm_test = std::sqrt(m.norm_test);
  m.norm_diff = std::sqrt(m.norm_diff);
  m.rel_diff  = (m.norm_ref > 0.0) ? m.norm_diff / m.norm_ref : m.norm_diff;
  return m;
}

static void print_prop_metrics(const char *label, const PropMetrics &m)
{
  PLEGMA_printf("[%s]\n", label);
  PLEGMA_printf("  |ref|       = %.16e\n", m.norm_ref);
  PLEGMA_printf("  |test|      = %.16e\n", m.norm_test);
  PLEGMA_printf("  |diff|      = %.16e\n", m.norm_diff);
  PLEGMA_printf("  rel diff    = %.16e\n", m.rel_diff);
}

// ---------------------------------------------------------------------------
// Build a 12-column point-source propagator with optional Gaussian smearing.
// Solver and smearing gauge must be supplied.  Output is float.
// ---------------------------------------------------------------------------
static void build_prop(PLEGMA_Propagator<float> &prop,
                       QUDA_solver              &solver,
                       const site               &src_site,
                       PLEGMA_Gauge<double>     &smearedGauge,
                       bool                      do_smear,
                       bool                      rotateQ,
                       double                    mass_sign)
{
  const int Nt   = HGC_totalL[3];
  const int src_t = src_site[DIM_T];

  PLEGMA_Gauge3D<double> smearedGauge3D;
  if (do_smear)
    smearedGauge3D.absorb(smearedGauge, src_t);

  for (int isc = 0; isc < 12; ++isc) {
    const int spin = isc / 3, col = isc % 3;

    PLEGMA_Vector<double> rhs, sol;

    if (do_smear) {
      PLEGMA_Vector3D<double> s3D_raw, s3D_sm;
      s3D_raw.pointSource(src_site, spin, col, DEVICE);
      TIME(s3D_sm.gaussianSmearing(s3D_raw, smearedGauge3D,
                                   nsmearGauss, alphaGauss));
      rhs.absorb(s3D_sm, src_t);
    } else {
      PLEGMA_Vector3D<double> s3D;
      s3D.pointSource(src_site, spin, col, DEVICE);
      rhs.absorb(s3D, src_t);
    }

    PLEGMA_Vector<double> rhs2;
    if (rotateQ) {
      rhs2.rotateToPhysicalBasis(rhs, mass_sign);
    } else {
      rhs2.copy(rhs);
    }

    TIME(solver.solve(rhs2, rhs2));

    PLEGMA_Vector<double> sol2;
    if (rotateQ) {
      sol2.rotateToPhysicalBasis(rhs2, mass_sign);
    } else {
      sol2.copy(rhs2);
    }

    PLEGMA_Vector<float> sol_f;
    sol_f.copy(sol2);
    prop.absorb(sol_f, spin, col);
  }
}

// ---------------------------------------------------------------------------
// Flow all 12 columns with float spinors, assemble result into prop_out.
// One batch call = one Wilson flow run.
// ---------------------------------------------------------------------------
static void flow_prop_float(PLEGMA_Propagator<float>  &prop_out,
                            PLEGMA_Propagator<double> &prop_in,
                            PLEGMA_Gauge<float>       &gauge_out,
                            PLEGMA_Gauge<double>      &gauge_in,
                            int n_steps, double epsilon, double t0_start,
                            int rhs_block)
{
  // Allocate 12 column vectors (downcast double input → float spinors)
  std::vector<std::unique_ptr<PLEGMA_Vector<float>>> in_own(12), out_own(12);
  std::vector<PLEGMA_Vector<float>*> in_ptrs(12), out_ptrs(12);

  for (int isc = 0; isc < 12; ++isc) {
    in_own[isc]  = std::make_unique<PLEGMA_Vector<float>>(DEVICE);
    out_own[isc] = std::make_unique<PLEGMA_Vector<float>>(DEVICE);
    PLEGMA_Vector<double> tmp_d;
    tmp_d.absorb(prop_in, isc / 3, isc % 3);
    in_own[isc]->copy(tmp_d);  // double → float downcast
    in_ptrs[isc]  = in_own[isc].get();
    out_ptrs[isc] = out_own[isc].get();
  }

  TIME(fermionFlow_PLEGMA(out_ptrs, in_ptrs, gauge_out, gauge_in,
                          n_steps, epsilon, t0_start,
                          false, false, true, true, rhs_block));

  for (int isc = 0; isc < 12; ++isc)
    prop_out.absorb(*out_own[isc], isc / 3, isc % 3);
}

// ---------------------------------------------------------------------------
// Flow all 12 columns with double spinors, assemble result into prop_out (double).
// One batch call = one Wilson flow run.
// ---------------------------------------------------------------------------
static void flow_prop_double(PLEGMA_Propagator<double> &prop_out,
                             PLEGMA_Propagator<double> &prop_in,
                             PLEGMA_Gauge<double>      &gauge_out,
                             PLEGMA_Gauge<double>      &gauge_in,
                             int n_steps, double epsilon, double t0_start,
                             int rhs_block)
{
  std::vector<std::unique_ptr<PLEGMA_Vector<double>>> in_own(12), out_own(12);
  std::vector<PLEGMA_Vector<double>*> in_ptrs(12), out_ptrs(12);

  for (int isc = 0; isc < 12; ++isc) {
    in_own[isc]  = std::make_unique<PLEGMA_Vector<double>>(DEVICE);
    out_own[isc] = std::make_unique<PLEGMA_Vector<double>>(DEVICE);
    in_own[isc]->absorb(prop_in, isc / 3, isc % 3);  // direct double copy
    in_ptrs[isc]  = in_own[isc].get();
    out_ptrs[isc] = out_own[isc].get();
  }

  TIME(fermionFlow_PLEGMA(out_ptrs, in_ptrs, gauge_out, gauge_in,
                          n_steps, epsilon, t0_start,
                          false, false, true, true, rhs_block));

  // Store double result directly into double propagator
  for (int isc = 0; isc < 12; ++isc)
    prop_out.absorb(*out_own[isc], isc / 3, isc % 3);
}

// ---------------------------------------------------------------------------
// GPU memory reporter (no nvidia-smi needed on compute nodes)
// ---------------------------------------------------------------------------
static void print_gpu_mem(const char *label) {
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  PLEGMA_printf("[GPU mem] %-48s  free=%6.2f GB  used=%6.2f GB  total=%6.2f GB\n",
                label,
                free_b          / 1.073741824e9,
                (total_b - free_b) / 1.073741824e9,
                total_b         / 1.073741824e9);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  std::vector<std::string> listOpt = {
      "verbosity",
      "load-gauge",
      "nsmear-APE",
      "alpha-APE",
      "nsmear-gauss",
      "alpha-gauss",
      "nsrc",
      "src-filename",
      "twop-filename",
      "threep-filename",
      "corr-file-format",
      "corr-space",
      "tSinks",
      "maxQsq",
      "max-order"
  };
  initializeOptions(argc, argv, true, listOpt);

  // --- Flow parameters (command-line overridable) ---
  int    flow_steps   = 500;
  double flow_epsilon = 0.01;
  double flow_t0      = 0.0;
  int    rhs_block    = 6;   // max simultaneous RHS in batch Laplace
  int    max_order    = 6;   // max derivative order for fiveD patterns (1-6); lower saves GPU memory

  HGC_options->set("flow-steps",   "Number of Wilson flow steps", verbosity, flow_steps);
  HGC_options->set("flow-epsilon", "Wilson flow step size",       verbosity, flow_epsilon);
  HGC_options->set("flow-t0",      "Initial flow time",           verbosity, flow_t0);
  HGC_options->set("rhs-block",    "Max simultaneous RHS in Laplace batch", verbosity, rhs_block);
  HGC_options->set("max-order",    "Max derivative order for fiveD patterns (1-6)", verbosity, max_order);

  // --- env-var overridable behaviour flags ---
  bool rotateQ = true;
  bool piplusQ = true;
  auto read_bool_env = [](const char *name, bool def_val) {
    const char *v = std::getenv(name);
    if (!v) return def_val;
    std::string s(v);
    if (s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES") return true;
    if (s == "0" || s == "false" || s == "FALSE" || s == "no" || s == "NO") return false;
    return def_val;
  };
  rotateQ = read_bool_env("ROTATEQ", rotateQ);
  piplusQ = read_bool_env("PIPLUSQ", piplusQ);

  initializePLEGMA();
  {
    // ------------------------------------------------------------------
    // Gauge setup
    // ------------------------------------------------------------------
    PLEGMA_Gauge<double> gauge(BOTH);
    TIME(gauge.readFile(latfile, LIME_FORMAT));
    PLEGMA_printf("Plaquette (unsmeared):\n");
    gauge.calculatePlaq();

    // Anti-periodic BC baked into QUDA resident gauge (gaugePrecise).
    initGaugeQuda(gauge, true);
    plaqQuda();

    // APE-smearing for optional Gaussian source smearing.
    const bool do_smear = (nsmearGauss > 0);
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    if (do_smear) {
      TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
      PLEGMA_printf("Plaquette after APE-smearing:\n");
      smearedGauge.calculatePlaq();
    }

    // ------------------------------------------------------------------
    // Solver
    // ------------------------------------------------------------------
    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu, 1));

    const double msign    = mu / std::abs(mu);
    const site   src      = sourcePositions[0];

    PLEGMA_printf("\n=== Source at (%d,%d,%d,%d) ===\n",
                  src[0], src[1], src[2], src[3]);

    // ------------------------------------------------------------------
    // Build forward propagator: SL (smeared-local) and SS (smeared-smeared)
    // THIRD_SIDE ghost pre-allocates buffers needed by contractNucleonThrp_twoD
    // ------------------------------------------------------------------
    PLEGMA_Propagator<double> prop_fwd(BOTH, plegma::THIRD_SIDE);
    PLEGMA_Propagator<double> prop_fwd_ss(BOTH, plegma::THIRD_SIDE);
    PLEGMA_printf("\n--- Building forward propagator (SL + SS, double precision) ---\n");

    {
      PLEGMA_Gauge3D<double> smearedGauge3D_src;
      if (do_smear) smearedGauge3D_src.absorb(smearedGauge, src[DIM_T]);

      for (int isc = 0; isc < 12; ++isc) {
        const int spin = isc / 3, col = isc % 3;

        PLEGMA_Vector<double> rhs, sol;
        if (do_smear) {
          PLEGMA_Vector3D<double> s3D_raw, s3D_sm;
          s3D_raw.pointSource(src, spin, col, DEVICE);
          TIME(s3D_sm.gaussianSmearing(s3D_raw, smearedGauge3D_src, nsmearGauss, alphaGauss));
          rhs.absorb(s3D_sm, src[DIM_T]);
        } else {
          PLEGMA_Vector3D<double> s3D;
          s3D.pointSource(src, spin, col, DEVICE);
          rhs.absorb(s3D, src[DIM_T]);
        }

        PLEGMA_Vector<double> rhs2;
        if (rotateQ) rhs2.rotateToPhysicalBasis(rhs, msign);
        else         rhs2.copy(rhs);

        TIME(solver.solve(rhs2, rhs2));

        PLEGMA_Vector<double> sol2;
        if (rotateQ) sol2.rotateToPhysicalBasis(rhs2, msign);
        else         sol2.copy(rhs2);

        // SL: store full double precision
        prop_fwd.absorb(sol2, spin, col);

        // SS: apply sink Gaussian smearing (remain in double)
        PLEGMA_Vector<double> sol_sm;
        TIME(sol_sm.gaussianSmearing(sol2, smearedGauge, nsmearGauss, alphaGauss));
        prop_fwd_ss.absorb(sol_sm, spin, col);
      }
    }

    // Per-source output filenames (tag last-4 chars of latfile as conf tag)
    const std::string conf_tag  = latfile.substr(latfile.length() - 4);
    char *src_tag = nullptr;
    asprintf(&src_tag, "_sx%02dsy%02dsz%02dst%03d",
             src[0], src[1], src[2], src[3]);
    const std::string twop_out  = twop_filename  + src_tag + "_chi." + conf_tag;
    const std::string threep_out = threep_filename + src_tag + "_chi." + conf_tag;

    // ------------------------------------------------------------------
    // (B) 2pt: unflowed (Wilson flow not applied to 2pt)
    // ------------------------------------------------------------------
    // double precision
    {
      PLEGMA_Correlator<double> corr_local(corr_space, src, maxQsq);
      TIME(corr_local.contractMesonsNew(prop_fwd, prop_fwd));
      TIME(corr_local.writeFile(twop_out + "_double.h5", HDF5_FORMAT));
    }
    // float precision
    {
      PLEGMA_Propagator<float> prop_fwd_f(BOTH, plegma::THIRD_SIDE);
      prop_fwd_f.copy(prop_fwd);
      PLEGMA_Correlator<float> corr_local(corr_space, src, maxQsq);
      TIME(corr_local.contractMesonsNew(prop_fwd_f, prop_fwd_f));
      TIME(corr_local.writeFile(twop_out + "_float.h5", HDF5_FORMAT));
    }

    // ------------------------------------------------------------------
    // (C) 3pt: sequential propagator loop over tSinks
    //
    // For each tsink:
    //   - build seq prop (invert with -mu)
    //   - flow both fwd+seq with float spinors AND with double spinors
    //   - 3pt contractions at each flow checkpoint: local, oneD, twoD, fiveD
    // ------------------------------------------------------------------
    GAMMAS Gsrc = G5;
    GAMMAS Gsnk = G5;

    // Flip mu sign for seq prop inversion (pi+ convention: seq uses -mu)
    const double mu_seq = piplusQ ? -std::abs(mu) : +std::abs(mu);
    solver.UpdateSolver();

    // Pre-extract 3D slices for ALL tsinks before main loop allocations
    std::vector<std::unique_ptr<PLEGMA_Propagator3D<double>>> prop_3D_slices;
    for (size_t i = 0; i < tSinks.size(); ++i) {
      const int st = (src[3] + tSinks[i]) % HGC_totalL[3];
      auto p = std::make_unique<PLEGMA_Propagator3D<double>>();
      p->absorb(prop_fwd_ss, st);
      prop_3D_slices.push_back(std::move(p));
    }

    // Save forward SL prop to HOST for reload inside tsink loop (double precision)
    prop_fwd.unload();
    PLEGMA_Propagator<double> prop_fwd_host(HOST);
    prop_fwd_host.copy(prop_fwd, HOST);

    // Update solver for seq mass
    {
      updateOptions(LIGHT);
      mu = mu_seq;
      solver.UpdateSolver();
    }

    for (size_t its = 0; its < tSinks.size(); ++its) {
      const int dt_sink = tSinks[its];
      if (dt_sink >= HGC_totalL[3])
        PLEGMA_error("Provided tsink=%d >= temporal extent", dt_sink);

      const int sink_t = (src[3] + dt_sink) % HGC_totalL[3];

      PLEGMA_Gauge3D<double> smearedGauge3D_sink;
      smearedGauge3D_sink.absorb(smearedGauge, sink_t);

      // 3D slice at sink time from pre-extracted array
      PLEGMA_Propagator3D<double> &prop_q_3D = *prop_3D_slices[its];
      prop_q_3D.apply_gamma(Gsnk, LEFT);

      // Build sequential propagator (double precision, matching forward prop)
      PLEGMA_Propagator<double> seq_prop(BOTH, plegma::THIRD_SIDE);
      {
        PLEGMA_Propagator<double> seqPropTmp(BOTH);
        for (int nu = 0; nu < 4; ++nu) {
          for (int c2 = 0; c2 < 3; ++c2) {
            PLEGMA_Vector3D<double> seq3D_d, seq3D_sm;
            seq3D_d.absorb(prop_q_3D, nu, c2);
            TIME(seq3D_sm.gaussianSmearing(seq3D_d, smearedGauge3D_sink,
                                           nsmearGauss, alphaGauss));

            PLEGMA_Vector<double> rhs4D1, rhs4D2;
            rhs4D2.absorb(seq3D_sm, sink_t);

            const double nrm = rhs4D2.norm();
            if (nrm > 0.0) rhs4D2.scale(1.0 / nrm);
            if (rotateQ) rhs4D1.rotateToPhysicalBasis(rhs4D2, mu / std::abs(mu));
            else         rhs4D1.copy(rhs4D2);

            TIME(solver.solve(rhs4D1, rhs4D1));

            if (rotateQ) rhs4D2.rotateToPhysicalBasis(rhs4D1, mu / std::abs(mu));
            else         rhs4D2.copy(rhs4D1);
            if (nrm > 0.0) rhs4D2.scale(nrm);

            seqPropTmp.absorb(rhs4D2, nu, c2);
          }
        }
        seq_prop.copy(seqPropTmp);
      }

      // --- Double-spinor flow first (higher memory pressure → fail fast) ---
      print_gpu_mem("before double-spinor 3pt block");
      PLEGMA_printf("\n--- 3pt dt=%d: double-spinor flow ---\n", dt_sink);
      {
        PLEGMA_Propagator<double> fwd_in(BOTH, plegma::THIRD_SIDE),
                                   fwd_out(BOTH, plegma::THIRD_SIDE);
        PLEGMA_Propagator<double> seq_in(BOTH, plegma::THIRD_SIDE),
                                   seq_out(BOTH, plegma::THIRD_SIDE);
        PLEGMA_Gauge<double> gf_d(BOTH, plegma::THIRD_SIDE);

        fwd_in.copy(prop_fwd_host, HOST);
        fwd_in.load();
        seq_in.copy(seq_prop);  // double seq_prop → double seq_in directly

        PLEGMA_Propagator<double> *pFwd_in  = &fwd_in,  *pFwd_out = &fwd_out;
        PLEGMA_Propagator<double> *pSeq_in  = &seq_in,  *pSeq_out = &seq_out;

        PLEGMA_Correlator<double> corr3(corr_space, src, maxQsq);
        PLEGMA_Correlator<double> corr3_oneD(corr_space, src, maxQsq);
        PLEGMA_Correlator<double> corr3_twoD(corr_space, src, maxQsq);

        PLEGMA_Gauge<double> contractGauge(BOTH, plegma::THIRD_SIDE);
        contractGauge.copy(gauge);
        applyBoundaryConditions(contractGauge, true);
        contractGauge.communicateSecondSideGhost();

        const double eps_3pt = flow_epsilon;
        const int n_flow_save = 10;  // must match float block
        int step = 0;
        double t = flow_t0;

        // nt=0
        {
          const std::string base = threep_out + "_dt" + std::to_string(dt_sink) + "_nt00_double";
          pFwd_in->conjugate();
          pFwd_in->apply_gamma(G5, LEFT);
          pFwd_in->apply_gamma(G5, RIGHT);
          pSeq_in->apply_gamma(Gsrc, RIGHT);

          TIME(corr3.contractNucleonThrp_local(*pFwd_in, *pSeq_in, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_oneD.contractNucleonThrp_oneD(*pFwd_in, *pSeq_in, contractGauge, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_twoD.contractNucleonThrp_twoD(*pFwd_in, *pSeq_in, contractGauge, 0, {G1,G2,G3,G4,G5G4}));
          TIME(contractNucleonThrp_fiveD_patterns(*pFwd_in, *pSeq_in, contractGauge,
              corr_space, src, maxQsq, base + "_patterns", max_order));

          pSeq_in->apply_gamma(Gsrc, RIGHT);
          pFwd_in->apply_gamma(G5, RIGHT);
          pFwd_in->apply_gamma(G5, LEFT);
          pFwd_in->conjugate();

          TIME(corr3.writeFile(base + "_local", corr_file_format));
          TIME(corr3_oneD.writeFile(base + "_oneD", corr_file_format));
          TIME(corr3_twoD.writeFile(base + "_twoD", corr_file_format));
        }

        // flow loop: double-spinor batch flow, keep double precision
        updateGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
        for (int ic = 1; ic <= flow_steps; ic += (flow_steps / n_flow_save > 0 ? flow_steps / n_flow_save : 1)) {
          const int target_step = std::min(ic, flow_steps);
          const int chunk = target_step - step;
          if (chunk <= 0) continue;

          if (ic == 1) print_gpu_mem("double flow loop iter 1 (peak alloc)");
          // Flow fwd prop (double spinors, double output)
          {
            std::vector<std::unique_ptr<PLEGMA_Vector<double>>> in_own(12), out_own(12);
            std::vector<PLEGMA_Vector<double>*> in_ptrs(12), out_ptrs(12);
            for (int isc = 0; isc < 12; ++isc) {
              in_own[isc]  = std::make_unique<PLEGMA_Vector<double>>(DEVICE);
              out_own[isc] = std::make_unique<PLEGMA_Vector<double>>(DEVICE);
              in_own[isc]->absorb(*pFwd_in, isc/3, isc%3);
              in_ptrs[isc]  = in_own[isc].get();
              out_ptrs[isc] = out_own[isc].get();
            }
            PLEGMA_Gauge<double> gf_d_fwd(BOTH);
            TIME(fermionFlow_PLEGMA(out_ptrs, in_ptrs, gf_d_fwd, gauge,
                                    chunk, eps_3pt, t,
                                    false, false, false, true, rhs_block));
            for (int isc = 0; isc < 12; ++isc)
              pFwd_out->absorb(*out_own[isc], isc/3, isc%3);
          }
          // Flow seq prop (double spinors, double output)
          {
            std::vector<std::unique_ptr<PLEGMA_Vector<double>>> in_own(12), out_own(12);
            std::vector<PLEGMA_Vector<double>*> in_ptrs(12), out_ptrs(12);
            for (int isc = 0; isc < 12; ++isc) {
              in_own[isc]  = std::make_unique<PLEGMA_Vector<double>>(DEVICE);
              out_own[isc] = std::make_unique<PLEGMA_Vector<double>>(DEVICE);
              in_own[isc]->absorb(*pSeq_in, isc/3, isc%3);
              in_ptrs[isc]  = in_own[isc].get();
              out_ptrs[isc] = out_own[isc].get();
            }
            updateGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
            TIME(fermionFlow_PLEGMA(out_ptrs, in_ptrs, gf_d, gauge,
                                    chunk, eps_3pt, t,
                                    false, false, true, true, rhs_block));
            for (int isc = 0; isc < 12; ++isc)
              pSeq_out->absorb(*out_own[isc], isc/3, isc%3);
          }
          std::swap(pFwd_in, pFwd_out);
          std::swap(pSeq_in, pSeq_out);
          step = target_step;
          t    = step * eps_3pt;

          contractGauge.copy(gf_d);

          char nt_tag[32];
          snprintf(nt_tag, sizeof(nt_tag), "_nt%02d_double", ic);
          const std::string base = threep_out + "_dt" + std::to_string(dt_sink) + nt_tag;

          pFwd_in->conjugate();
          pFwd_in->apply_gamma(G5, LEFT);
          pFwd_in->apply_gamma(G5, RIGHT);
          pSeq_in->apply_gamma(Gsrc, RIGHT);

          TIME(corr3.contractNucleonThrp_local(*pFwd_in, *pSeq_in, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_oneD.contractNucleonThrp_oneD(*pFwd_in, *pSeq_in, contractGauge, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_twoD.contractNucleonThrp_twoD(*pFwd_in, *pSeq_in, contractGauge, 0, {G1,G2,G3,G4,G5G4}));
          TIME(contractNucleonThrp_fiveD_patterns(*pFwd_in, *pSeq_in, contractGauge,
              corr_space, src, maxQsq, base + "_patterns", max_order));

          pSeq_in->apply_gamma(Gsrc, RIGHT);
          pFwd_in->apply_gamma(G5, RIGHT);
          pFwd_in->apply_gamma(G5, LEFT);
          pFwd_in->conjugate();

          TIME(corr3.writeFile(base + "_local", corr_file_format));
          TIME(corr3_oneD.writeFile(base + "_oneD", corr_file_format));
          TIME(corr3_twoD.writeFile(base + "_twoD", corr_file_format));

          if (step >= flow_steps) break;
        }
      }

      // --- Float-spinor flow: fwd + seq simultaneously ---
      print_gpu_mem("before float-spinor 3pt block");
      PLEGMA_printf("\n--- 3pt dt=%d: float-spinor flow ---\n", dt_sink);
      {
        PLEGMA_Propagator<float> fwd_in(BOTH, plegma::THIRD_SIDE),
                                  fwd_out(BOTH, plegma::THIRD_SIDE);
        PLEGMA_Propagator<float> seq_in(BOTH, plegma::THIRD_SIDE),
                                  seq_out(BOTH, plegma::THIRD_SIDE);
        PLEGMA_Gauge<float> gf(BOTH, plegma::THIRD_SIDE);

        // Reload fwd prop: downcast double host → float for float-spinor flow
        {
          PLEGMA_Propagator<double> fwd_d_tmp(BOTH, plegma::THIRD_SIDE);
          fwd_d_tmp.copy(prop_fwd_host, HOST);
          fwd_d_tmp.load();
          fwd_in.copy(fwd_d_tmp);
        }
        // Downcast double seq_prop → float for float-spinor flow
        seq_in.copy(seq_prop);

        PLEGMA_Propagator<float> *pFwd_in  = &fwd_in,  *pFwd_out = &fwd_out;
        PLEGMA_Propagator<float> *pSeq_in  = &seq_in,  *pSeq_out = &seq_out;

        PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
        PLEGMA_Correlator<float> corr3_oneD(corr_space, src, maxQsq);
        PLEGMA_Correlator<float> corr3_twoD(corr_space, src, maxQsq);

        // contractGauge for derivative operators
        PLEGMA_Gauge<float> contractGauge(BOTH, plegma::THIRD_SIDE);
        contractGauge.copy(gauge);
        applyBoundaryConditions(contractGauge, true);
        contractGauge.communicateSecondSideGhost();

        const double eps_3pt  = flow_epsilon;
        const double t0_3pt   = flow_t0;
        const double t_target = flow_steps * eps_3pt;
        const int n_flow_save = 10;  // 10 checkpoints evenly spaced
        int step = 0;
        double t = t0_3pt;

        // nt=0 contraction
        {
          const std::string base = threep_out + "_dt" + std::to_string(dt_sink) + "_nt00_float";
          pFwd_in->conjugate();
          pFwd_in->apply_gamma(G5, LEFT);
          pFwd_in->apply_gamma(G5, RIGHT);
          pSeq_in->apply_gamma(Gsrc, RIGHT);

          TIME(corr3.contractNucleonThrp_local(*pFwd_in, *pSeq_in, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_oneD.contractNucleonThrp_oneD(*pFwd_in, *pSeq_in, contractGauge, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_twoD.contractNucleonThrp_twoD(*pFwd_in, *pSeq_in, contractGauge, 0, {G1,G2,G3,G4,G5G4}));
          TIME(contractNucleonThrp_fiveD_patterns(*pFwd_in, *pSeq_in, contractGauge,
              corr_space, src, maxQsq, base + "_patterns", max_order));

          pSeq_in->apply_gamma(Gsrc, RIGHT);
          pFwd_in->apply_gamma(G5, RIGHT);
          pFwd_in->apply_gamma(G5, LEFT);
          pFwd_in->conjugate();

          TIME(corr3.writeFile(base + "_local", corr_file_format));
          TIME(corr3_oneD.writeFile(base + "_oneD", corr_file_format));
          TIME(corr3_twoD.writeFile(base + "_twoD", corr_file_format));
        }

        // flow loop with checkpoints
        updateGaugeQuda(gauge, true, QUDA_WILSON_LINKS);
        for (int ic = 1; ic <= flow_steps; ic += (flow_steps / n_flow_save > 0 ? flow_steps / n_flow_save : 1)) {
          const int target_step = std::min(ic, flow_steps);
          const int chunk = target_step - step;
          if (chunk <= 0) continue;

          TIME(fermionFlow_PLEGMA(*pFwd_out, *pFwd_in, *pSeq_out, *pSeq_in, gf,
                                  gauge, chunk, eps_3pt, t,
                                  true, false, true, true, rhs_block));
          std::swap(pFwd_in, pFwd_out);
          std::swap(pSeq_in, pSeq_out);
          step = target_step;
          t    = step * eps_3pt;

          contractGauge.copy(gf);

          char nt_tag[32];
          snprintf(nt_tag, sizeof(nt_tag), "_nt%02d_float", ic);
          const std::string base = threep_out + "_dt" + std::to_string(dt_sink) + nt_tag;

          pFwd_in->conjugate();
          pFwd_in->apply_gamma(G5, LEFT);
          pFwd_in->apply_gamma(G5, RIGHT);
          pSeq_in->apply_gamma(Gsrc, RIGHT);

          TIME(corr3.contractNucleonThrp_local(*pFwd_in, *pSeq_in, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_oneD.contractNucleonThrp_oneD(*pFwd_in, *pSeq_in, contractGauge, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_twoD.contractNucleonThrp_twoD(*pFwd_in, *pSeq_in, contractGauge, 0, {G1,G2,G3,G4,G5G4}));
          TIME(contractNucleonThrp_fiveD_patterns(*pFwd_in, *pSeq_in, contractGauge,
              corr_space, src, maxQsq, base + "_patterns", max_order));

          pSeq_in->apply_gamma(Gsrc, RIGHT);
          pFwd_in->apply_gamma(G5, RIGHT);
          pFwd_in->apply_gamma(G5, LEFT);
          pFwd_in->conjugate();

          TIME(corr3.writeFile(base + "_local", corr_file_format));
          TIME(corr3_oneD.writeFile(base + "_oneD", corr_file_format));
          TIME(corr3_twoD.writeFile(base + "_twoD", corr_file_format));

          if (step >= flow_steps) break;
        }
      }

    } // tsink loop

    free(src_tag);

    PLEGMA_printf("\n=== Summary ===\n");
    PLEGMA_printf("Flow parameters: n_steps=%d  epsilon=%.4f  t0=%.4f\n",
                  flow_steps, flow_epsilon, flow_t0);
    PLEGMA_printf("Output files:\n");
    PLEGMA_printf("  2pt (unflowed): {twop_out}_{float,double}.h5\n");
    PLEGMA_printf("  3pt: {threep_out}_dt*_nt*_{float,double}_{local,oneD,twoD,patterns}.h5\n");

    finalizeGaugeQuda();
  }

  while (!threads.empty()) {
    threads.back().join();
    threads.pop_back();
  }
  finalize();
  return 0;
}
