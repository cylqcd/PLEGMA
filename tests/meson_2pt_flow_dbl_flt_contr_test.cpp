// tests/meson_2pt_flow_dbl_flt_contr_test.cpp
//
// Precision variant: gauge and propagators are kept in double throughout;
// Wilson flow uses double-precision spinors and double-precision gauge as
// both input and output.  At contraction time correlators are computed
// with BOTH float and double accumulators so that the two can be compared.
// Output file names carry the contraction precision as the last suffix:
//   ..._nt<N>_double_flow_<type>_float_contr  and  ..._nt<N>_double_flow_<type>_double_contr
//
// This isolates contraction precision from flow-storage precision: the flow
// trajectory is identical to the pure-double path of
// meson_2pt_flow_precision_test.
//
// Usage (same flags as meson_2pt_flow_precision_test):
//   mpirun -n 1 ./meson_2pt_flow_dbl_flt_contr --load-gauge <file> \
//          --flow-steps 500 --flow-epsilon 0.01 \
//          [--src-filename <file>]
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
// GPU memory reporter
// ---------------------------------------------------------------------------
static void print_gpu_mem(const char *label) {
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  PLEGMA_printf("[GPU mem] %-48s  free=%6.2f GB  used=%6.2f GB  total=%6.2f GB\n",
                label,
                free_b             / 1.073741824e9,
                (total_b - free_b) / 1.073741824e9,
                total_b            / 1.073741824e9);
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
  int    rhs_block    = 6;
  int    max_order    = 6;

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
    // Gauge setup  (double throughout)
    // ------------------------------------------------------------------
    PLEGMA_Gauge<double> gauge(BOTH);
    TIME(gauge.readFile(latfile, LIME_FORMAT));
    PLEGMA_printf("Plaquette (unsmeared):\n");
    gauge.calculatePlaq();

    initGaugeQuda(gauge, true);
    plaqQuda();

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

    const double msign = mu / std::abs(mu);
    const site   src   = sourcePositions[0];

    PLEGMA_printf("\n=== Source at (%d,%d,%d,%d) ===\n",
                  src[0], src[1], src[2], src[3]);

    // ------------------------------------------------------------------
    // Build forward propagator (double precision, SL + SS)
    // ------------------------------------------------------------------
    PLEGMA_Propagator<double> prop_fwd(BOTH, plegma::THIRD_SIDE);
    PLEGMA_Propagator<double> prop_fwd_ss(BOTH, plegma::THIRD_SIDE);
    PLEGMA_printf("\n--- Building forward propagator (SL + SS, double) ---\n");

    {
      PLEGMA_Gauge3D<double> smearedGauge3D_src;
      if (do_smear) smearedGauge3D_src.absorb(smearedGauge, src[DIM_T]);

      for (int isc = 0; isc < 12; ++isc) {
        const int spin = isc / 3, col = isc % 3;

        PLEGMA_Vector<double> rhs;
        if (do_smear) {
          PLEGMA_Vector3D<double> s3D_raw, s3D_sm;
          s3D_raw.pointSource(src, spin, col, DEVICE);
          TIME(s3D_sm.gaussianSmearing(s3D_raw, smearedGauge3D_src,
                                       nsmearGauss, alphaGauss));
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

        prop_fwd.absorb(sol2, spin, col);

        // SS: sink Gaussian smearing
        PLEGMA_Vector<double> sol_sm;
        TIME(sol_sm.gaussianSmearing(sol2, smearedGauge, nsmearGauss, alphaGauss));
        prop_fwd_ss.absorb(sol_sm, spin, col);
      }
    }

    // Per-source output filenames
    const std::string conf_tag = latfile.substr(latfile.length() - 4);
    char *src_tag = nullptr;
    asprintf(&src_tag, "_sx%02dsy%02dsz%02dst%03d",
             src[0], src[1], src[2], src[3]);
    const std::string twop_out  = twop_filename  + src_tag + "_chi." + conf_tag;
    const std::string threep_out = threep_filename + src_tag + "_chi." + conf_tag;

    // ------------------------------------------------------------------
    // 2pt: unflowed  (double flow path; float and double contraction)
    // ------------------------------------------------------------------
    {
      // float contraction
      PLEGMA_Propagator<float> prop_fwd_f(BOTH, plegma::THIRD_SIDE);
      prop_fwd_f.copy(prop_fwd);
      PLEGMA_Correlator<float> corr_local_f(corr_space, src, maxQsq);
      TIME(corr_local_f.contractMesonsNew(prop_fwd_f, prop_fwd_f));
      TIME(corr_local_f.writeFile(twop_out + "_double_flow_float_contr.h5", HDF5_FORMAT));
      // double contraction
      PLEGMA_Correlator<double> corr_local_d(corr_space, src, maxQsq);
      TIME(corr_local_d.contractMesonsNew(prop_fwd, prop_fwd));
      TIME(corr_local_d.writeFile(twop_out + "_double_flow_double_contr.h5", HDF5_FORMAT));
    }

    // ------------------------------------------------------------------
    // 3pt: sequential propagator loop over tSinks
    //
    // Gauge and propagators remain double throughout.
    // Only at contraction: downcast to float, use PLEGMA_Correlator<float>.
    // ------------------------------------------------------------------
    GAMMAS Gsrc = G5;
    GAMMAS Gsnk = G5;

    const double mu_seq = piplusQ ? -std::abs(mu) : +std::abs(mu);
    solver.UpdateSolver();

    // Pre-extract 3D slices for all tsinks
    std::vector<std::unique_ptr<PLEGMA_Propagator3D<double>>> prop_3D_slices;
    for (size_t i = 0; i < tSinks.size(); ++i) {
      const int st = (src[3] + tSinks[i]) % HGC_totalL[3];
      auto p = std::make_unique<PLEGMA_Propagator3D<double>>();
      p->absorb(prop_fwd_ss, st);
      prop_3D_slices.push_back(std::move(p));
    }

    // Save forward SL prop to HOST for reload inside tsink loop
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

      PLEGMA_Propagator3D<double> &prop_q_3D = *prop_3D_slices[its];
      prop_q_3D.apply_gamma(Gsnk, LEFT);

      // Build sequential propagator (double precision)
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

      // ----------------------------------------------------------------
      // Double-precision flow; float-precision contraction
      // ----------------------------------------------------------------
      print_gpu_mem("before double-spinor 3pt block");
      PLEGMA_printf("\n--- 3pt dt=%d: double flow / float contraction ---\n", dt_sink);
      {
        PLEGMA_Propagator<double> fwd_in(BOTH, plegma::THIRD_SIDE),
                                   fwd_out(BOTH, plegma::THIRD_SIDE);
        PLEGMA_Propagator<double> seq_in(BOTH, plegma::THIRD_SIDE),
                                   seq_out(BOTH, plegma::THIRD_SIDE);
        PLEGMA_Gauge<double> gf_d(BOTH, plegma::THIRD_SIDE);

        fwd_in.copy(prop_fwd_host, HOST);
        fwd_in.load();
        seq_in.copy(seq_prop);

        PLEGMA_Propagator<double> *pFwd_in  = &fwd_in,  *pFwd_out = &fwd_out;
        PLEGMA_Propagator<double> *pSeq_in  = &seq_in,  *pSeq_out = &seq_out;

        // Correlators — declared once, reused per checkpoint
        PLEGMA_Correlator<float>  corr3_f(corr_space, src, maxQsq);
        PLEGMA_Correlator<float>  corr3_oneD_f(corr_space, src, maxQsq);
        PLEGMA_Correlator<float>  corr3_twoD_f(corr_space, src, maxQsq);
        PLEGMA_Correlator<double> corr3_d(corr_space, src, maxQsq);
        PLEGMA_Correlator<double> corr3_oneD_d(corr_space, src, maxQsq);
        PLEGMA_Correlator<double> corr3_twoD_d(corr_space, src, maxQsq);

        // nt=0 contraction -------------------------------------------
        {
          const std::string base = threep_out + "_dt" + std::to_string(dt_sink) + "_nt00_double_flow";

          // --- float contraction ---
          PLEGMA_Propagator<float> fwd_f(BOTH, plegma::THIRD_SIDE);
          PLEGMA_Propagator<float> seq_f(BOTH, plegma::THIRD_SIDE);
          fwd_f.copy(*pFwd_in);
          seq_f.copy(*pSeq_in);
          fwd_f.conjugate();
          fwd_f.apply_gamma(G5, LEFT);
          fwd_f.apply_gamma(G5, RIGHT);
          seq_f.apply_gamma(Gsrc, RIGHT);

          PLEGMA_Gauge<float> contractGauge_f(BOTH, plegma::THIRD_SIDE);
          contractGauge_f.copy(gauge);
          applyBoundaryConditions(contractGauge_f, true);
          contractGauge_f.communicateSecondSideGhost();

          TIME(corr3_f.contractNucleonThrp_local(fwd_f, seq_f, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_oneD_f.contractNucleonThrp_oneD(fwd_f, seq_f, contractGauge_f, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_twoD_f.contractNucleonThrp_twoD(fwd_f, seq_f, contractGauge_f, 0, {G1,G2,G3,G4,G5G4}));
          TIME(contractNucleonThrp_fiveD_patterns(fwd_f, seq_f, contractGauge_f,
              corr_space, src, maxQsq, base + "_patterns_float_contr", max_order));
          TIME(corr3_f.writeFile(base + "_local_float_contr", corr_file_format));
          TIME(corr3_oneD_f.writeFile(base + "_oneD_float_contr", corr_file_format));
          TIME(corr3_twoD_f.writeFile(base + "_twoD_float_contr", corr_file_format));

          // --- double contraction ---
          PLEGMA_Propagator<double> fwd_d(BOTH, plegma::THIRD_SIDE);
          PLEGMA_Propagator<double> seq_d(BOTH, plegma::THIRD_SIDE);
          fwd_d.copy(*pFwd_in);
          seq_d.copy(*pSeq_in);
          fwd_d.conjugate();
          fwd_d.apply_gamma(G5, LEFT);
          fwd_d.apply_gamma(G5, RIGHT);
          seq_d.apply_gamma(Gsrc, RIGHT);

          PLEGMA_Gauge<double> contractGauge_d(BOTH, plegma::THIRD_SIDE);
          contractGauge_d.copy(gauge);
          applyBoundaryConditions(contractGauge_d, true);
          contractGauge_d.communicateSecondSideGhost();

          TIME(corr3_d.contractNucleonThrp_local(fwd_d, seq_d, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_oneD_d.contractNucleonThrp_oneD(fwd_d, seq_d, contractGauge_d, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_twoD_d.contractNucleonThrp_twoD(fwd_d, seq_d, contractGauge_d, 0, {G1,G2,G3,G4,G5G4}));
          TIME(contractNucleonThrp_fiveD_patterns(fwd_d, seq_d, contractGauge_d,
              corr_space, src, maxQsq, base + "_patterns_double_contr", max_order));
          TIME(corr3_d.writeFile(base + "_local_double_contr", corr_file_format));
          TIME(corr3_oneD_d.writeFile(base + "_oneD_double_contr", corr_file_format));
          TIME(corr3_twoD_d.writeFile(base + "_twoD_double_contr", corr_file_format));
        }

        // flow loop with checkpoints ----------------------------------
        const double eps_3pt  = flow_epsilon;
        const int    n_flow_save = 10;
        int    step = 0;
        double t    = flow_t0;

        updateGaugeQuda(gauge, true, QUDA_WILSON_LINKS);

        for (int ic = 1; ic <= flow_steps;
             ic += (flow_steps / n_flow_save > 0 ? flow_steps / n_flow_save : 1))
        {
          const int target_step = std::min(ic, flow_steps);
          const int chunk = target_step - step;
          if (chunk <= 0) continue;

          if (ic == 1) print_gpu_mem("double flow loop iter 1 (peak alloc)");

          // Flow fwd + seq simultaneously on the same gauge trajectory.
          // Using the double-propagator overload avoids a second updateGaugeQuda
          // call that would corrupt the internal smeared-gauge state.
          TIME(fermionFlow_PLEGMA(*pFwd_out, *pFwd_in, *pSeq_out, *pSeq_in, gf_d, gauge,
                                  chunk, eps_3pt, t,
                                  false, false, true, true, rhs_block));

          std::swap(pFwd_in, pFwd_out);
          std::swap(pSeq_in, pSeq_out);
          step = target_step;
          t    = step * eps_3pt;

          char nt_tag[32];
          snprintf(nt_tag, sizeof(nt_tag), "_nt%02d_double_flow", ic);
          const std::string base = threep_out + "_dt" + std::to_string(dt_sink) + nt_tag;

          // --- float contraction ---
          PLEGMA_Gauge<float> contractGauge_f(BOTH, plegma::THIRD_SIDE);
          contractGauge_f.copy(gf_d);

          PLEGMA_Propagator<float> fwd_f(BOTH, plegma::THIRD_SIDE);
          PLEGMA_Propagator<float> seq_f(BOTH, plegma::THIRD_SIDE);
          fwd_f.copy(*pFwd_in);
          seq_f.copy(*pSeq_in);
          fwd_f.conjugate();
          fwd_f.apply_gamma(G5, LEFT);
          fwd_f.apply_gamma(G5, RIGHT);
          seq_f.apply_gamma(Gsrc, RIGHT);

          TIME(corr3_f.contractNucleonThrp_local(fwd_f, seq_f, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_oneD_f.contractNucleonThrp_oneD(fwd_f, seq_f, contractGauge_f, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_twoD_f.contractNucleonThrp_twoD(fwd_f, seq_f, contractGauge_f, 0, {G1,G2,G3,G4,G5G4}));
          TIME(contractNucleonThrp_fiveD_patterns(fwd_f, seq_f, contractGauge_f,
              corr_space, src, maxQsq, base + "_patterns_float_contr", max_order));
          TIME(corr3_f.writeFile(base + "_local_float_contr", corr_file_format));
          TIME(corr3_oneD_f.writeFile(base + "_oneD_float_contr", corr_file_format));
          TIME(corr3_twoD_f.writeFile(base + "_twoD_float_contr", corr_file_format));

          // --- double contraction ---
          PLEGMA_Propagator<double> fwd_d(BOTH, plegma::THIRD_SIDE);
          PLEGMA_Propagator<double> seq_d(BOTH, plegma::THIRD_SIDE);
          fwd_d.copy(*pFwd_in);
          seq_d.copy(*pSeq_in);
          fwd_d.conjugate();
          fwd_d.apply_gamma(G5, LEFT);
          fwd_d.apply_gamma(G5, RIGHT);
          seq_d.apply_gamma(Gsrc, RIGHT);

          TIME(corr3_d.contractNucleonThrp_local(fwd_d, seq_d, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_oneD_d.contractNucleonThrp_oneD(fwd_d, seq_d, gf_d, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3_twoD_d.contractNucleonThrp_twoD(fwd_d, seq_d, gf_d, 0, {G1,G2,G3,G4,G5G4}));
          TIME(contractNucleonThrp_fiveD_patterns(fwd_d, seq_d, gf_d,
              corr_space, src, maxQsq, base + "_patterns_double_contr", max_order));
          TIME(corr3_d.writeFile(base + "_local_double_contr", corr_file_format));
          TIME(corr3_oneD_d.writeFile(base + "_oneD_double_contr", corr_file_format));
          TIME(corr3_twoD_d.writeFile(base + "_twoD_double_contr", corr_file_format));

          if (step >= flow_steps) break;
        }
      }

    } // tsink loop

    free(src_tag);

    PLEGMA_printf("\n=== Summary ===\n");
    PLEGMA_printf("Flow parameters: n_steps=%d  epsilon=%.4f  t0=%.4f\n",
                  flow_steps, flow_epsilon, flow_t0);
    PLEGMA_printf("Strategy: double-precision flow, float+double contraction\n");
    PLEGMA_printf("Output files:\n");
    PLEGMA_printf("  2pt (unflowed): {twop_out}_double_flow_{float,double}_contr.h5\n");
    PLEGMA_printf("  3pt: {threep_out}_dt*_nt*_double_flow_{local,oneD,twoD,patterns}_{float,double}_contr.h5\n");

    finalizeGaugeQuda();
  }

  while (!threads.empty()) {
    threads.back().join();
    threads.pop_back();
  }
  finalize();
  return 0;
}
