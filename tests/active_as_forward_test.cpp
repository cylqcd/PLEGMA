// tests/active_as_forward_test.cpp
//
// Minimal reproducer for the pion 3-pt "active_as_bwd" cross-check path.
// PURPOSE: compare intermediate propagators and final correlators between
//   * myapps/build/active_as_forward_test  (current code)
//   * myapps_dev/build/meson_2pt_3pt       (dev, piplusQ=false)
//
// Both binaries should be run with ROTATEQ=false and PIPLUSQ=false on the
// same configuration / source.  The HDF5 dump files produced here can then
// be directly compared with those produced by the dev binary.
//
// Pipeline mirrors dev piplusQ=false exactly:
//   fwd propagator  : prop_q_d  (smeared-local, mu = -|mu_l|)
//   smeared source  : prop_q_ss_d (smeared-smeared, mu = -|mu_l|)
//   seq source      : extracted from prop_q_ss_d at sink_t with G5 LEFT
//   seq solver mu   : +|mu_l|
//   contraction (active_as_bwd variant):
//     in_f   = conj(prop_q_d); G5 LEFT; G5 RIGHT  → bwd slot
//     seq_in_f = seqProp; G5 RIGHT               → fwd slot
//     kernel: Tr[Γ · in_f · seq_in_f^T]
//
// Saved HDF5 files per source×tsink:
//   prop_fwd_d_<srctag>.h5       — prop_q_d (forward, -mu, SL)
//   prop_ss_d_<srctag>.h5        — prop_q_ss_d (smeared-smeared, -mu)
//   seqprop_<srctag>_dt<dt>.h5  — sequential propagator (+mu solve)
//   threep_<srctag>_dt<dt>_active_bwd_local.h5    — contraction (all 5 gammas)

#include "PLEGMA_global.h"
#include "quda.h"
#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include <memory>

std::vector<std::thread> threads;
using namespace plegma;
using namespace quda;

static std::vector<double> runtime;
#define TIME(fnc)                                                              \
  runtime.push_back(MPI_Wtime());                                              \
  fnc;                                                                         \
  PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back());  \
  runtime.pop_back()

int main(int argc, char **argv) {
  std::vector<std::string> listOpt = {"verbosity",
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
                                      "maxQsq"};

  initializeOptions(argc, argv, true, listOpt);
  initializePLEGMA();
  {
    bool rotateQ = false;  // default OFF — matches dev default

    auto read_bool_env = [](const char *name, bool def_val) {
      const char *v = std::getenv(name);
      if (!v) return def_val;
      std::string s(v);
      if (s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES") return true;
      if (s == "0" || s == "false" || s == "FALSE" || s == "no" || s == "NO") return false;
      return def_val;
    };

    rotateQ = read_bool_env("ROTATEQ", rotateQ);
    PLEGMA_printf("Run flags: rotateQ=%d\n", (int)rotateQ);

    PLEGMA_Gauge<double> smearedGauge(BOTH);
    PLEGMA_Gauge<double> gauge(BOTH);
    gauge.readFile(latfile, LIME_FORMAT);
    initGaugeQuda(gauge, true);

    auto print_gpu_mem = [](const char *tag) {
      size_t free_b = 0, total_b = 0;
      cudaMemGetInfo(&free_b, &total_b);
      PLEGMA_printf("[GMEM][%s] free=%.2f GB  used=%.2f GB  total=%.2f GB\n",
                    tag,
                    free_b  / 1073741824.0,
                    (total_b - free_b) / 1073741824.0,
                    total_b / 1073741824.0);
    };

    plaqQuda();
    print_gpu_mem("after_initGaugeQuda");

    TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
    PLEGMA_printf("Plaquette after APE-smearing:\n");
    smearedGauge.calculatePlaq();

    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu, 1));
    print_gpu_mem("after_QUDA_solver_init");

    const std::string threep_base = threep_filename;
    const double mu_orig = mu;

    for (int isrc = 0; isrc < numSourcePositions; ++isrc) {
      mu = mu_orig;
      solver.UpdateSolver();
      site &src = sourcePositions[isrc];
      PLEGMA_printf("\n=== Source %d at (%02d,%02d,%02d,%02d) ===\n", isrc,
                    src[0], src[1], src[2], src[3]);

      PLEGMA_Gauge3D<double> smearedGauge3D_src;
      smearedGauge3D_src.absorb(smearedGauge, src[DIM_T]);

      // -------------------------------------------------------------------
      // Propagator builder: identical to both dev and current myapps
      // -------------------------------------------------------------------
      auto build_quark_prop = [&](PLEGMA_Propagator<float> &prop_sl,
                                  PLEGMA_Propagator<float> &prop_ss,
                                  double run_mu,
                                  const site &src_site,
                                  PLEGMA_Gauge3D<double> &sg3D) {
        if (mu != run_mu) {
          updateOptions(LIGHT);
          mu = run_mu;
          solver.UpdateSolver();
        }
        for (int isc = 0; isc < 12; ++isc) {
          PLEGMA_Vector<double> v4D1, v4D2;
          PLEGMA_Vector3D<double> s3D_raw, s3D_sm;
          s3D_raw.pointSource(src_site, isc / 3, isc % 3, DEVICE);
          TIME(s3D_sm.gaussianSmearing(s3D_raw, sg3D, nsmearGauss, alphaGauss));
          v4D2.absorb(s3D_sm, src_site[DIM_T]);
          if (rotateQ) {
            TIME(v4D1.rotateToPhysicalBasis(v4D2, run_mu / std::abs(run_mu)));
          } else {
            v4D1.copy(v4D2);
          }
          TIME(solver.solve(v4D1, v4D1));
          if (rotateQ) {
            TIME(v4D2.rotateToPhysicalBasis(v4D1, run_mu / std::abs(run_mu)));
          } else {
            v4D2.copy(v4D1);
          }
          PLEGMA_Vector<float> v4D2_f;
          v4D2_f.copy(v4D2);
          prop_sl.absorb(v4D2_f, isc / 3, isc % 3);
          PLEGMA_Vector<double> v_sm;
          TIME(v_sm.gaussianSmearing(v4D2, smearedGauge, nsmearGauss, alphaGauss));
          PLEGMA_Vector<float> v_sm_f;
          v_sm_f.copy(v_sm);
          prop_ss.absorb(v_sm_f, isc / 3, isc % 3);
        }
      };

      // Source tag for file naming
      char *src_tag = nullptr;
      std::string conf_tag = latfile.substr(latfile.length() - 4);
      asprintf(&src_tag, "_sx%02dsy%02dsz%02dst%03d", src[0], src[1], src[2], src[3]);
      const std::string threep_out = threep_base + src_tag + "_chi." + conf_tag;

      // -------------------------------------------------------------------
      // Build +mu propagator FIRST — mirrors dev which always solves +mu
      // before -mu, so MG deflation state matches exactly.
      // The +mu prop itself is not used for contraction; it is discarded.
      // -------------------------------------------------------------------
      {
        PLEGMA_Propagator<float> prop_q_warmup(BOTH), prop_q_ss_warmup(BOTH);
        TIME(build_quark_prop(prop_q_warmup, prop_q_ss_warmup, +std::abs(mu_orig), src,
                              smearedGauge3D_src));
        PLEGMA_printf("Warm-up +mu solve done (discarding prop).\n");
      }
      print_gpu_mem("after_warmup_prop_freed");

      // -------------------------------------------------------------------
      // Build d-quark propagators  (mu = -|mu_l|)
      //   prop_q_sl_d : smeared-local  (= "prop_q_d" in dev)
      //   prop_q_ss_d : smeared-smeared
      // -------------------------------------------------------------------
      PLEGMA_Propagator<float> prop_q_sl_d(BOTH), prop_q_ss_d(BOTH);
      TIME(build_quark_prop(prop_q_sl_d, prop_q_ss_d, -std::abs(mu_orig), src,
                            smearedGauge3D_src));
      print_gpu_mem("after_build_prop_q_d");

      // Save forward and smeared propagators to HDF5 for comparison with dev
      {
        const std::string fwd_fname  = threep_out + "_prop_fwd_d";
        const std::string smr_fname  = threep_out + "_prop_ss_d";
        prop_q_sl_d.writeHDF5(fwd_fname);
        prop_q_ss_d.writeHDF5(smr_fname);
        PLEGMA_printf("Saved forward prop: %s.h5\n",  fwd_fname.c_str());
        PLEGMA_printf("Saved smeared prop: %s.h5\n",  smr_fname.c_str());
      }

      // -------------------------------------------------------------------
      // Sequential solver setup — matches dev piplusQ=false:
      //   source from prop_q_ss_d, solve at mu = +|mu_l|
      // -------------------------------------------------------------------
      updateOptions(LIGHT);
      mu = +std::abs(mu_orig);
      PLEGMA_printf("mu for sequential solve: %f\n", mu);
      solver.UpdateSolver();

      const GAMMAS Gsrc = G5;
      const GAMMAS Gsnk = G5;

      // Pre-extract 3D slices from prop_q_ss_d for all tsinks
      std::vector<std::unique_ptr<PLEGMA_Propagator3D<float>>> prop_3D_slices;
      for (size_t i = 0; i < tSinks.size(); ++i) {
        const int st = (src[3] + tSinks[i]) % HGC_totalL[3];
        auto p = std::make_unique<PLEGMA_Propagator3D<float>>();
        p->absorb(prop_q_ss_d, st);
        prop_3D_slices.push_back(std::move(p));
      }

      // Keep a HOST backup of the forward prop (unmodified)
      prop_q_sl_d.unload();
      PLEGMA_Propagator<float> prop_fwd_backup(HOST);
      prop_fwd_backup.copy(prop_q_sl_d, HOST);
      print_gpu_mem("after_tsink_presetup");

      // -------------------------------------------------------------------
      // tsink loop
      // -------------------------------------------------------------------
      for (size_t its = 0; its < tSinks.size(); ++its) {
        const int dt_sink = tSinks[its];
        if (dt_sink >= HGC_totalL[3]) {
          PLEGMA_error("Provided tsink=%d is >= temporal extent", dt_sink);
        }
        const int sink_t = (src[3] + dt_sink) % HGC_totalL[3];

        gauge.calculatePlaq();
        solver.UpdateSolver();

        PLEGMA_Gauge3D<double> smearedGauge3D_sink;
        smearedGauge3D_sink.absorb(smearedGauge, sink_t);

        PLEGMA_Propagator3D<float> &prop_q_3D = *prop_3D_slices[its];

        // Sink-side Gsnk=G5 dressing — identical in both codes
        prop_q_3D.apply_gamma(Gsnk, LEFT);

        // Sequential solve
        PLEGMA_Propagator<float> seqProp(BOTH);
        {
          print_gpu_mem("before_seqprop_solve");
          for (int nu = 0; nu < 4; ++nu) {
            for (int c2 = 0; c2 < 3; ++c2) {
              PLEGMA_Vector3D<float> seq3D_f;
              seq3D_f.absorb(prop_q_3D, nu, c2);
              PLEGMA_Vector3D<double> seq3D;
              seq3D.copy(seq3D_f);

              PLEGMA_Vector3D<double> seq3D_sm_d;
              TIME(seq3D_sm_d.gaussianSmearing(seq3D, smearedGauge3D_sink,
                                               nsmearGauss, alphaGauss));

              PLEGMA_Vector<double> rhs4D1, rhs4D2;
              rhs4D2.absorb(seq3D_sm_d, sink_t);
              PLEGMA_printf("[SEQDBG] sink_t=%d dt_sink=%d nu=%d c2=%d  norm=%e\n",
                            sink_t, dt_sink, nu, c2, rhs4D2.norm());

              const double nrm = rhs4D2.norm();
              if (nrm > 0.0) rhs4D2.scale(1.0 / nrm);
              if (rotateQ) {
                rhs4D1.rotateToPhysicalBasis(rhs4D2, mu / std::abs(mu));
              } else {
                rhs4D1.copy(rhs4D2);
              }
              TIME(solver.solve(rhs4D1, rhs4D1));
              if (rotateQ) {
                rhs4D2.rotateToPhysicalBasis(rhs4D1, mu / std::abs(mu));
              } else {
                rhs4D2.copy(rhs4D1);
              }
              if (nrm > 0.0) rhs4D2.scale(nrm);

              PLEGMA_Vector<float> rhs4D2_f;
              rhs4D2_f.copy(rhs4D2);
              seqProp.absorb(rhs4D2_f, nu, c2);
            }
          }
          print_gpu_mem("after_seqprop_solve");
        }

        // Save sequential propagator for comparison with dev
        {
          const std::string seq_fname = threep_out + "_seqprop_dt"
                                      + std::to_string(dt_sink);
          seqProp.writeHDF5(seq_fname);
          PLEGMA_printf("Saved seqprop: %s.h5\n", seq_fname.c_str());
        }

        // -------------------------------------------------------------------
        // Load forward prop to device (float, no double round-trip) — same
        // as dev's direct float copy into prop_tmp_a
        // -------------------------------------------------------------------
        PLEGMA_Propagator<float> in_f(BOTH);
        in_f.copy(prop_fwd_backup, HOST); // HOST float → device float
        in_f.load();
        PLEGMA_Propagator<float> seq_in_f(BOTH);
        seq_in_f.copy(seqProp);

        const std::string base = threep_out + "_dt" + std::to_string(dt_sink) + "_tau00";

        // ---- Boundary conditions on contractGauge (same as both codes) ----
        PLEGMA_Gauge<float> contractGauge(BOTH);
        contractGauge.copy(gauge);
        applyBoundaryConditions(contractGauge, true);

        // -------------------------------------------------------------------
        // contraction: active_as_bwd convention
        //   bwd = G5 · in_f† · G5  (i.e. γ5-Hermiticity on the forward prop)
        //   fwd = seqProp · G5      (Gsrc=G5 on sequential)
        //   kernel computes Tr[Γ · bwd · fwd^T]
        // -------------------------------------------------------------------
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> bwd(BOTH);
          PLEGMA_Propagator<float> fwd(BOTH);
          bwd.copy(in_f);
          fwd.copy(seq_in_f);
          bwd.conjugate();
          bwd.apply_gamma(G5, LEFT);
          bwd.apply_gamma(G5, RIGHT);
          fwd.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(bwd, fwd, 0, {G1,G2,G3,G4,G5G4}));
          const std::string fname = base + "_active_bwd_local";
          TIME(corr3.writeFile(fname, corr_file_format));
          PLEGMA_printf("Saved: %s\n", fname.c_str());
        }

        // -------------------------------------------------------------------
        // contraction: default (forward) convention — what the dev code does
        // with the seq prop in the bwd slot and the forward prop in the fwd slot
        // (This is the mesons_3pt.cpp reference scheme: bwd=(G5·seq)*, fwd=S_raw)
        // -------------------------------------------------------------------
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> bwd(BOTH);
          PLEGMA_Propagator<float> fwd_raw(BOTH);
          bwd.copy(seq_in_f);
          fwd_raw.copy(in_f);
          bwd.apply_gamma(G5, LEFT);
          bwd.conjugate();
          TIME(corr3.contractNucleonThrp_local(bwd, fwd_raw, 0, {G1,G2,G3,G4,G5G4}));
          const std::string fname = base + "_fwd_conv_local";
          TIME(corr3.writeFile(fname, corr_file_format));
          PLEGMA_printf("Saved: %s\n", fname.c_str());
        }

        // -------------------------------------------------------------------
        // contraction: dev inline convention (exactly what the dev code does
        // inside the no-flow checkpoint, piplusQ=false)
        //   in->conjugate(); in->G5L; in->G5R; seq_in->G5R; contract(in, seq)
        // Applied on fresh copies so we don't pollute in_f/seq_in_f.
        // -------------------------------------------------------------------
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> bwd(BOTH);
          PLEGMA_Propagator<float> fwd(BOTH);
          bwd.copy(in_f);
          fwd.copy(seq_in_f);
          // Exactly lines 507-513 of dev meson_2pt_3pt_flowed.cpp
          bwd.conjugate();
          bwd.apply_gamma(G5, LEFT);
          bwd.apply_gamma(G5, RIGHT);
          fwd.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(bwd, fwd, 0, {G1,G2,G3,G4,G5G4}));
          const std::string fname = base + "_dev_inline_local";
          TIME(corr3.writeFile(fname, corr_file_format));
          PLEGMA_printf("Saved: %s\n", fname.c_str());
        }

        print_gpu_mem("after_contractions");
      } // tsink loop

      free(src_tag);
    } // source loop

    while (not threads.empty()) {
      threads.back().join();
      threads.pop_back();
    }
  }
  finalize();
  return 0;
}
