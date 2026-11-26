// tests/meson_3pt_local_test.cpp
// Stripped-down version of meson_2pt_3pt_flowed.cpp for testing the sign of
// the 3-point function (local operator, no Wilson flow).
// Only contractNucleonThrp_local is called; fiveD patterns and flow are removed.

#include "PLEGMA_global.h"
#include "quda.h"
#include "QUDA_wilson_flow.h"
#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include "PLEGMA_fiveD.h"
#include <memory>

std::vector<std::thread> threads;
using namespace plegma;
using namespace quda;

static std::vector<double> runtime;
#define TIME(fnc)                                                              \
  runtime.push_back(MPI_Wtime());                                              \
  fnc;                                                                         \
  PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back());   \
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

    const std::string twop_base   = twop_filename;
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

      auto build_quark_prop = [&](PLEGMA_Propagator<float> &prop_sl,
                                  PLEGMA_Propagator<float> &prop_ss,
                                  double run_mu, const site &src_site,
                                  PLEGMA_Gauge3D<double> &smearedGauge3D_src) {
        if (mu != run_mu) {
          updateOptions(LIGHT);
          mu = run_mu;
          solver.UpdateSolver();
        }
        for (int isc = 0; isc < 12; ++isc) {
          PLEGMA_Vector<double> v4D1, v4D2;
          PLEGMA_Vector3D<double> s3D_raw, s3D_sm;
          s3D_raw.pointSource(src_site, isc / 3, isc % 3, DEVICE);
          TIME(s3D_sm.gaussianSmearing(s3D_raw, smearedGauge3D_src, nsmearGauss, alphaGauss));
          v4D2.absorb(s3D_sm, src_site[DIM_T]);
          if (rotateQ) {
            TIME(v4D1.rotateToPhysicalBasis(v4D2, run_mu / abs(run_mu)));
          } else {
            v4D1.copy(v4D2);
          }
          TIME(solver.solve(v4D1, v4D1));
          if (rotateQ) {
            TIME(v4D2.rotateToPhysicalBasis(v4D1, run_mu / abs(run_mu)));
          } else {
            v4D2.copy(v4D1);
          }
          PLEGMA_Vector<float> v4D2_f; v4D2_f.copy(v4D2);
          prop_sl.absorb(v4D2_f, isc / 3, isc % 3);
          PLEGMA_Vector<double> v_sm;
          TIME(v_sm.gaussianSmearing(v4D2, smearedGauge, nsmearGauss, alphaGauss));
          PLEGMA_Vector<float> v_sm_f; v_sm_f.copy(v_sm);
          prop_ss.absorb(v_sm_f, isc / 3, isc % 3);
        }
      };

      char *src_tag = nullptr;
      std::string conf_tag = latfile.substr(latfile.length() - 4);
      asprintf(&src_tag, "_sx%02dsy%02dsz%02dst%03d", src[0], src[1], src[2], src[3]);
      const std::string twop_out   = twop_base   + src_tag + "_chi." + conf_tag;
      const std::string threep_out = threep_base + src_tag + "_chi." + conf_tag;

      auto prop_q_ptr      = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
      auto prop_q_d_ptr    = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
      auto prop_q_ss_ptr   = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
      auto prop_q_ss_d_ptr = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
      auto &prop_q   = *prop_q_ptr;
      auto &prop_q_d = *prop_q_d_ptr;
      auto &prop_q_ss   = *prop_q_ss_ptr;
      auto &prop_q_ss_d = *prop_q_ss_d_ptr;

      print_gpu_mem("before_build_prop_q");
      TIME(build_quark_prop(prop_q, prop_q_ss, +mu, src, smearedGauge3D_src));
      print_gpu_mem("after_build_prop_q");
      TIME(build_quark_prop(prop_q_d, prop_q_ss_d, -mu, src, smearedGauge3D_src));
      print_gpu_mem("after_build_prop_q_d");

      PLEGMA_Propagator<float> *prop_src = piplusQ ? &prop_q_ss : &prop_q_ss_d;
      if (piplusQ) prop_q_ss_d_ptr.reset();
      else         prop_q_ss_ptr.reset();
      print_gpu_mem("after_smeared_props_freed");

      // ---- 2-point
      {
        PLEGMA_Correlator<float> corr2(corr_space, src, maxQsq);
        TIME(corr2.contractMesonsNew(*prop_src, *prop_src));
        TIME(corr2.writeFile(twop_out, corr_file_format));
      }

      // ---- Sequential prop setup
      updateOptions(LIGHT);
      if (piplusQ) { mu = -abs(mu); } else { mu = abs(mu); }
      PLEGMA_printf("mu after update option %f", mu);
      solver.UpdateSolver();
      GAMMAS Gsrc = G5;
      GAMMAS Gsnk = G5;

      std::vector<std::unique_ptr<PLEGMA_Propagator3D<float>>> prop_3D_slices;
      for (size_t i = 0; i < tSinks.size(); ++i) {
        const int st = (src[3] + tSinks[i]) % HGC_totalL[3];
        auto p = std::make_unique<PLEGMA_Propagator3D<float>>();
        p->absorb(*prop_src, st);
        prop_3D_slices.push_back(std::move(p));
      }
      if (piplusQ) prop_q_ss_ptr.reset();
      else         prop_q_ss_d_ptr.reset();
      prop_src = nullptr;

      PLEGMA_Propagator<float> &prop_fwd_ref = piplusQ ? prop_q : prop_q_d;
      prop_fwd_ref.unload();
      PLEGMA_Propagator<float> prop_fwd_backup(HOST);
      prop_fwd_backup.copy(prop_fwd_ref, HOST);
      prop_q_ptr.reset();
      prop_q_d_ptr.reset();
      print_gpu_mem("after_tsink_presetup_freed");

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
        print_gpu_mem("after_prop_q_3D_absorb");

        prop_q_3D.apply_gamma(Gsnk, LEFT);

        PLEGMA_Propagator<float> seq_tmp(BOTH);
        {
          PLEGMA_Propagator<float> seqProp(BOTH);
          print_gpu_mem("before_seqprop_solve");

          for (int nu = 0; nu < 4; ++nu) {
            for (int c2 = 0; c2 < 3; ++c2) {
              PLEGMA_Vector3D<float> seq3D_f;
              seq3D_f.absorb(prop_q_3D, nu, c2);
              PLEGMA_Vector3D<double> seq3D;
              seq3D.copy(seq3D_f);

              PLEGMA_Vector3D<double> seq3D_sm_d;
              TIME(seq3D_sm_d.gaussianSmearing(seq3D, smearedGauge3D_sink, nsmearGauss, alphaGauss));

              PLEGMA_Vector<double> rhs4D1, rhs4D2;
              rhs4D2.absorb(seq3D_sm_d, sink_t);

              const double nrm = rhs4D2.norm();
              if (nrm > 0.0) rhs4D2.scale(1.0 / nrm);
              if (rotateQ) {
                rhs4D1.rotateToPhysicalBasis(rhs4D2, mu / abs(mu));
              } else {
                rhs4D1.copy(rhs4D2);
              }
              TIME(solver.solve(rhs4D1, rhs4D1));
              if (rotateQ) {
                rhs4D2.rotateToPhysicalBasis(rhs4D1, mu / abs(mu));
              } else {
                rhs4D2.copy(rhs4D1);
              }
              if (nrm > 0.0) rhs4D2.scale(nrm);

              PLEGMA_Vector<float> rhs4D2_f; rhs4D2_f.copy(rhs4D2);
              seqProp.absorb(rhs4D2_f, nu, c2);
            }
          }
          print_gpu_mem("after_seqprop_solve");
          seq_tmp.copy(seqProp);
        } // seqProp freed

        // ---- 3-point local contraction (no flow, tau=0 only)
        // Pre-load propagators (shared by all variants)
        PLEGMA_Propagator<double> prop_in_base(BOTH);
        prop_in_base.copy(prop_fwd_backup, HOST);
        prop_in_base.load();

        PLEGMA_Propagator<double> seq_in_base(BOTH);
        seq_in_base.copy(seq_tmp);

        const std::string base = threep_out + "_dt" + std::to_string(dt_sink) + "_tau00";

        // === variant: current code (baseline) ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G4}));
          TIME(corr3.writeFile(base + "_local", corr_file_format));
        }

        // === variant v1: signProps = -1 (TMM rotation) ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, -1, {G4}));
          TIME(corr3.writeFile(base + "_local_v1_signm1", corr_file_format));
        }

        // === variant v2: negate bwd prop before contraction ===
        // in_f scaled by -1 after all gamma ops → overall sign flip
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          in_f.scale(-1.0f);
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G4}));
          TIME(corr3.writeFile(base + "_local_v2_negated", corr_file_format));
        }

        // === variant v3: Gsrc = G5G4 instead of G5 ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          seq_in_f.apply_gamma(G5G4, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G4}));
          TIME(corr3.writeFile(base + "_local_v3_Gsrc_G5G4", corr_file_format));
        }

        // === variant v4: no apply_gamma on seq_in_f (no Gsrc) ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          // no Gsrc on seq_in_f
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G4}));
          TIME(corr3.writeFile(base + "_local_v4_noGsrc", corr_file_format));
        }

        // === variant v5: in_f without conjugate (G5 S G5, no conj) ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          // NO conjugate
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G4}));
          TIME(corr3.writeFile(base + "_local_v5_noconj", corr_file_format));
        }

        // === variant v6: in_f only conjugate, no G5 wrapping ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          // NO G5 LEFT/RIGHT
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G4}));
          TIME(corr3.writeFile(base + "_local_v6_conjonly", corr_file_format));
        }

        // === variant v7: negate seq_in_f instead of in_f ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          seq_in_f.scale(-1.0f);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G4}));
          TIME(corr3.writeFile(base + "_local_v7_negseq", corr_file_format));
        }

        // === variant v8: all 5 gammas, current code (for full operator comparison) ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3.writeFile(base + "_local_v8_allgammas", corr_file_format));
        }

        // === variant v9: all 5 gammas, negated in_f (corrected sign) ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          in_f.scale(-1.0f);
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3.writeFile(base + "_local_v9_allgammas_neginf", corr_file_format));
        }

        // === variant v10: reference convention (mesons_3pt.cpp scheme) ===
        // Apply γ5 + conjugate to seq prop, pass it as bwdProp.
        // Forward prop is passed raw (no γ5 sandwich, no conjugate).
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> seq_bwd(BOTH);
          PLEGMA_Propagator<float> fwd(BOTH);
          seq_bwd.copy(seq_in_base);
          fwd.copy(prop_in_base);
          seq_bwd.apply_gamma(G5, LEFT);   // γ5 · seq
          seq_bwd.conjugate();              // (γ5 · seq)*
          // fwd: no transformations
          TIME(corr3.contractNucleonThrp_local(seq_bwd, fwd, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3.writeFile(base + "_local_v10_refconv", corr_file_format));
        }

        // === variant v11: same as v8 but drop seq_in_f.apply_gamma(Gsrc, RIGHT) ===
        // Tests if the extra G5 on seq side is the source of the -1
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          // NO seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3.writeFile(base + "_local_v11_noGsrc5g", corr_file_format));
        }

        // === variant v12: drop one γ5 from in_f sandwich (keep RIGHT only) ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          // in_f.apply_gamma(G5, LEFT);   // dropped
          in_f.apply_gamma(G5, RIGHT);
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3.writeFile(base + "_local_v12_onlyG5R", corr_file_format));
        }

        // === variant v13: drop one γ5 from in_f sandwich (keep LEFT only) ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          // in_f.apply_gamma(G5, RIGHT);   // dropped
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {G1,G2,G3,G4,G5G4}));
          TIME(corr3.writeFile(base + "_local_v13_onlyG5L", corr_file_format));
        }

        // === variant v14: current scheme, gammas = {ONE,G5,G1,G2,G3,G4,G5G4} ===
        // For testing γ5-algebra hypothesis: ONE,G5 should NOT pick up -1,
        // while G1..G4,G5G4 should.
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0, {ONE,G5,G1,G2,G3,G4,G5G4}));
          TIME(corr3.writeFile(base + "_local_v14_scalar_vector", corr_file_format));
        }

        // === variant v15: reference scheme, same gamma list as v14 ===
        // Reference: bwd = (γ5·Σ)*, fwd = S(raw).
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> seq_bwd(BOTH);
          PLEGMA_Propagator<float> fwd(BOTH);
          seq_bwd.copy(seq_in_base);
          fwd.copy(prop_in_base);
          seq_bwd.apply_gamma(G5, LEFT);
          seq_bwd.conjugate();
          TIME(corr3.contractNucleonThrp_local(seq_bwd, fwd, 0, {ONE,G5,G1,G2,G3,G4,G5G4}));
          TIME(corr3.writeFile(base + "_local_v15_refconv_scalar_vector", corr_file_format));
        }

        // === variant v16: ALL 16 gammas, current scheme ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> in_f(BOTH);
          PLEGMA_Propagator<float> seq_in_f(BOTH);
          in_f.copy(prop_in_base);
          seq_in_f.copy(seq_in_base);
          in_f.conjugate();
          in_f.apply_gamma(G5, LEFT);
          in_f.apply_gamma(G5, RIGHT);
          seq_in_f.apply_gamma(Gsrc, RIGHT);
          TIME(corr3.contractNucleonThrp_local(in_f, seq_in_f, 0,
              {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43}));
          TIME(corr3.writeFile(base + "_local_v16_all16", corr_file_format));
        }

        // === variant v17: ALL 16 gammas, reference scheme ===
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> seq_bwd(BOTH);
          PLEGMA_Propagator<float> fwd(BOTH);
          seq_bwd.copy(seq_in_base);
          fwd.copy(prop_in_base);
          seq_bwd.apply_gamma(G5, LEFT);
          seq_bwd.conjugate();
          TIME(corr3.contractNucleonThrp_local(seq_bwd, fwd, 0,
              {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43}));
          TIME(corr3.writeFile(base + "_local_v17_refconv_all16", corr_file_format));
        }

        // === variant v18a: reference scheme but ONLY γ5 (no conjugate) ===
        // Isolates the γ5·Σ insertion contribution.
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> seq_bwd(BOTH);
          PLEGMA_Propagator<float> fwd(BOTH);
          seq_bwd.copy(seq_in_base);
          fwd.copy(prop_in_base);
          seq_bwd.apply_gamma(G5, LEFT);
          // NO conjugate
          TIME(corr3.contractNucleonThrp_local(seq_bwd, fwd, 0,
              {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43}));
          TIME(corr3.writeFile(base + "_local_v18a_only_g5", corr_file_format));
        }

        // === variant v18b: reference scheme but ONLY conjugate (no γ5) ===
        // Isolates the complex-conjugation contribution.
        {
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Propagator<float> seq_bwd(BOTH);
          PLEGMA_Propagator<float> fwd(BOTH);
          seq_bwd.copy(seq_in_base);
          fwd.copy(prop_in_base);
          // NO γ5
          seq_bwd.conjugate();
          TIME(corr3.contractNucleonThrp_local(seq_bwd, fwd, 0,
              {ONE,G1,G2,G3,G4,G5,G5G1,G5G2,G5G3,G5G4,S12,S13,S23,S41,S42,S43}));
          TIME(corr3.writeFile(base + "_local_v18b_only_conj", corr_file_format));
        }

        print_gpu_mem("after_3pt_contraction");
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
