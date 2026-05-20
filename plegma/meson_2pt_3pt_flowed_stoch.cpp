// meson_2pt_3pt_flowed_stoch.cpp
//
// Stochastic-source version of meson_2pt_3pt_flowed.cpp.
//
// Key difference: instead of a point source δ(x - x₀) for each of the
// 12 spin-color components, this code uses a Z4 stochastic noise vector
// ηᵅᵃ(x) that is non-zero over the ENTIRE source timeslice but restricted
// to a single spin-color (α,a) via dilution.  The 12 diluted components
// share the same noise realization so that the one-end trick (OET) holds:
//
//   C(t) = Σₓ Tr[S^stoch(x,t) Γ S^stoch†(x,t)]
//          ──→ Σ_{x,y} Tr[S(x,t;y,t₀) Γ S†(x,t;y,t₀)]   (noise avg.)
//
// which is the all-to-all propagator on the source timeslice, equivalent
// to summing point-source results over all source positions – at the same
// cost of 12 inversions per noise sample.
//
// For multiple noise samples (nsamples > 1) each sample writes to a file
// tagged with _sampleN; average over samples in the analysis stage.
//
// New options compared to point-source code:
//   --nsamples <int>   Number of Z4 noise samples per source position (default 1)
//   --seed   <int>   Base seed for noise generation              (default 1234)

#include "PLEGMA_global.h"
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

#include <complex>
#include <cstdio>
#include <cstddef>
#include <cstdlib>

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
                                      "maxQsq",
                                      "max-order",
                                      "rotate",
                                      "pion",
                                      "kaon-uins",
                                      "kaon-sins"};

  initializeOptions(argc, argv, true, listOpt);

  int max_order = 6;
  HGC_options->set("max-order", "Max derivative order for fiveD patterns (1-6)",
                   verbosity, max_order);

  // ---- Stochastic source options ----
  int nsamples = 1;
  int seed   = 1234;
  HGC_options->set("nsamples",
                   "Number of Z4 stochastic noise samples per source position",
                   verbosity, nsamples);
  HGC_options->set("seed",
                   "Base seed for Z4 noise generation (each sample/source gets a unique derived seed)",
                   verbosity, seed);

  // ---- Strange (kaon) options ----
  double mu_s = std::numeric_limits<double>::quiet_NaN();
  double mu_ud = mu;
  int nsmearGauss_s = nsmearGauss;
  HGC_options->set("Q-mu-s", "twisted-mass mu for strange quark (kaon)",
                   verbosity, mu_s);
  HGC_options->set("nsmear-gauss-s",
                   "Gaussian smearing steps for strange quark propagator",
                   verbosity, nsmearGauss_s);

  double flow_t0 = 2.5;  // dimensionless t0 for this ensemble (in lattice units)
  HGC_options->set("t0-ref", "Dimensionless Wilson-flow reference scale t0 for this ensemble (lattice units); sets t_target=2.5*t0, checkpoints every 0.5*t0",
                   verbosity, flow_t0);

  // ---- Run-mode flags (0/1) ----
  int rotateQ    = 1;  // apply twisted-mass rotation to physical basis
  int pionQ      = 1;  // compute pion 2pt + 3pt  (needs prop_q_sl, prop_q_ss)
  int kaonQ_uins = 0;  // compute K+ 2pt + uins 3pt (needs prop_q_sl, prop_q_ss, prop_s_ss_d)
  int kaonQ_sins = 0;  // compute K+ sins 3pt        (needs prop_q_ss, prop_s_sl_d)
  HGC_options->set("rotate",    "Apply twisted-mass rotation to physical basis (0/1)", verbosity, rotateQ);
  HGC_options->set("pion",      "Compute pion 2pt+3pt channel (0/1)",                  verbosity, pionQ);
  HGC_options->set("kaon-uins", "Compute K+ 2pt + u-insertion 3pt channel (0/1)",     verbosity, kaonQ_uins);
  HGC_options->set("kaon-sins", "Compute K+ s-insertion 3pt channel (0/1)",           verbosity, kaonQ_sins);

  initializePLEGMA();
  {
    if (!pionQ && !kaonQ_uins && !kaonQ_sins) {
      PLEGMA_printf("WARNING: all channels disabled -- forcing pion on.\n");
      pionQ = 1;
    }
    if ((kaonQ_uins || kaonQ_sins) && std::isnan(mu_s))
      PLEGMA_error("kaonQ_uins/sins enabled but --Q-mu-s was not provided. Please set --Q-mu-s.");

    PLEGMA_printf(
        "Run flags: rotateQ=%d pionQ=%d kaonQ_uins=%d kaonQ_sins=%d  mu_ud=%g  mu_s=%g"
        "  nsamples=%d  seed=%d\n",
        (int)rotateQ, (int)pionQ, (int)kaonQ_uins, (int)kaonQ_sins, mu_ud, mu_s, nsamples, seed);

    // ---- Gauge field ----
    PLEGMA_Gauge<double> smearedGauge(BOTH);
    PLEGMA_Gauge<double> gauge(BOTH);

    gauge.readFile(latfile, LIME_FORMAT);
    initGaugeQuda(gauge, true);
    plaqQuda();

    TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
    PLEGMA_printf("Plaquette after APE-smearing:\n");
    smearedGauge.calculatePlaq();

    // ---- Solver ----
    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu, 1));

    const std::string twop_base   = twop_filename;
    const std::string threep_base = threep_filename;
    const double mu_ud_save = mu_ud;

    // ---- Source loop ----
    for (int isrc = 0; isrc < numSourcePositions; ++isrc) {
      updateOptions(LIGHT);
      mu    = mu_ud_save;
      mu_ud = mu_ud_save;
      solver.UpdateSolver();

      site &src = sourcePositions[isrc];
      PLEGMA_printf("\n=== Source %d at (%02d,%02d,%02d,%02d) ===\n",
                    isrc, src[0], src[1], src[2], src[3]);

      PLEGMA_Gauge3D<double> smearedGauge3D_src;
      smearedGauge3D_src.absorb(smearedGauge, src[DIM_T]);

      // ------------------------------------------------------------------
      // build_quark_prop_stoch
      // ------------------------------------------------------------------
      // Replaces the point-source build_quark_prop with a Z4 stochastic
      // timeslice source.  Workflow per spin-color component (α,a):
      //
      //   1. Generate a single 4D Z4 noise vector η (shared across all 12).
      //   2. Spin-color dilute: η^{α,a}(x) = η(x) if (spin,color)==(α,a) else 0.
      //   3. Extract source timeslice → 3D; apply 3D Gaussian smearing.
      //   4. Embed 3D → 4D at t=t_src; rotate; invert; rotate back.
      //   5. Absorb solution into prop_sl (local sink).
      //   6. Smear solution (sink side); absorb into prop_ss (smeared sink).
      //
      // Using ONE noise vector for all 12 columns is essential for the OET:
      // the off-diagonal noise contributions average to zero over samples.
      // ------------------------------------------------------------------
      auto build_quark_prop_stoch = [&](PLEGMA_Propagator<float> &prop_sl,
                                        PLEGMA_Propagator<float> &prop_ss,
                                        double run_mu, WHICHFLAVOR fl,
                                        int nSmear,
                                        const site &src_site,
                                        PLEGMA_Gauge3D<double> &smGauge3D,
                                        int noise_seed) {
        if (mu != run_mu) {
          updateOptions(fl);
          mu = run_mu;
          solver.UpdateSolver();
        }

        // Single Z4 noise vector shared by all 12 spin-color columns
        PLEGMA_Vector<double> eta_full(BOTH);
        eta_full.randInit(noise_seed);
        eta_full.stochastic_Z(4);

        for (int isc = 0; isc < 12; ++isc) {
          PLEGMA_Vector<double> v4D1, v4D2;

          // (a) Spin-color dilution in 4D
          PLEGMA_Vector<double> eta_dil(BOTH);
          eta_dil.dilutespincolor(eta_full, isc / 3, isc % 3);

          // (b) Extract source timeslice → 3D and apply source-side smearing
          PLEGMA_Vector3D<double> eta3D, eta3D_sm;
          eta3D.absorb(eta_dil, src_site[DIM_T]);
          TIME(eta3D_sm.gaussianSmearing(eta3D, smGauge3D, nSmear, alphaGauss));

          // (c) Embed smeared 3D source → 4D at t=t_src
          v4D2.absorb(eta3D_sm, src_site[DIM_T]);

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

          // (d) Store SL (smeared source, local sink)
          PLEGMA_Vector<float> v4D2_f;
          v4D2_f.copy(v4D2);
          prop_sl.absorb(v4D2_f, isc / 3, isc % 3);

          // (e) Smear sink side → store SS (smeared source, smeared sink)
          PLEGMA_Vector<double> v_sm;
          TIME(v_sm.gaussianSmearing(v4D2, smearedGauge, nSmear, alphaGauss));
          PLEGMA_Vector<float> v_sm_f;
          v_sm_f.copy(v_sm);
          prop_ss.absorb(v_sm_f, isc / 3, isc % 3);
        }
      };

      // Per-source filename tags (same as original; sample tag added in noise loop)
      char *src_tag = nullptr;
      const std::string conf_tag = latfile.substr(latfile.length() - 4);
      // Stochastic source fills the full spatial volume; only the source
      // timeslice is meaningful, so we drop sx/sy/sz from the filename.
      asprintf(&src_tag, "_st%02d", src[3]);
      const std::string twop_out   = twop_base   + src_tag + "_chi." + conf_tag;
      const std::string threep_out = threep_base + src_tag + "_chi." + conf_tag;

      // ------------------------------------------------------------------
      // 3pt channel runner (identical logic to point-source version;
      // output tag carries the sample suffix via the `tag` argument)
      // ------------------------------------------------------------------
      auto run_3pt_channel = [&](const std::string &tag,
                                 PLEGMA_Propagator<float> &fwd_full,
                                 PLEGMA_Propagator<float> &fwd_smeared,
                                 WHICHFLAVOR seq_fl,
                                 double seq_mu,
                                 bool active_as_bwd = false) {
        PLEGMA_printf("\n--- 3pt channel: tag='%s' seq_fl=%s seq_mu=%g active_as_bwd=%d ---\n",
                      tag.c_str(),
                      seq_fl == LIGHT ? "LIGHT" : (seq_fl == STRANGE ? "STRANGE" : "CHARM"),
                      seq_mu, (int)active_as_bwd);
        updateOptions(seq_fl);
        mu = seq_mu;
        PLEGMA_printf("mu after updateOptions %f\n", mu);
        solver.UpdateSolver();
        const int seq_nsmear = (seq_fl == LIGHT) ? nsmearGauss : nsmearGauss_s;

        // Pre-extract 3D slices for all tsinks
        std::vector<std::unique_ptr<PLEGMA_Propagator3D<float>>> prop_3D_slices;
        for (size_t i = 0; i < tSinks.size(); ++i) {
          const int st = (src[3] + tSinks[i]) % HGC_totalL[3];
          auto p = std::make_unique<PLEGMA_Propagator3D<float>>();
          p->absorb(fwd_smeared, st);
          prop_3D_slices.push_back(std::move(p));
        }

        // HOST backup of fwd_full (D2H once); device copy left intact for caller
        fwd_full.unload();
        PLEGMA_Propagator<float> prop_fwd_backup(HOST);
        prop_fwd_backup.copy(fwd_full, HOST);

        for (size_t its = 0; its < tSinks.size(); ++its) {
          const int dt_sink = tSinks[its];
          if (dt_sink >= HGC_totalL[3])
            PLEGMA_error("Provided tsink=%d is >= temporal extent", dt_sink);

          const int sink_t = (src[3] + dt_sink) % HGC_totalL[3];

          gauge.calculatePlaq();
          solver.UpdateSolver();

          PLEGMA_Gauge3D<double> smearedGauge3D_sink;
          smearedGauge3D_sink.absorb(smearedGauge, sink_t);

          PLEGMA_Propagator3D<float> &prop_q_3D = *prop_3D_slices[its];

          // Sink-side dressing for the sequential source
          prop_q_3D.apply_gamma(G5, LEFT);

          PLEGMA_Propagator<float> seq_tmp_a(BOTH);
          {
            PLEGMA_Propagator<float> seqProp(BOTH);

            for (int nu = 0; nu < 4; ++nu) {
              for (int c2 = 0; c2 < 3; ++c2) {
                PLEGMA_Vector3D<float> seq3D_f;
                seq3D_f.absorb(prop_q_3D, nu, c2);
                PLEGMA_Vector3D<double> seq3D;
                seq3D.copy(seq3D_f);

                PLEGMA_Vector3D<double> seq3D_sm_d;
                TIME(seq3D_sm_d.gaussianSmearing(seq3D, smearedGauge3D_sink,
                                                 seq_nsmear, alphaGauss));

                PLEGMA_Vector<double> rhs4D1, rhs4D2;
                rhs4D2.absorb(seq3D_sm_d, sink_t);
                PLEGMA_printf("[SEQDBG] sink_t=%d dt_sink=%d nu=%d c2=%d\n",
                              sink_t, dt_sink, nu, c2);
                PLEGMA_printf("[SEQDBG] rhs4D2.norm=%e\n", rhs4D2.norm());

                const double nrm = rhs4D2.norm();
                if (nrm > 0.0) rhs4D2.scale(1.0 / nrm);
                if (rotateQ) rhs4D1.rotateToPhysicalBasis(rhs4D2, mu / abs(mu));
                else         rhs4D1.copy(rhs4D2);

                TIME(solver.solve(rhs4D1, rhs4D1));
                if (rotateQ) rhs4D2.rotateToPhysicalBasis(rhs4D1, mu / abs(mu));
                else         rhs4D2.copy(rhs4D1);

                if (nrm > 0.0) rhs4D2.scale(nrm);

                PLEGMA_Vector<float> rhs4D2_f;
                rhs4D2_f.copy(rhs4D2);
                seqProp.absorb(rhs4D2_f, nu, c2);
              }
            }

            seq_tmp_a.copy(seqProp);
          } // seqProp dtor frees device memory

          // Contract: all bilinear channels in one shot
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Correlator<float> corr3_oneD(corr_space, src, maxQsq);

          PLEGMA_Propagator<double> prop_tmp_a(BOTH);
          PLEGMA_Propagator<double> *in = &prop_tmp_a;
          prop_tmp_a.copy(prop_fwd_backup, HOST);
          prop_tmp_a.load();

          PLEGMA_Propagator<double> seq_in_a(BOTH);
          seq_in_a.copy(seq_tmp_a);
          PLEGMA_Propagator<double> *seq_in = &seq_in_a;

          PLEGMA_Gauge<float> contractGauge(BOTH);
          contractGauge.copy(gauge);
          applyBoundaryConditions(contractGauge, true);

          const double epsilon    = 0.02;
          const double t0         = flow_t0;
          const double t_target   = 2.5 * t0;
          const int    n_steps    = (int)std::lround(t_target / epsilon);
          const int    n_flow_save = (int)std::lround(t_target / (0.1 * t0));

          double t    = 0.0;
          int    step = 0;

          // nt=0 checkpoint (no flow)
          {
            const std::string base = threep_out + tag + "_dt"
                                   + std::to_string(dt_sink) + "_tau00";
            const std::string threep_name_local  = base + "_local";
            const std::string threep_name_oneD   = base + "_oneD";
            const std::string threep_patterns    = base + "_patterns";

            PLEGMA_Propagator<float> in_f(BOTH);
            PLEGMA_Propagator<float> seq_in_f(BOTH);
            in_f.copy(*in);
            seq_in_f.copy(*seq_in);
            PLEGMA_Propagator<float> *bwd_contract = &seq_in_f;
            PLEGMA_Propagator<float> *fwd_contract = &in_f;
            if (active_as_bwd) {
              in_f.conjugate();
              in_f.apply_gamma(G5, LEFT);
              bwd_contract = &in_f;
              fwd_contract = &seq_in_f;
            } else {
              seq_in_f.conjugate();
              seq_in_f.apply_gamma(G5, LEFT);
            }

            // TIME(corr3.contractNucleonThrp_local(*bwd_contract, *fwd_contract,
            //                                      0, {G1,G2,G3,G4,G5G4}));
            TIME(corr3_oneD.contractNucleonThrp_oneD(*bwd_contract, *fwd_contract,
                                                      contractGauge, 0, {G1,G2,G3,G4,G5G4}));
            TIME(contractNucleonThrp_fiveD_patterns(
                *bwd_contract, *fwd_contract, contractGauge, corr_space, src,
                maxQsq, threep_patterns, max_order));
            // TIME(corr3.writeFile(threep_name_local, corr_file_format));
            TIME(corr3_oneD.writeFile(threep_name_oneD, corr_file_format));
          }

          // Flow output buffers
          PLEGMA_Propagator<double> prop_tmp_b(BOTH);
          PLEGMA_Propagator<double> *out = &prop_tmp_b;
          PLEGMA_Propagator<double> seq_tmp_b(BOTH);
          PLEGMA_Propagator<double> *seq_out = &seq_tmp_b;
          PLEGMA_Gauge<double> gauge_flowed(BOTH);

          for (int ic = 1; ic <= n_flow_save; ++ic) {
            const int target_step  = (int)std::lround((double)n_steps * ic / n_flow_save);
            const int chunk_steps  = target_step - step;
            if (chunk_steps <= 0) continue;

            TIME(fermionFlow_PLEGMA(*out, *in, *seq_out, *seq_in, gauge_flowed,
                                    gauge, chunk_steps, epsilon, t,
                                    false, false, true, true, 1));
            std::swap(in,  out);
            std::swap(seq_in, seq_out);
            step = target_step;
            t    = step * epsilon;
            PLEGMA_printf("  checkpoint %d/%d: step=%d  t=%.6f\n",
                          ic, n_flow_save, step, t);

            {
              char tau_tag[32];
              snprintf(tau_tag, sizeof(tau_tag), "_tau%02d",
                       (int)std::lround(t / (0.1 * t0)));
              const std::string base = threep_out + tag + "_dt"
                                     + std::to_string(dt_sink) + tau_tag;
              const std::string threep_name_local  = base + "_local";
              const std::string threep_name_oneD   = base + "_oneD";
              const std::string threep_patterns    = base + "_patterns";

              contractGauge.copy(gauge_flowed);

              PLEGMA_Propagator<float> in_f(BOTH);
              PLEGMA_Propagator<float> seq_in_f(BOTH);
              in_f.copy(*in);
              seq_in_f.copy(*seq_in);
              PLEGMA_Propagator<float> *bwd_contract = &seq_in_f;
              PLEGMA_Propagator<float> *fwd_contract = &in_f;
              if (active_as_bwd) {
                in_f.conjugate();
                in_f.apply_gamma(G5, LEFT);
                bwd_contract = &in_f;
                fwd_contract = &seq_in_f;
              } else {
                seq_in_f.conjugate();
                seq_in_f.apply_gamma(G5, LEFT);
              }

              // TIME(corr3.contractNucleonThrp_local(*bwd_contract, *fwd_contract,
              //                                      0, {G1,G2,G3,G4,G5G4}));
              TIME(corr3_oneD.contractNucleonThrp_oneD(*bwd_contract, *fwd_contract,
                                                        contractGauge, 0, {G1,G2,G3,G4,G5G4}));
              TIME(contractNucleonThrp_fiveD_patterns(
                  *bwd_contract, *fwd_contract, contractGauge, corr_space, src,
                  maxQsq, threep_patterns, max_order));
              // TIME(corr3.writeFile(threep_name_local, corr_file_format));
              TIME(corr3_oneD.writeFile(threep_name_oneD, corr_file_format));
            }
          } // checkpoint loop
        }   // tsink loop
      };    // run_3pt_channel

      // ------------------------------------------------------------------
      // Noise sample loop
      // ------------------------------------------------------------------
      // Seed scheme:
      //   Pion:  u (+mu) and d (-mu) use DIFFERENT seeds to avoid spurious
      //          correlations between the degenerate-doublet propagators.
      //          C2_pion = contractMesonsNew(u, u) uses the SAME propagator
      //          object twice, so it is automatically positive-definite.
      //
      //   Kaon:  C2_kaon = contractMesonsNew(u, sbar) and all C3_kaon channels
      //          involve the pair (u, sbar).  For the stochastic estimate of
      //          C2 to avoid fluctuating through zero (which would blow up the
      //          ratio R = C3/C2), u and sbar MUST share the same Z4 noise
      //          vector eta_full.  Using different seeds gives two independent
      //          noise vectors whose product has unit-scale fluctuations and
      //          can become negative, destroying the signal.
      //          => seed_s = seed_sd = seed_q  (all kaon quarks share eta_u)
      // ------------------------------------------------------------------
      for (int ir = 0; ir < nsamples; ++ir) {
        const std::string sample_tag = (nsamples > 1)
                                       ? ("_sample" + std::to_string(ir))
                                       : "";
        // Use src[3] (source timeslice) rather than the loop index isrc so that
        // the seed depends only on the physical source position, not on how
        // many sources are batched together in the same run.
        const int seed_q  = seed + src[3] * 100000 + ir * 10 + 0;  // u  (+mu)
        // const int seed_qd = seed + src[3] * 100000 + ir * 10 + 1;  // d  (-mu) -- unused
        // Kaon noise cancellation: s and sbar reuse the same seed as u so that
        // all kaon propagators are derived from the same eta_full.
        const int seed_s  = seed_q;   // s  (-mu_s): same eta as u
        const int seed_sd = seed_q;   // sbar (+mu_s): same eta as u

        PLEGMA_printf("\n  --- Noise sample %d / %d (seed_q=%d) ---\n",
                      ir, nsamples, seed_q);

        // ---- Light sector ----
        //   prop_q_sl (+mu_ud, SL): pionQ || kaonQ_uins
        //   prop_q_ss (+mu_ud, SS): pionQ || kaonQ_uins || kaonQ_sins
        std::unique_ptr<PLEGMA_Propagator<float>> prop_q_sl_ptr, prop_q_ss_ptr;
        if (pionQ || kaonQ_uins) {
          prop_q_sl_ptr = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
          prop_q_ss_ptr = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
          TIME(build_quark_prop_stoch(*prop_q_sl_ptr, *prop_q_ss_ptr, +mu_ud_save, LIGHT,
                                      nsmearGauss, src, smearedGauge3D_src, seed_q));
        } else if (kaonQ_sins) {
          prop_q_ss_ptr = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
          {
            PLEGMA_Propagator<float> dummy_sl(BOTH);
            TIME(build_quark_prop_stoch(dummy_sl, *prop_q_ss_ptr, +mu_ud_save, LIGHT,
                                        nsmearGauss, src, smearedGauge3D_src, seed_q));
          }
        }

        // ---- Strange sector ----
        //   prop_s_ss_d (+mu_s, SS): kaonQ_uins  (kaon 2pt + uins seq smear; SL discarded)
        //   prop_s_sl_d (+mu_s, SL): kaonQ_sins  (sins fwd line; SS discarded)
        std::unique_ptr<PLEGMA_Propagator<float>> prop_s_ss_d_ptr, prop_s_sl_d_ptr;
        if (kaonQ_uins) {
          prop_s_ss_d_ptr = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
          {
            PLEGMA_Propagator<float> dummy_sl(BOTH);
            TIME(build_quark_prop_stoch(dummy_sl, *prop_s_ss_d_ptr, +mu_s,
                                        STRANGE, nsmearGauss_s, src,
                                        smearedGauge3D_src, seed_sd));
          }
        }
        if (kaonQ_sins) {
          prop_s_sl_d_ptr = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
          {
            PLEGMA_Propagator<float> dummy_ss(BOTH);
            TIME(build_quark_prop_stoch(*prop_s_sl_d_ptr, dummy_ss, +mu_s,
                                        STRANGE, nsmearGauss_s, src,
                                        smearedGauge3D_src, seed_s));
          }
}

        // Restore light options after potential STRANGE updateOptions in kaon channels
        updateOptions(LIGHT);
        mu    = mu_ud_save;
        mu_ud = mu_ud_save;
        solver.UpdateSolver();

        // ---- 2-point functions ----
        if (pionQ) {
          PLEGMA_Correlator<float> corr2(corr_space, src, maxQsq);
          TIME(corr2.contractMesonsNew(*prop_q_ss_ptr, *prop_q_ss_ptr));
          TIME(corr2.writeFile(twop_out + "_pion" + sample_tag, corr_file_format));
        }
        if (kaonQ_uins) {
          PLEGMA_Correlator<float> corr2(corr_space, src, maxQsq);
          TIME(corr2.contractMesonsNew(*prop_q_ss_ptr, *prop_s_ss_d_ptr));
          TIME(corr2.writeFile(twop_out + "_kaon" + sample_tag, corr_file_format));
        }

        // ---- 3-point functions ----
        const double seq_mu_light = -std::abs(mu_ud_save);

        if (pionQ) {
          run_3pt_channel("_pion_uins" + sample_tag, *prop_q_sl_ptr, *prop_q_ss_ptr,
                          LIGHT, seq_mu_light);
        }
        if (kaonQ_uins) {
          run_3pt_channel("_kaon_uins" + sample_tag, *prop_q_sl_ptr, *prop_s_ss_d_ptr,
                          LIGHT, seq_mu_light);
        }
        if (kaonQ_sins) {
          const double seq_mu_s = -std::abs(mu_s);
          run_3pt_channel("_kaon_sins" + sample_tag, *prop_s_sl_d_ptr, *prop_q_ss_ptr,
                          STRANGE, seq_mu_s, true);
        }
      } // noise sample loop

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
