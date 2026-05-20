// meson_2pt_3pt_all.cpp  -- Meson 2pt & 3pt with contractMesonsAll (no manual
// gamma lists) Build against your PLEGMA/QUDA stack. Comments are in English
// only.

#include "PLEGMA_global.h"
#include "QUDA_wilson_flow.h"
#include <PLEGMA.h>
#include <PLEGMA_utils.h>
#include "PLEGMA_fiveD.h"
#include <memory>

std::vector<std::thread> threads;
using namespace plegma;
using namespace quda;

// Simple scoped timer
static std::vector<double> runtime;
#define TIME(fnc)                                                              \
  runtime.push_back(MPI_Wtime());                                              \
  fnc;                                                                         \
  PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back());   \
  runtime.pop_back()

int main(int argc, char **argv) {
  // Options we rely on from PLEGMA's option system
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

  // Initialize option system and global state
  initializeOptions(argc, argv, true, listOpt);

  int max_order = 6;
  HGC_options->set("max-order", "Max derivative order for fiveD patterns (1-6)", verbosity, max_order);

  // ---- Strange (kaon) options ----
  // mu_s        : twisted-mass parameter for strange quark
  // nsmearGauss_s : Gaussian smearing steps for strange quark sources
  // mu_ud       : alias of light-quark mu (saved before any updateOptions(STRANGE))
  double mu_s = std::numeric_limits<double>::quiet_NaN();  // must be set if kaonQ_uins or kaonQ_sins
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
    PLEGMA_printf("Run flags: rotateQ=%d pionQ=%d kaonQ_uins=%d kaonQ_sins=%d  mu_ud=%g  mu_s=%g\n",
                  (int)rotateQ, (int)pionQ, (int)kaonQ_uins, (int)kaonQ_sins, mu_ud, mu_s);
    // Read gauge, initialize QUDA, then APE-smear (once)
    PLEGMA_Gauge<double> smearedGauge(BOTH);

    // {
    PLEGMA_Gauge<double> gauge(BOTH); // host-side, high precision
    // Read gauge from LIME file; measure host plaquette
    gauge.readFile(latfile, LIME_FORMAT);
    // gauge.setUnit((std::vector<int>) {0,4,8, 9,13,17, 18,22,26, 27,31,35});
    // gauge.readFile("/home/hubl/run/haobo/gauge.lime");
    // gauge.calculatePlaq();
    // gauge.writeHDF5("gauge_dummy.h5");

    // Upload to QUDA; check plaquette on device side.
    // initGaugeQuda(gauge, true) does the following:
    //   1. Copies `gauge` (PLEGMA, NO boundary phase) into a temp buffer
    //   2. Applies anti-periodic BC to the buffer: U_t(T-1,x) *= -1
    //   3. Loads the BC-modified buffer into QUDA as gaugePrecise
    // After this call:
    //   - PLEGMA `gauge` is UNCHANGED (no BC, raw lattice links)
    //   - QUDA  `gaugePrecise` CONTAINS anti-periodic BC
    // All subsequent QUDA solves and Wilson flow use gaugePrecise, so BC
    // is automatically respected in Dirac operator and flow Laplacian.
    initGaugeQuda(gauge, true);
    // dump_time_boundary_links_from_H_elem(gauge, "AFTER_initGaugeQuda");
//     PLEGMA_Gauge<double> g0(BOTH), g1(BOTH);
// g0.copy(gauge);
// g1.copy(gauge);

// dump_time_boundary_links_from_H_elem(g0, "BEFORE_applyBC");
// g0.unload();

// applyBoundaryConditions(g1, true);

// dump_time_boundary_links_from_H_elem(g1, "AFTER_applyBC");
// g1.unload();
    // dump_resident_gauge("AFTER_INIT");
    // dump_quda_gauge_ptrs("AFTER_INIT");

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

    // APE-smearing to build smeared gauge (used by Gaussian smearing)
    TIME(smearedGauge.APEsmearing(gauge, nsmearAPE, alphaAPE, 3));
    PLEGMA_printf("Plaquette after APE-smearing:\n");
    smearedGauge.calculatePlaq();
    // }

    // Make solver at current mu (twisted-mass convention)
    updateOptions(LIGHT);
    TIME(QUDA_solver solver(mu, 1));
    print_gpu_mem("after_QUDA_solver_init");

    // Keep base filenames; we will append source-specific suffixes
    const std::string twop_base = twop_filename;
    const std::string threep_base = threep_filename;

    // PLEGMA_printf("nSmearGauss is "  " %d ", nsmearGauss);
    // PLEGMA_printf("nSmearAPE is "  " %d ", nsmearAPE);
    // 3D gauge slice at the source time (for source-side smearing)

    // Loop over sources
    const double mu_ud_save = mu_ud;  // save before loop — mu gets flipped for seq props
    for (int isrc = 0; isrc < numSourcePositions; ++isrc) {
      // Restore light mu / LIGHT options at start of each source
      updateOptions(LIGHT);
      mu = mu_ud_save;
      mu_ud = mu_ud_save;
      solver.UpdateSolver();
      site &src = sourcePositions[isrc];
      PLEGMA_printf("\n=== Source %d at (%02d,%02d,%02d,%02d) ===\n", isrc,
                    src[0], src[1], src[2], src[3]);
      PLEGMA_Gauge3D<double> smearedGauge3D_src;
      smearedGauge3D_src.absorb(smearedGauge, src[DIM_T]);
      // Helper: build a smeared-local (SL) quark propagator at given mass
      auto build_quark_prop = [&](PLEGMA_Propagator<float> &prop_sl,
                                  PLEGMA_Propagator<float> &prop_ss,
                                  double run_mu, WHICHFLAVOR fl,
                                  int nSmear,
                                  const site &src_site,
                                  PLEGMA_Gauge3D<double> &smearedGauge3D_src) {
        // Switch mass / clover-csw / kappa in solver if needed
        if (mu != run_mu) {
          updateOptions(fl);
          mu = run_mu;
          solver.UpdateSolver();
        }

        // Loop over 12 spin-color components of a point source
        for (int isc = 0; isc < 12; ++isc) {
          PLEGMA_Vector<double> v4D1, v4D2;

          // 3D point source -> 3D Gaussian smear -> embed to 4D at t=src_t
          PLEGMA_Vector3D<double> s3D_raw, s3D_sm;
          s3D_raw.pointSource(src_site, isc / 3, isc % 3, DEVICE);
          TIME(s3D_sm.gaussianSmearing(s3D_raw, smearedGauge3D_src, nSmear,
                                       alphaGauss));
          v4D2.absorb(s3D_sm, src_site[DIM_T]);
          if (rotateQ == true) {
            TIME(v4D1.rotateToPhysicalBasis(v4D2, run_mu / abs(run_mu)));
          } else {
            v4D1.copy(v4D2);
          }

          TIME(solver.solve(v4D1, v4D1));
          if (rotateQ == true) {
            TIME(v4D2.rotateToPhysicalBasis(v4D1, run_mu / abs(run_mu)));
          } else {
            v4D2.copy(v4D1);
          }

          PLEGMA_Vector<float> v4D2_f; v4D2_f.copy(v4D2);
          prop_sl.absorb(v4D2_f, isc / 3, isc % 3);
          // Sink-side Gaussian smearing -> absorb to propagator column
          PLEGMA_Vector<double> v_sm;
          TIME(v_sm.gaussianSmearing(v4D2, smearedGauge, nSmear,
                                     alphaGauss));
          PLEGMA_Vector<float> v_sm_f; v_sm_f.copy(v_sm);
          prop_ss.absorb(v_sm_f, isc / 3, isc % 3);
        }

        //   if( rotateQ == true){
        // prop_sl.rotateToPhysicalBase_device(run_mu/abs(run_mu));
        // prop_ss.rotateToPhysicalBase_device(run_mu/abs(run_mu));
        //   }

        // applyBoundaries_device(t0) multiplies ALL propagator elements at
        // global time t < t0 by -1.  This absorbs the anti-periodic phase so
        // that the correlator looks like source at t=0 (shift-invariant).
        //
        // For MESONS (2 quark lines) this is UNNECESSARY: the two propagators
        // each pick up (-1) when crossing the boundary, and (-1)*(-1) = +1
        // cancels in the trace.  (For baryons with 3 quark lines, (-1)^3 = -1
        // does NOT cancel, so applyBoundaries_device would be needed.)
        //
        // Furthermore, applyBoundaries_device must NEVER be called before
        // gauge-dependent operations (Wilson flow Laplacian, covariant
        // derivatives in oneD/fiveD contractions), because the artificial
        // sign discontinuity at t=t0 would corrupt any operator that
        // shifts the field across that timeslice.
        //
        // prop_sl.applyBoundaries_device(src_site[DIM_T]);
        // prop_ss.applyBoundaries_device(src_site[DIM_T]);
      };

      // Per-source suffix for outputs
      char *src_tag = nullptr;
      std::string conf_tag;

      conf_tag = latfile.substr(latfile.length() - 4);
      asprintf(&src_tag, "_sx%02dsy%02dsz%02dst%03d", src[0], src[1], src[2],
               src[3]);
      const std::string twop_out = twop_base + src_tag + "_chi." + conf_tag;
      const std::string threep_out = threep_base + src_tag + "_chi." + conf_tag;

      // Propagator allocation policy (only build what each active channel needs):
      //   prop_q_sl  (+mu_ud, SL): pionQ || kaonQ_uins
      //   prop_q_ss  (+mu_ud, SS): pionQ || kaonQ_uins || kaonQ_sins
      //   prop_s_ss_d(+mu_s,  SS): kaonQ_uins   (kaon 2pt + uins seq smear)
      //   prop_s_sl_d(+mu_s,  SL): kaonQ_sins   (sins fwd line)
      // All as unique_ptr so the destructor (via reset()) truly frees device memory.
      std::unique_ptr<PLEGMA_Propagator<float>> prop_q_sl_ptr, prop_q_ss_ptr;
      std::unique_ptr<PLEGMA_Propagator<float>> prop_s_ss_d_ptr, prop_s_sl_d_ptr;

      // ---- Light sector ----
      if (pionQ || kaonQ_uins) {
        // Both SL and SS needed; build together in one pass.
        prop_q_sl_ptr = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
        prop_q_ss_ptr = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
        print_gpu_mem("before_build_prop_q");
        TIME(build_quark_prop(*prop_q_sl_ptr, *prop_q_ss_ptr, +mu_ud_save, LIGHT,
                              nsmearGauss, src, smearedGauge3D_src));
        print_gpu_mem("after_build_prop_q");
      } else if (kaonQ_sins) {
        // Only SS needed for sins; SL not retained (scoped dummy freed after build).
        prop_q_ss_ptr = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
        print_gpu_mem("before_build_prop_q_ss_only");
        {
          PLEGMA_Propagator<float> dummy_sl(BOTH);
          TIME(build_quark_prop(dummy_sl, *prop_q_ss_ptr, +mu_ud_save, LIGHT,
                                nsmearGauss, src, smearedGauge3D_src));
        }
        print_gpu_mem("after_build_prop_q_ss_only");
      }

      // ---- Strange sector ----
      // kaonQ_uins: SS needed for kaon 2pt and as seq smear source.
      //   SL is discarded via scoped dummy.
      if (kaonQ_uins) {
        prop_s_ss_d_ptr = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
        print_gpu_mem("before_build_prop_s_ss_d");
        {
          PLEGMA_Propagator<float> dummy_sl(BOTH);
          TIME(build_quark_prop(dummy_sl, *prop_s_ss_d_ptr, +mu_s,
                                STRANGE, nsmearGauss_s, src, smearedGauge3D_src));
        }
        print_gpu_mem("after_build_prop_s_ss_d");
      }
      // kaonQ_sins: only SL needed (SS discarded via scoped dummy).
      if (kaonQ_sins) {
        prop_s_sl_d_ptr = std::make_unique<PLEGMA_Propagator<float>>(BOTH);
        print_gpu_mem("before_build_prop_s_sl_d");
        {
          PLEGMA_Propagator<float> dummy_ss(BOTH);
          TIME(build_quark_prop(*prop_s_sl_d_ptr, dummy_ss, +mu_s,
                                STRANGE, nsmearGauss_s, src, smearedGauge3D_src));
        }
        print_gpu_mem("after_build_prop_s_sl_d");
      }

      // ---- 2-point ----
      // Pion: contractMesonsNew(prop_u, prop_u) -- standard pion 2pt with
      //   PLEGMA's built-in γ5-Hermiticity (single-flavor TM).
      // Kaon K+ = u sbar: contractMesonsNew(S_u(+mu_l), S_sbar(+mu_s)).
      //   prop1 is the forward u line; prop2 is conjugated by the kernel and
      //   provides the backward anti-s line via OS gamma5-Hermiticity.
      if (pionQ) {
        PLEGMA_Correlator<float> corr2(corr_space, src, maxQsq);
        TIME(corr2.contractMesonsNew(*prop_q_ss_ptr, *prop_q_ss_ptr));
        TIME(corr2.writeFile(twop_out + "_pion", corr_file_format));
      }
      if (kaonQ_uins) {
        PLEGMA_Correlator<float> corr2(corr_space, src, maxQsq);
        TIME(corr2.contractMesonsNew(*prop_q_ss_ptr, *prop_s_ss_d_ptr));
        TIME(corr2.writeFile(twop_out + "_kaon", corr_file_format));
      }

      // ---- 3-point: sink-sequential through fwd-line (pseudoscalar sink) ----
      // Channels:
      //   pion       : active u(+mu_l), spectator u(+mu_l), seq solver -mu_l
      //   kaon u-ins : active u(+mu_l), spectator sbar(+mu_s), seq solver -mu_l
      //   kaon s-ins : active s(-mu_s), spectator d(-mu_l), seq solver +mu_s

      // ---- One-channel 3pt + Wilson-flow worker ----
      //   tag         : suffix appended to threep_out (e.g. "", "_kaon_uins", "_kaon_sins")
      //   fwd_full    : full forward propagator (will be HOST-backed up here;
      //                 device copy preserved so other channels may reuse it)
      //   fwd_smeared : sink-smeared forward prop (3D slices extracted at sink times)
      //   seq_fl      : flavor used by sequential solver
      //   seq_mu      : mu used by sequential solver (sign already flipped by caller)
      auto run_3pt_channel = [&](const std::string &tag,
                                 PLEGMA_Propagator<float> &fwd_full,
                                 PLEGMA_Propagator<float> &fwd_smeared,
                                 WHICHFLAVOR seq_fl,
                                 double seq_mu,
                                 bool active_as_bwd = false) {
        PLEGMA_printf("\n--- 3pt channel: tag='%s' seq_fl=%s seq_mu=%g active_as_bwd=%d ---\n",
                      tag.c_str(),
                      seq_fl == LIGHT ? "LIGHT"
                                      : (seq_fl == STRANGE ? "STRANGE" : "CHARM"),
                      seq_mu, (int)active_as_bwd);
        updateOptions(seq_fl);
        mu = seq_mu;
        PLEGMA_printf("mu after update option %f", mu);
        solver.UpdateSolver();
        const int seq_nsmear = (seq_fl == LIGHT) ? nsmearGauss : nsmearGauss_s;

        // --- Pre-extract 3D slices for ALL tsinks ---
        std::vector<std::unique_ptr<PLEGMA_Propagator3D<float>>> prop_3D_slices;
        for (size_t i = 0; i < tSinks.size(); ++i) {
          const int st = (src[3] + tSinks[i]) % HGC_totalL[3];
          auto p = std::make_unique<PLEGMA_Propagator3D<float>>();
          p->absorb(fwd_smeared, st);
          prop_3D_slices.push_back(std::move(p));
        }
        // HOST backup of fwd_full (D2H once); device copy left intact for caller.
        fwd_full.unload();
        PLEGMA_Propagator<float> prop_fwd_backup(HOST);
        prop_fwd_backup.copy(fwd_full, HOST);
        print_gpu_mem("after_tsink_presetup");

        for (size_t its = 0; its < tSinks.size(); ++its) {
          const int dt_sink = tSinks[its];
          if (dt_sink >= HGC_totalL[3]) {
            PLEGMA_error("Provided tsink=%d is >= temporal extent", dt_sink);
          }

          const int sink_t = (src[3] + dt_sink) % HGC_totalL[3];

          PLEGMA_printf("AFTER updateGaugeQuda in loop of its");
          gauge.calculatePlaq();
          solver.UpdateSolver();

          // 3D gauge slice at sink time (for seq-source smearing)
          PLEGMA_Gauge3D<double> smearedGauge3D_sink;
          smearedGauge3D_sink.absorb(smearedGauge, sink_t);

          PLEGMA_Propagator3D<float> &prop_q_3D = *prop_3D_slices[its];
          print_gpu_mem("after_prop_q_3D_absorb");

          // ---- Sink-side dressing for the sequential source ----
          prop_q_3D.apply_gamma(G5, LEFT);

          // Build sequential propagator in sub-scope so its dtor frees ~1.22 GB
          PLEGMA_Propagator<float> seq_tmp_a(BOTH);
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
                TIME(seq3D_sm_d.gaussianSmearing(seq3D, smearedGauge3D_sink,
                                                 seq_nsmear, alphaGauss));

                PLEGMA_Vector<double> rhs4D1, rhs4D2;
                rhs4D2.absorb(seq3D_sm_d, sink_t);
                PLEGMA_printf("[SEQDBG] sink_t=%d dt_sink=%d nu=%d c2=%d\n",
                              sink_t, dt_sink, nu, c2);
                PLEGMA_printf("[SEQDBG] rhs4D2.norm=%e\n", rhs4D2.norm());

                const double nrm = rhs4D2.norm();
                if (nrm > 0.0) rhs4D2.scale(1.0 / nrm);
                if (rotateQ)  rhs4D1.rotateToPhysicalBasis(rhs4D2, mu / abs(mu));
                else          rhs4D1.copy(rhs4D2);

                TIME(solver.solve(rhs4D1, rhs4D1));
                if (rotateQ) rhs4D2.rotateToPhysicalBasis(rhs4D1, mu / abs(mu));
                else         rhs4D2.copy(rhs4D1);

                if (nrm > 0.0) rhs4D2.scale(nrm);

                PLEGMA_Vector<float> rhs4D2_f; rhs4D2_f.copy(rhs4D2);
                seqProp.absorb(rhs4D2_f, nu, c2);
              }
            }
            print_gpu_mem("after_seqprop_solve");

            seq_tmp_a.copy(seqProp);
          } // seqProp dtor frees device memory
          print_gpu_mem("after_seqprop_freed");

          // Now contract: all bilinear channels in one shot
          PLEGMA_Correlator<float> corr3(corr_space, src, maxQsq);
          PLEGMA_Correlator<float> corr3_oneD(corr_space, src, maxQsq);

          PLEGMA_Propagator<double> prop_tmp_a(BOTH);
          PLEGMA_Propagator<double> *in = &prop_tmp_a;
          prop_tmp_a.copy(prop_fwd_backup, HOST); // float(H)→double(H)
          prop_tmp_a.load();                       // H→D
          print_gpu_mem("after_fwd_props_loaded");

          PLEGMA_Propagator<double> seq_in_a(BOTH);
          seq_in_a.copy(seq_tmp_a);
          PLEGMA_Propagator<double> *seq_in = &seq_in_a;

          PLEGMA_Gauge<float> contractGauge(BOTH);
          contractGauge.copy(gauge);
          applyBoundaryConditions(contractGauge, true);
          print_gpu_mem("after_input_buf_alloc");

          const double epsilon = 0.01;
          const double t0 = flow_t0;
          const double t_target = 2.5 * t0;
          const int n_steps = (int)std::lround(t_target / epsilon);
          const int n_flow_save = (int)std::lround(t_target / (0.5 * t0));

          double t = 0.0;
          int step = 0;

          // nt = 0 checkpoint (no flow)
          {
            const std::string base = threep_out + tag + "_dt"
                                   + std::to_string(dt_sink) + "_tau00";
            const std::string threep_name_local = base + "_local";
            const std::string threep_name_oneD  = base + "_oneD";
            const std::string threep_patterns   = base + "_patterns";
            PLEGMA_Propagator<float> in_f(BOTH);
            PLEGMA_Propagator<float> seq_in_f(BOTH);
            in_f.copy(*in);
            seq_in_f.copy(*seq_in);
            PLEGMA_Propagator<float> *bwd_contract = &seq_in_f;
            PLEGMA_Propagator<float> *fwd_contract = &in_f;
            if (active_as_bwd) {
                // in_f is used as the backward prop via γ₅-hermiticity, leaving a residual γ₅
                // on its right. Since threep_local transposes the backward prop internally,
                // G5 must be applied on the LEFT after conjugation.
              in_f.conjugate();
              in_f.apply_gamma(G5, LEFT);
              bwd_contract = &in_f;
              fwd_contract = &seq_in_f;
            } else {
                // seq_in_f is the backward prop; same γ₅-hermiticity treatment applies:
                // conjugate first, then apply Gsrc on the LEFT before contraction.
              seq_in_f.conjugate();
              seq_in_f.apply_gamma(G5, LEFT);
            }

            // TIME(corr3.contractNucleonThrp_local(*bwd_contract, *fwd_contract, 0, {G1,G2,G3,G4,G5G4}));
            // TIME(corr3_oneD.contractNucleonThrp_oneD(*bwd_contract, *fwd_contract, contractGauge, 0, {G1,G2,G3,G4,G5G4}));
            TIME(contractNucleonThrp_fiveD_patterns(
                *bwd_contract, *fwd_contract, contractGauge, corr_space, src, maxQsq,
                threep_patterns, max_order));

            // TIME(corr3.writeFile(threep_name_local, corr_file_format));
            // TIME(corr3_oneD.writeFile(threep_name_oneD, corr_file_format));
          }

          // Flow output buffers
          PLEGMA_Propagator<double> prop_tmp_b(BOTH);
          PLEGMA_Propagator<double> *out = &prop_tmp_b;
          PLEGMA_Propagator<double> seq_tmp_b(BOTH);
          PLEGMA_Propagator<double> *seq_out = &seq_tmp_b;
          PLEGMA_Gauge<double> gauge_flowed(BOTH);
          print_gpu_mem("after_flow_buf_alloc");

          for (int ic = 1; ic <= n_flow_save; ++ic) {
            const int target_step = (int)std::lround((double)n_steps * ic / n_flow_save);
            const int chunk_steps = target_step - step;
            if (chunk_steps <= 0) continue;

            TIME(fermionFlow_PLEGMA(*out, *in, *seq_out, *seq_in, gauge_flowed,
                                    gauge, chunk_steps, epsilon, t,
                                    false, false, true, true, 1));
            std::swap(in, out);
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
                // in_f is used as the backward prop via γ₅-hermiticity, leaving a residual γ₅
                // on its right. Since threep_local transposes the backward prop internally,
                // G5 must be applied on the LEFT after conjugation.
                in_f.conjugate();
                in_f.apply_gamma(G5, LEFT);
                bwd_contract = &in_f;
                fwd_contract = &seq_in_f;
              } else {
                // seq_in_f is the backward prop; same γ₅-hermiticity treatment applies:
                // conjugate first, then apply Gsrc on the LEFT before contraction.
                seq_in_f.conjugate();
                seq_in_f.apply_gamma(G5, LEFT);
              }

              // TIME(corr3.contractNucleonThrp_local(*bwd_contract, *fwd_contract, 0, {G1,G2,G3,G4,G5G4}));
              // TIME(corr3_oneD.contractNucleonThrp_oneD(*bwd_contract, *fwd_contract, contractGauge, 0, {G1,G2,G3,G4,G5G4}));
              TIME(contractNucleonThrp_fiveD_patterns(
                  *bwd_contract, *fwd_contract, contractGauge, corr_space, src, maxQsq,
                  threep_patterns, max_order));

              // TIME(corr3.writeFile(threep_name_local, corr_file_format));
              // TIME(corr3_oneD.writeFile(threep_name_oneD, corr_file_format));
            }
          } // checkpoint loop
        } // tsink loop
      }; // run_3pt_channel

      // ---- Dispatch channels ----
      const double seq_mu_light = -std::abs(mu_ud_save);

      if (pionQ) {
        run_3pt_channel("_pion_uins", *prop_q_sl_ptr, *prop_q_ss_ptr, LIGHT, seq_mu_light);
      }
      if (kaonQ_uins) {
        run_3pt_channel("_kaon_uins", *prop_q_sl_ptr, *prop_s_ss_d_ptr, LIGHT, seq_mu_light);
      }
      if (kaonQ_sins) {
        const double seq_mu_s = -std::abs(mu_s);
        run_3pt_channel("_kaon_sins", *prop_s_sl_d_ptr, *prop_q_ss_ptr, STRANGE, seq_mu_s, true);
      }

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
