#include <PLEGMA.h>
#include <PLEGMA_Hprobing.h>
#include <PLEGMA_utils.h>
#include <stdio.h>
#include <string>
#include <memory>

#include "PLEGMA_global.h"
#include "global/PLEGMA_prints.hpp"
#include "QUDA_wilson_flow.h"
#include "PLEGMA_QLoops_patterns.h"
#include "PLEGMA_QLoops_fast.h"
#include "PLEGMA_fiveD.h"
#include "PatternsHelper.h"
#include "qloops_utils.h"


using namespace plegma;
using namespace quda;

static std::vector<std::string> listOpt = {"verbosity", "load-gauge",  "Eig-isACC", "Eig-PolyDeg",      "Eig-amin",    "Eig-amax", "Eig-spectrumPart", "Eig-tol", "Eig-maxIters", "Eig-NeV",
#if defined(HAVE_ARPACK) || defined(QUDAEIG)
                                           "Eig-NkV",   "Eig-logFile",
#elif defined(HAVE_PRIMME)
					   "Eig-printLevel", "Eig-method-PRIMME",
#endif
                                           "nsrc",      "maxQsq",      "rng-seed",  "corr-file-format", "quark-flavor",
                                           "pat-double"};

int main(int argc, char** argv)
{
    initializeOptions(argc, argv, true, listOpt);  // Put list of options later
    //================ Add your options in this between initializeOptions and
    // initializePLEGMA ================//
    int  k_probing     = 0;
    bool spinColorDil  = false;
    bool lowModesRecon = false;
    HGC_options->set("k-probing",
                     "Hierarchical probing, with distance D=2**k "
                     "(Options:0,1,2,3,...) (0 means No probing)",
                     verbosity, k_probing);
    int         hadamLow    = 0;
    int         Nhadam      = (k_probing > 0) ? 2 * std::pow(2, N_DIMS * (k_probing - 1)) : 1;
    int         hadamHgh    = Nhadam;
    std::string loopsPrefix = "./";
    HGC_options->set("output-path", "Path to the directory to dump results", verbosity, loopsPrefix);
    // Usefull if one wants to go up to high distance in probing and walltime does
    // not allow to do all Hadamard vectors
    HGC_options->set("hadamard-low", "From which Hadamard vector to start (Options:[0,max))", verbosity, hadamLow);
    HGC_options->set("hadamard-high",
                     "Up to which Hadamard vector to stop (Options: 0>= , <=max) "
                     "(default max)",
                     verbosity, hadamHgh);
    if ((k_probing > 0) && (hadamLow < 0 || hadamHgh < 0))
        PLEGMA_error("Negative values for number of Hadamard vector not allowed");
    if ((k_probing > 0) && (hadamLow > hadamHgh))
        PLEGMA_error("hadamard-high should be > hadamard-low");
    if ((k_probing > 0) && (hadamHgh > Nhadam))
        PLEGMA_error("hadamard-high should be <= from max number of Hadamard vectors");
    HGC_options->set("spin-color-dil", "Whether we want spin color dilution", verbosity, spinColorDil);
    int Nsc = spinColorDil ? N_SPINS * N_COLS : 1;
    bool patDoublePrecision = false;
    HGC_options->set("pat-double", "Use double precision for higher-order derivative patterns (default: float)", verbosity, patDoublePrecision);
    HGC_options->set("low-modes-recon",
                     "Whether we want to use low modes of the operator to "
                     "reconstruct part of the quark loop",
                     verbosity, lowModesRecon);
    if (!lowModesRecon)
        Eig_NeV = 0;
    std::string Eig_outputFile = "./eigsVdagG5V.dat";
    HGC_options->set("Eig-outputFile",
                     "Path to dump the eigenvalues and vdag g5 v if "
                     "low-modes-recon is enabled",
                     verbosity, Eig_outputFile);
    bool        isReadEigenVecs = false, isWriteEigenVecs = false;
    std::string fnameEigenVecsPrefix = "";
    HGC_options->set("readEigenVectors", "Where we want to read EigenVectors from file", verbosity, isReadEigenVecs);
    HGC_options->set("writeEigenVectors", "Where we want to read EigenVectors from file", verbosity, isWriteEigenVecs);
    HGC_options->set("prefixEigenVecsFile", "Path with prefix for the filenames of the eigenvectors", verbosity, fnameEigenVecsPrefix);
#ifdef QUDAEIG
    int batched_rotate = 1;
    HGC_options->set("batched-rotate", "The size of the batch during Ritz rotation", verbosity, batched_rotate);
#endif

    bool oneDLoops = false;
    bool accumFlag = true;
    bool twoDLoops = false;
    int  NdumpStep = 1;
    HGC_options->set("oneD-loops",
                     "Whether we want to use covariant derivative for the quark "
                     "loops calculation",
                     verbosity, oneDLoops);
    HGC_options->set("twoD-loops",
                     "Whether we want to use two covariant derivative for the "
                     "quark loops calculation",
                     verbosity, twoDLoops);
    HGC_options->set("accum-loops", "Accumulate loops over the stochastic source vectors", verbosity, accumFlag);
    HGC_options->set("dump-step", "If accumulation is ON, Every how many stochastic vector to dump results", verbosity, NdumpStep);
    bool debugMode = false;
    HGC_options->set("debug-mode", "If debug mode is enabled, run 1 source with units everywhere for check", verbosity, debugMode);

    //=========================================================================================================//
    initializePLEGMA();

    if (!accumFlag)
        NdumpStep = 1;
    if (accumFlag && (NdumpStep < 1))
        PLEGMA_error("dump-step should be >= 1");

    if (Eig_NeV <= 0 && lowModesRecon)
    {
        PLEGMA_warning(
            "You enabled Low modes reconstruction but NeV is <= 0. "
            "Switching off Low modes reconstruction");
        lowModesRecon = false;
    }

    if (!lowModesRecon && Eig_NeV > 0)
    {
        PLEGMA_warning(
            "The Low modes reconstruction is off forcing number of "
            "eigenvalues to zero");
        Eig_NeV = 0;
    }

    // Reading from Lime file and loading to device
    PLEGMA_Gauge<double> gauge;
    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();

    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauge, true);
    plaqQuda();
    PLEGMA_Gauge<double> gauge_flowed;

    // apply boundary conditions since is needed for the covariant derivative
    // this needs to be done after initGaugeQuda otherwise causes troubles
    // applyBoundaryConditions(gauge, true);

    // TODO: LMR (Low Modes Reconstruction) not yet implemented for this driver.
    // Open questions: (1) whether LMR is valid after Wilson flow; (2) how to
    // compute and apply the LMR exact part consistently at each flow checkpoint.
    if (lowModesRecon)
        PLEGMA_error("lowModesRecon is not yet supported in qLoops_patterns (TODO)");

    PLEGMA_FT<double>* ft[2] = {nullptr, nullptr};
    ft[0]                    = new PLEGMA_FT<double>(maxQsq, 3);
    if (twoDLoops)
        ft[1] = new PLEGMA_FT<double>(0, 3);

    PLEGMA_QLoops<double> qloops_std(BOTH, NO_GHOSTS, true, oneDLoops, twoDLoops);
    PLEGMA_QLoops<double> qloops_gen(BOTH, NO_GHOSTS, true, oneDLoops, twoDLoops);
    PLEGMA_QLoops<double> qloops_direct(BOTH, NO_GHOSTS, true, oneDLoops, twoDLoops);

    // ---- Higher-order derivative patterns (float or double precision) ----
    std::unique_ptr<PatternsHelper<float>>  patHelperF;
    std::unique_ptr<PatternsHelper<double>> patHelperD;
    if (patDoublePrecision)
        patHelperD = std::make_unique<PatternsHelper<double>>(maxQsq);
    else
        patHelperF = std::make_unique<PatternsHelper<float>>(maxQsq);

    // // ensuring mu negative
    if (mu > 0)
        mu *= -1.;
    PLEGMA_printf("muuuuuuuuuuu%f", mu);
    QUDA_solver*    solverDN   = new QUDA_solver(mu);
    QudaInvertParam inv_params = solverDN->getInvParams();

    QUDA_dirac* D = nullptr;
    if (inv_params.dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
        D = new QUDA_dirac(QUDA_CLOVER_WILSON_DSLASH);
    else if (inv_params.dslash_type == QUDA_TWISTED_MASS_DSLASH)
        D = new QUDA_dirac(QUDA_WILSON_DSLASH);
    else
        PLEGMA_error(
            "Only QUDA_TWISTED_CLOVER_DSLASH and QUDA_TWISTED_MASS_DSLASH "
            "are allowed for the one-end trick");

    PLEGMA_Vector<double>  phi;
    PLEGMA_Vector<double>  phi_flowed;
    PLEGMA_Vector<double>  phi_r;
    PLEGMA_Vector<double>* tmp[16] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    PLEGMA_QLoops<double>* qLtmp   = nullptr;
    if (oneDLoops)
        tmp[0] = new PLEGMA_Vector<double>(DEVICE);
    if (twoDLoops)
    {
        for (int i = 1; i < 16; i++)
            tmp[i] = new PLEGMA_Vector<double>(DEVICE);
        qLtmp = new PLEGMA_QLoops<double>(DEVICE, FIRST_SIDE, true);  // this we need to do the shifts where needed
    }

    // PLEGMA_Vector<double> source(DEVICE);
    // PLEGMA_Vector<double> source(BOTH), source_flowed(BOTH);
    // PLEGMA_Vector<double> rng_warmup(DEVICE);
    // PLEGMA_Vector<double>*source_to_contract, phi_to_contract;
    // if (!debugMode)
        // source.randInit(rng_seed);
        // rng_warmup.randInit(rng_seed);
    PLEGMA_Vector<double>* sourceDil = nullptr;
    if (k_probing > 0 || spinColorDil)
        sourceDil = new PLEGMA_Vector<double>(DEVICE);

    if (oneDLoops)
        gauge.communicateGhost();

        // TODO: LMR exact part (see TODO above)

    std::size_t foundPos  = latfile.find("conf.");
    std::size_t foundPos2 = latfile.find("conf_lgfix.");
    if (foundPos == std::string::npos && foundPos2 == std::string::npos)
        PLEGMA_error(
            "Cannot find (conf.) or (conf_lgfix.) in configuration path "
            "to get confID");
    std::string confID;
    if (foundPos != std::string::npos)
        confID = latfile.substr(foundPos + 5, latfile.length());
    else if (foundPos2 != std::string::npos)
        confID = latfile.substr(foundPos2 + 5, latfile.length());
    else
        PLEGMA_error("Cannot happen to reach this error");

    PLEGMA_Hprobing* hprop = nullptr;
    if (k_probing > 0)
        hprop = new PLEGMA_Hprobing(k_probing);

    std::vector<int> indDof = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    // std::cout<<"numSourcePositions: "<<numSourcePositions<<std::endl;
    PLEGMA_printf("numSourcePositions: %d ", numSourcePositions);
    // const double eps = 0.01;

    // const std::vector<int> flow_steps = {0, 1, 10};
    // // const std::vector<int> flow_steps = { 10};

const double eps = 0.01;
// const std::vector<int> checkpoints = {0, 1, 10};     // any non-decreasing list
// const std::vector<int> checkpoints = {0,1, 2, 3 ,4,5,6,7,8,9,10};     // any non-decreasing list
const std::vector<int> checkpoints = {0,25,50,75,100};     // any non-decreasing list
const int tf = checkpoints.back();
const int rhs_block_user = 1;                       // tune if you want

auto is_checkpoint = [&](int s) {
  return std::find(checkpoints.begin(), checkpoints.end(), s) != checkpoints.end();
};

const int Ns = numSourcePositions;

// ---- RNG init ONCE ----
PLEGMA_Vector<double> source(BOTH);
if (!debugMode) source.randInit(rng_seed);

PLEGMA_printf("Flow: Ns=%d, Nsc=%d, Nhadam=%d", Ns, Nsc, Nhadam);

// Simple scoped timer
static std::vector<double> runtime;
#define TIME(fnc)                                                              \
  runtime.push_back(MPI_Wtime());                                              \
  fnc;                                                                         \
  PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back()); \
  runtime.pop_back()

// nv = Nsc: one source at a time, all Nsc dilution components flowed together
const int nv = Nsc;

// ---- Loop over stochastic sources (one at a time) ----
for (int isrc = 0; isrc < Ns; isrc++) {
  if (HGC_fullRank == 0) PLEGMA_printf("\n=== Source isrc=%d/%d ===\n", isrc, Ns);

  // Nsc solution and source vectors for this stochastic source
  std::vector<std::unique_ptr<PLEGMA_Vector<double>>> phis(nv), srcs(nv);
  for (int i = 0; i < nv; i++) {
    phis[i] = std::make_unique<PLEGMA_Vector<double>>(BOTH);
    srcs[i] = std::make_unique<PLEGMA_Vector<double>>(BOTH);
  }

  // Single Z4 noise vector for this source
  PLEGMA_Vector<double> noise(BOTH);
  if (debugMode) source.setUnit(indDof);
  else source.stochastic_Z(4);
  noise.copy(source);

  // Ping-pong buffers (Nsc vectors each)
  std::vector<std::unique_ptr<PLEGMA_Vector<double>>> phi_a(nv), phi_b(nv);
  std::vector<std::unique_ptr<PLEGMA_Vector<double>>> src_a(nv), src_b(nv);
  for (int i = 0; i < nv; i++) {
    phi_a[i] = std::make_unique<PLEGMA_Vector<double>>(BOTH);
    phi_b[i] = std::make_unique<PLEGMA_Vector<double>>(BOTH);
    src_a[i] = std::make_unique<PLEGMA_Vector<double>>(BOTH);
    src_b[i] = std::make_unique<PLEGMA_Vector<double>>(BOTH);
  }

  // Pointer lists for fermionFlow_PLEGMA (phi + src, 2*Nsc total)
  std::vector<PLEGMA_Vector<double>*> in_all(2 * nv), out_all(2 * nv);

  auto bind_ptrs = [&](bool in_is_a) {
    for (int i = 0; i < nv; i++) {
      if (in_is_a) {
        in_all[i]      = phi_a[i].get();  out_all[i]      = phi_b[i].get();
        in_all[nv + i] = src_a[i].get();  out_all[nv + i] = src_b[i].get();
      } else {
        in_all[i]      = phi_b[i].get();  out_all[i]      = phi_a[i].get();
        in_all[nv + i] = src_b[i].get();  out_all[nv + i] = src_a[i].get();
      }
    }
  };

  // isrc_global = ih * Ns + isrc  (ih=0 when k_probing=0, unchanged behaviour)
  int ih_current = hadamLow;
  auto contract_batch = [&](int step, bool in_is_a) {
    auto &phi_now = in_is_a ? phi_a : phi_b;
    auto &src_now = in_is_a ? src_a : src_b;
    std::ostringstream tag;
    tag << loopsPrefix << "t" << std::setw(3) << std::setfill('0') << step;
    int isrc_global = ih_current * Ns + isrc;

    // --- Standard oneEnd_trick: phi x phi ---
    qloops_std.clearAccumBuffs();
    {
      PLEGMA_Vector<double> phi_contr(BOTH);
      TIME(for (int isc = 0; isc < Nsc; ++isc) {
        phi_contr.copy(*phi_now[isc]);
        phi_contr.scale(1.0 / (2.0 * inv_params.kappa));
        if (oneDLoops || twoDLoops)
          qloops_std.oneEnd_trick(phi_contr, phi_contr, tmp, qLtmp, gauge_flowed, -1.0, true);
        else
          qloops_std.oneEnd_trick(phi_contr, phi_contr, -1.0, true);
      });
    }
    TIME(dumpLoops(qloops_std, ft, tag.str() + "stoch_part_std", confID, corr_file_format, isrc_global));

    // --- Direct contraction: flowed_src x flowed_phi ---
    qloops_direct.clearAccumBuffs();
    {
      PLEGMA_Vector<double> phi_contr(BOTH);
      PLEGMA_Vector<double> src_contr(BOTH);
      TIME(for (int isc = 0; isc < Nsc; ++isc) {
        phi_contr.copy(*phi_now[isc]);
        phi_contr.scale(1.0 / (2.0 * inv_params.kappa));
        src_contr.copy(*src_now[isc]);
        if (oneDLoops || twoDLoops)
          qloops_direct.oneEnd_trick(src_contr, phi_contr, tmp, qLtmp, gauge_flowed, -1.0, true);
        else
          qloops_direct.oneEnd_trick(src_contr, phi_contr, -1.0, true);
      });
    }
    TIME(dumpLoops(qloops_direct, ft, tag.str() + "stoch_part_direct", confID, corr_file_format, isrc_global));

    // --- Higher-order derivative patterns: std (phi x phi) ---
    if (patHelperD) {
      TIME(patHelperD->contractAndFlush(phi_now, 0, Nsc, gauge_flowed, D, inv_params,
                                        tag.str(), confID, corr_file_format, isrc_global));
    } else {
      TIME(patHelperF->contractAndFlush(phi_now, 0, Nsc, gauge_flowed, D, inv_params,
                                        tag.str(), confID, corr_file_format, isrc_global));
    }

    // --- Higher-order derivative patterns: direct (src x phi) ---
    if (patHelperD) {
      TIME(patHelperD->contractAndFlush(src_now, phi_now, 0, Nsc, gauge_flowed, D, inv_params,
                                        tag.str(), confID, corr_file_format, isrc_global));
    } else {
      TIME(patHelperF->contractAndFlush(src_now, phi_now, 0, Nsc, gauge_flowed, D, inv_params,
                                        tag.str(), confID, corr_file_format, isrc_global));
    }
  };

  // ---- Hadamard outer loop ----
  // k_probing=0: Nhadam=1, ih=0 → runs once, isrc_global = isrc (unchanged)
  // k_probing=1: Nhadam=2 → 2 passes with different Hadamard coloring
  for (int ih = hadamLow; ih < hadamHgh; ih++) {
    ih_current = ih;
    if (HGC_fullRank == 0)
      PLEGMA_printf("\n--- Hadamard ih=%d/%d (isrc=%d) ---\n", ih, hadamHgh, isrc);

    // Dilute + Hadamard coloring + invert (all Nsc components)
    for (int isc = 0; isc < Nsc; ++isc) {
      if (spinColorDil) {
        srcs[isc]->dilutespincolor(noise, isc / N_COLS, isc % N_COLS);
        if (k_probing > 0) srcs[isc]->applyHpropColoring4D(*srcs[isc], *hprop, ih, indDof);
      } else {
        srcs[isc]->copy(noise);
        if (k_probing > 0) srcs[isc]->applyHpropColoring4D(*srcs[isc], *hprop, ih, indDof);
      }
      solverDN->solve(*phis[isc], *srcs[isc]);
    }

    // Copy solutions/sources to ping-pong buffers
    for (int i = 0; i < nv; i++) phi_a[i]->copy(*phis[i]);
    for (int i = 0; i < nv; i++) src_a[i]->copy(*srcs[i]);

    // ---- Flow + checkpoint loop ----
    int step = 0;
    double t = 0.0;
    bool in_is_a = true;

    gauge_flowed.copy(gauge);
    applyBoundaryConditions(gauge_flowed, true);
    if (is_checkpoint(0)) contract_batch(0, true);

    for (size_t k = 1; k < checkpoints.size(); k++) {
      const int target = checkpoints[k];
      const int delta  = target - step;
      if (delta < 0) PLEGMA_error("checkpoints must be non-decreasing");

      if (delta > 0) {
        bind_ptrs(in_is_a);

        const bool is_last_call_in_batch = (k == checkpoints.size() - 1);
        const bool keep_smeared = !is_last_call_in_batch;
        // Flow both phi and src vectors (2*Nsc RHS)
        TIME(fermionFlow_PLEGMA(out_all, in_all,
                                gauge_flowed, gauge,
                                delta, eps, t,
                                false, false,
                                true,
                                keep_smeared,
                                rhs_block_user));

        in_is_a = !in_is_a;
        step = target;
        t += delta * eps;
      }

      if (is_checkpoint(step)) contract_batch(step, in_is_a);
    }
  } // end for ih

  // Nsc vectors go out of scope here -> GPU memory freed before next source
} // end for isrc

    if (k_probing > 0)
        delete hprop;
    if (k_probing > 0 || spinColorDil)
        delete sourceDil;

    delete ft[0];
    if (oneDLoops)
        delete tmp[0];
    if (twoDLoops)
    {
        for (int i = 1; i < 16; i++)
            delete tmp[i];
        delete qLtmp;
        delete ft[1];
    }

    delete D;
    delete solverDN;
    finalize();

    return 0;
}
