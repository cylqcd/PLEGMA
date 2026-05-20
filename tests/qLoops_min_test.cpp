#include <PLEGMA.h>
#include <PLEGMA_utils.h>

#include <mpi.h>
#include <string>
#include <vector>

// #include "higher_derivative_patterns.h"
#include "PLEGMA_QLoops_patterns.h"
#include "PLEGMA_global.h"
#include "enum_quda.h"
#include "quda.h"

using namespace plegma;
using namespace quda;

// Use templated declaration from header
#include "PLEGMA_fiveD.h"

static std::vector<std::string> listOpt = {
    "verbosity",
    "load-gauge",
    "nsrc",
    "maxQsq",
    "rng-seed",
    "corr-file-format",
    "quark-flavor"
};

template<typename Float>
static void dumpLoops(PLEGMA_QLoops<Float>& qLoops,
                      PLEGMA_FT<Float>* ft[2],
                      std::string filenamePrefix,
                      std::string confID,
                      FILE_FORMAT format,
                      int isc = -1)
{
  using sv = std::vector<std::string>;
  std::string suffix = (format == ASCII_FORMAT) ? ".dat" : ".h5";

  std::string fname_base;
  if (isc >= 0)
    fname_base = join(sv({"Conf" + confID, "Ns" + std::to_string(isc)}), "/");
  else
    fname_base = join(sv({"Conf" + confID}), "/");

  qLoops.load(qLoops.H_loc());
  ft[0]->apply(qLoops, FT_GEMV);
  std::string fnameUl = fname_base + join(sv({"localLoops", "loop"}), "/");
  ft[0]->writeFile((format == HDF5_FORMAT)
                       ? filenamePrefix + suffix + fnameUl
                       : filenamePrefix + findAndReplace(fnameUl, '/', '_') + suffix,
                   format);

  if (qLoops.IsOneD())
    for (int mu = 0; mu < N_DIMS; mu++) {
      std::string fnameOneD  = fname_base + join(sv({"oneD",  "dir" + std::to_string(mu), "loop"}), "/");
      std::string fnameOneDC = fname_base + join(sv({"oneDC", "dir" + std::to_string(mu), "loop"}), "/");

      qLoops.load(qLoops.H_oneD()[mu]);
      ft[0]->apply(qLoops, FT_GEMV);
      ft[0]->scale(0.25);
      ft[0]->writeFile((format == HDF5_FORMAT)
                           ? filenamePrefix + suffix + fnameOneD
                           : filenamePrefix + findAndReplace(fnameOneD, '/', '_') + suffix,
                       format);

      qLoops.load(qLoops.H_oneDC()[mu]);
      ft[0]->apply(qLoops, FT_GEMV);
      ft[0]->scale(0.25);
      ft[0]->writeFile((format == HDF5_FORMAT)
                           ? filenamePrefix + suffix + fnameOneDC
                           : filenamePrefix + findAndReplace(fnameOneDC, '/', '_') + suffix,
                       format);
    }

  int count = 0;
  if (qLoops.IsTwoD())
    for (auto munu : qLoops.get_twoD_index()) {
      int mu = std::get<0>(munu), nu = std::get<1>(munu);
      std::string fnameTwoD = fname_base + join(sv({"twoD", "dirs" + std::to_string(mu) + std::to_string(nu), "loop"}), "/");

      qLoops.load(qLoops.H_twoD()[count]);
      ft[1]->apply(qLoops, FT_GEMV);
      if (mu != 3 && nu != 3)
        ft[1]->scale(0.25);
      else
        ft[1]->scale(0.125);
      ft[1]->writeFile((format == HDF5_FORMAT)
                           ? filenamePrefix + suffix + fnameTwoD
                           : filenamePrefix + findAndReplace(fnameTwoD, '/', '_') + suffix,
                       format);
      count++;
    }
}

int main(int argc, char **argv)
{
  initializeOptions(argc, argv, true, listOpt);

  bool oneDLoops = true;
  bool twoDLoops = true;
  std::string loopsPrefix = "./qloops_min_out/";
  HGC_options->set("oneD-loops", "Compute oneD loops", verbosity, oneDLoops);
  HGC_options->set("twoD-loops", "Compute twoD loops", verbosity, twoDLoops);
  HGC_options->set("output-path", "Output directory", verbosity, loopsPrefix);

  initializePLEGMA();

  // ---- Gauge ----
  PLEGMA_Gauge<double> gauge;
  gauge.readFile(latfile, LIME_FORMAT);
  gauge.calculatePlaq();

  initGaugeQuda(gauge, true);
  plaqQuda();

  // Apply boundary conditions (needed for covariant derivative)
  applyBoundaryConditions(gauge, true);

  // ---- Solver + Dirac ----
  if (mu > 0) mu *= -1.0;
  QUDA_solver *solverDN = new QUDA_solver(mu);
  QudaInvertParam inv_params = solverDN->getInvParams();

  QUDA_dirac *D = nullptr;
  if (inv_params.dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
    D = new QUDA_dirac(QUDA_CLOVER_WILSON_DSLASH);
  else if (inv_params.dslash_type == QUDA_TWISTED_MASS_DSLASH)
    D = new QUDA_dirac(QUDA_WILSON_DSLASH);
  else
    PLEGMA_error("Only QUDA_TWISTED_CLOVER_DSLASH and QUDA_TWISTED_MASS_DSLASH supported");

  // ---- FT ----
  PLEGMA_FT<double> *ft[2] = {nullptr, nullptr};
  ft[0] = new PLEGMA_FT<double>(maxQsq, 3);
  if (twoDLoops)
    ft[1] = new PLEGMA_FT<double>(0, 3);

  PLEGMA_FT<float> *ft_f[2] = {nullptr, nullptr};
  ft_f[0] = new PLEGMA_FT<float>(maxQsq, 3);

  // ---- QLoops: standard oneEnd_trick (oneD + twoD) ----
  PLEGMA_QLoops<double> qloops_std(BOTH, NO_GHOSTS, true, oneDLoops, twoDLoops);
  PLEGMA_QLoops<double> qloops_gen(BOTH, NO_GHOSTS, true, oneDLoops, twoDLoops);

  // ---- tmp buffers for oneD/twoD ----
  PLEGMA_Vector<double> *tmp[16] = {};
  PLEGMA_QLoops<double> *qLtmp = nullptr;
  if (oneDLoops)
    tmp[0] = new PLEGMA_Vector<double>(DEVICE);
  if (twoDLoops) {
    for (int i = 1; i < 16; i++)
      tmp[i] = new PLEGMA_Vector<double>(DEVICE);
    qLtmp = new PLEGMA_QLoops<double>(DEVICE, FIRST_SIDE, true);
  }

  // ---- QLoops patterns (higher-order derivatives) ----
  PLEGMA_QLoops_patterns<float> qloops_pat_std(BOTH);
  PLEGMA_QLoops_patterns<float> qloops_pat_gen(BOTH);

  // ---- Gauge in float for covariant_derivative ----
  PLEGMA_Gauge<float> gauge_f;
  gauge_f.copy(gauge);
  gauge_f.communicateGhost();

  // ---- confID from latfile ----
  size_t pos = latfile.find("conf.");
  if (pos == std::string::npos) pos = latfile.find("Conf.");
  std::string confID = (pos != std::string::npos)
                           ? latfile.substr(pos + 5)
                           : "0000";

  // ---- Source + solve + contract ----
  PLEGMA_Vector<double> source(BOTH);
  PLEGMA_Vector<double> phi(BOTH);
  PLEGMA_Vector<double> phi_r(BOTH);

  source.randInit(rng_seed);

  for (int isc = 0; isc < numSourcePositions; ++isc) {
    source.stochastic_Z(4);
    // phi.stochastic_Z(4);

    // Solve
    solverDN->solve(phi, source);

    // Scale
    PLEGMA_Vector<double> phi_contr(BOTH);
    phi_contr.copy(phi);
    phi_contr.scale(1.0 / (2.0 * inv_params.kappa));

    // ========== Standard oneEnd_trick (local + oneD + twoD) ==========
    qloops_std.clearAccumBuffs();
    qloops_gen.clearAccumBuffs();

    if (oneDLoops || twoDLoops)
      qloops_std.oneEnd_trick(phi_contr, phi_contr, tmp, qLtmp, gauge, -1.0, true);
    else
      qloops_std.oneEnd_trick(phi_contr, phi_contr, -1.0, true);

    // D->apply<M>(phi_r.D_elem(), phi_contr.D_elem());
    // phi_r.apply_gamma5();

    // if (oneDLoops || twoDLoops)
    //   qloops_gen.oneEnd_trick(phi_contr, phi_r, tmp, qLtmp, gauge, +1.0, true);
    // else
    //   qloops_gen.oneEnd_trick(phi_contr, phi_r, +1.0, true);

    dumpLoops(qloops_std, ft, loopsPrefix + "stoch_part_std", confID, corr_file_format, isc);
    // dumpLoops(qloops_gen, ft, loopsPrefix + "stoch_part_gen", confID, corr_file_format, isc);

    // ========== Higher-order derivative patterns ==========
    // Cast to float for covariant_derivative
    PLEGMA_Vector<float> phi_contr_f(BOTH, FIRST_SIDE);
    PLEGMA_Vector<float> phi_r_f(BOTH, FIRST_SIDE);
    phi_contr_f.copy(phi_contr);
    phi_r_f.copy(phi_r);
    phi_contr_f.communicateGhost();
    phi_r_f.communicateGhost();

    std::string ns_tag = "Ns" + std::to_string(isc);

    // std part: x_l = phi_contr, x_r = phi_contr
    contractG5_fiveD_patterns(phi_contr_f, phi_contr_f, gauge_f,
                              qloops_pat_std, *ft_f[0], -1.0f);
    qloops_pat_std.batchFlush(*ft_f[0], loopsPrefix + "pat_stoch_part_std",
                              confID, corr_file_format, isc);

    // // gen part: x_l = phi_contr, x_r = gamma5 D phi_contr
    // contractG5_fiveD_patterns(phi_contr_f, phi_r_f, gauge_f,
    //                           qloops_pat_gen, *ft_f[0], +1.0f);
    // qloops_pat_gen.batchFlush(*ft_f[0], loopsPrefix + "pat_stoch_part_gen",
    //                           confID, corr_file_format, isc);

    PLEGMA_printf("Source %d/%d done\n", isc + 1, numSourcePositions);
  }

  // ---- Cleanup ----
  if (oneDLoops)
    delete tmp[0];
  if (twoDLoops) {
    for (int i = 1; i < 16; i++)
      delete tmp[i];
    delete qLtmp;
    delete ft[1];
  }
  delete ft[0];
  delete ft_f[0];
  delete D;
  delete solverDN;

  finalize();
  return 0;
}
