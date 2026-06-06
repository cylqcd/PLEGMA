// #include "global/PLEGMA_prints.hpp"
#include "blas_quda.h"
#include <PLEGMA.h>
#include <PLEGMA_BLAS.h>
#include <PLEGMA_utils.h>
#include <invert_quda.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <gauge_tools.h>
#include <dslash_quda.h>
#include <util_quda.h>
#include "enum_quda.h"
#include <QUDA_wilson_flow.h>
#include <PLEGMA_prop_quda_offset.h>
// #include <reference_wrapper_helper.h>

using namespace std;
using namespace quda;

using quda::ColorSpinorField;
using quda::vector_ref;
using quda::cvector_ref;
using quda::lat_dim_t;
static lat_dim_t R = {1,1,1,1};

extern quda::GaugeField *gaugePrecise;
extern quda::GaugeField *gaugeSmeared;

static TimeProfile profileFermionFlow("FermionFlowPLEGMA");
static TimeProfile profileAdjointFermionFlow("AdjointFermionFlow_PLEGMA");

QudaInvertParam make_flow_inv_param(QudaPrecision prec)
{
  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  inv_param.cuda_prec = prec;
  inv_param.cuda_prec_sloppy = prec;
  return inv_param;
}

void resetFermionFlowSmearedGauge()
{
  if (gaugeSmeared) {
    freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
    gaugeSmeared = nullptr;
  }
}


void wilsonFlow_QUDA(PLEGMA_Gauge<double> &gaugeOut,
                            PLEGMA_Gauge<double> &gaugeIn,
                            int n_steps, double epsilon, double t0_start,
                            bool compute_plaquette, bool compute_qcharge) {
  if (n_steps < 0) n_steps = 0;

  // Upload input as QUDA resident (precise)
  // updateGaugeQuda(gaugeIn, false);

  // Host↔device metadata for saves
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  gauge_param.make_resident_gauge = 0;
  gauge_param.t_boundary =  QUDA_PERIODIC_T; 

  auto do_one_step = [&](double t0, bool restart_flag) {
    QudaGaugeSmearParam sp = newQudaGaugeSmearParam();
    sp.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;
    sp.n_steps       = 1;                // advance exactly one step per call
    sp.epsilon       = epsilon;
    sp.t0            = t0;
    sp.restart       = restart_flag ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    sp.meas_interval = 2;                // n_steps(=1) + 1

    QudaGaugeObservableParam op = newQudaGaugeObservableParam();
    op.compute_plaquette = compute_plaquette ? QUDA_BOOLEAN_TRUE :
    QUDA_BOOLEAN_FALSE; op.compute_qcharge   = compute_qcharge   ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;

    performWFlowQuda(&sp, &op);
  };

  // Step 0
  do_one_step(t0_start, false);

  // Steps 1..n_steps-1
  for (int step = 1; step < n_steps; ++step) {
    const double t0 = t0_start + step * epsilon;
    do_one_step(t0, true);

    // Save smeared gauge right before the final step (to mirror PyQUDA)
    if (step == n_steps - 1) {
      QudaLinkType old_type = gauge_param.type;
      QudaReconstructType old_recon = gauge_param.reconstruct;

      double *new_gauge[N_DIMS];
      // PLEGMA_printf(N_DIMS);
        PLEGMA_printf("DIM to invert %d for all component\n",N_DIMS);
      for (int mu = 0; mu < N_DIMS; ++mu)
        hostMalloc(new_gauge[mu], gaugeIn.Bytes_total() / N_DIMS);

      gauge_param.type = QUDA_SMEARED_LINKS;
      gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
      saveGaugeQuda(new_gauge, &gauge_param);
      gauge_param.type = QUDA_WILSON_LINKS;

      packGaugeToNormal(gaugeOut, new_gauge);
      gaugeOut.load();
      gaugeOut.calculatePlaq();
      //  initGaugeQuda(gaugeOut, false);
      // plaqQuda();
      // gaugeOut.load();

      for (int mu = 0; mu < N_DIMS; ++mu)
        hostFree(new_gauge[mu], gaugeIn.Bytes_total() / N_DIMS);

    }
  }

  // Final step (no save, matching PyQUDA behavior)
  {
    const double t0_final = t0_start + n_steps * epsilon;
    do_one_step(t0_final, true);
  }
}

void applyLaplace_PLEGMA(plegma::PLEGMA_Vector<double> &spinorOut,
                         plegma::PLEGMA_Vector<double> &spinorIn,
                         QudaInvertParam              &inv_param,
                         int dir,              // 省略的方向
                         double a, double b)   // 拉普拉斯系数
{
  // 0. 确保 gauge 已经在 QUDA 里初始化好了
  if (!gaugePrecise) {
    errorQuda("applyLaplace_PLEGMA: gaugePrecise is null. Did you call initGaugeQuda/updateGaugeQuda()?");
  }
  quda::GaugeField &U = *gaugePrecise;

  // 1. 构造 ColorSpinorField 参数（完全照抄 QUDA_solver / QUDA_dirac 风格）
  //    注意 lat_dim_t 是 {x, y, z, t}，这里用 HGC_localL 的顺序就行，
  //    你在 QUDA_dirac 里就是这么写的。
  lat_dim_t X = { HGC_localL[0], HGC_localL[1], HGC_localL[2], HGC_localL[3] };

  bool pc_solution = false;  // full field
  // CPU param：location 必须是 CPU，不要用 CUDA
  ColorSpinorParam cpu_param(nullptr, inv_param, X, pc_solution, QUDA_CPU_FIELD_LOCATION);

  // CUDA param：从 cpu_param 拷贝 meta，并指定 CUDA 作为 location
  ColorSpinorParam cuda_param(cpu_param, inv_param, QUDA_CUDA_FIELD_LOCATION);
  cuda_param.create = QUDA_ZERO_FIELD_CREATE;

  // 2. 构造 QUDA spinors
  std::vector<ColorSpinorField> qudaIn;
  std::vector<ColorSpinorField> qudaOut;
  qudaIn.emplace_back(cuda_param);
  qudaOut.emplace_back(cuda_param);

  // 3. PLEGMA → QUDA
  //    这里我们强制当作 full field，不用 even-odd preconditioning
  bool isEven = false;
  spinorIn.copyToQUDA(qudaIn, isEven);

  // 4. 构造 vector_ref 视图
  vector_ref<ColorSpinorField>       out_view(qudaOut);
  vector_ref<const ColorSpinorField> in_view(qudaIn);
  vector_ref<const ColorSpinorField> x_view(qudaIn);  // 这里用 in 做 xpay 的 x
    int comm_dim[4] = {};
  // only switch on comms needed for directions with a derivative
  for (int i = 0; i < 4; i++) { comm_dim[i] = comm_dim_partitioned(i); }

  // 5. 调 Laplace
  // const int   *comm_override = nullptr;
  TimeProfile profileLap("Laplace");

  // full-field Laplace 必须用 QUDA_INVALID_PARITY
  QudaParity quda_parity = QUDA_INVALID_PARITY;

  ApplyLaplace(out_view, in_view, U,
               dir, a, b,
               x_view, quda_parity,
               comm_dim, profileLap);

  // 6. QUDA → PLEGMA
  spinorOut.copyFromQUDA(qudaOut, isEven);
}

static void alloc_spinor_vec(std::vector<quda::ColorSpinorField> &v,
                             int n,
                             const quda::ColorSpinorParam &p)
{
  v.clear();
  v.reserve(n);
  for (int i = 0; i < n; i++) v.emplace_back(p);
}

static quda::ColorSpinorParam make_flow_device_spinor_param(QudaInvertParam &flow_inv_param)
{
  using namespace quda;

  if (!gaugePrecise) errorQuda("make_device_spinor_param: gaugePrecise is null");

  lat_dim_t X = gaugePrecise->X();
  bool pc_solution = false;

  // ColorSpinorParam metadata comes from QudaInvertParam (precision, dslash_type,
  // gamma basis, ordering, etc.), not from QudaGaugeSmearParam.
  ColorSpinorParam hostParam(nullptr, flow_inv_param, X, pc_solution, QUDA_CPU_FIELD_LOCATION);

  // Device param: actual allocation will happen on CUDA
  ColorSpinorParam devParam(hostParam, flow_inv_param, QUDA_CUDA_FIELD_LOCATION);
  return devParam;
}





static inline int pick_block(int n_rhs, int rhs_block)
{
  if (rhs_block <= 0) rhs_block = n_rhs;
  if (rhs_block > n_rhs) rhs_block = n_rhs;

  if (n_rhs % rhs_block != 0) {
    rhs_block = std::gcd(n_rhs, rhs_block);
    if (rhs_block <= 0) rhs_block = 1;
  }
  return rhs_block;
}

template <typename FloatOut>
void exportSmearedGaugeToPLEGMA(PLEGMA_Gauge<FloatOut> &gaugeOut)
{
  QudaGaugeParam gp = newQudaGaugeParam();
  setGaugeParam(gp);

  gp.type        = QUDA_SMEARED_LINKS;
  gp.reconstruct = QUDA_RECONSTRUCT_NO;
  gp.cpu_prec    = std::is_same<FloatOut,float>::value ? QUDA_SINGLE_PRECISION
                                                       : QUDA_DOUBLE_PRECISION;

  FloatOut *buf[N_DIMS];
  size_t bytes_per_mu = gaugeOut.Bytes_total() / N_DIMS;
  for (int mu=0; mu<N_DIMS; ++mu) hostMalloc(buf[mu], bytes_per_mu);

  saveGaugeQuda((void**)buf, &gp);
  packGaugeToNormal<FloatOut,FloatOut>(gaugeOut, buf);
  gaugeOut.load();
  gaugeOut.calculatePlaq();

  for (int mu=0; mu<N_DIMS; ++mu) hostFree(buf[mu], bytes_per_mu);
}

static inline const char *loc_str(QudaFieldLocation l)
{
  switch (l) {
  case QUDA_CPU_FIELD_LOCATION:  return "CPU";
  case QUDA_CUDA_FIELD_LOCATION: return "CUDA";
  default:                       return "UNKNOWN";
  }
}

static void dumpGaugeFieldMeta(const quda::GaugeField &g, const char *tag)
{
  const int rank = quda::comm_rank();

  auto X  = g.X();       // extended dims (local, checkerboarded-aware type)
  auto LX = g.LocalX();  // local checkerboarded dims
  auto R  = g.R();       // extended radius

  logQuda(QUDA_SUMMARIZE,
          "[FFMETA][%s][rank=%d] loc=%s nd=%d vol=%zu prec=%d recon=%d\n",
          tag, rank, loc_str(g.Location()), g.Ndim(), g.Volume(), g.Precision(), (int)g.Reconstruct());

  logQuda(QUDA_SUMMARIZE,
          "[FFMETA][%s][rank=%d] X(ext)=(%d,%d,%d,%d) LocalX=(%d,%d,%d,%d) R=(%d,%d,%d,%d)\n",
          tag, rank,
          X[0], X[1], X[2], X[3],
          LX[0], LX[1], LX[2], LX[3],
          R[0], R[1], R[2], R[3]);

  // full lattice dims (global)
  logQuda(QUDA_SUMMARIZE,
          "[FFMETA][%s][rank=%d] full_dim=(%d,%d,%d,%d)\n",
          tag, rank, g.full_dim(0), g.full_dim(1), g.full_dim(2), g.full_dim(3));
}

template <typename Float>
static void fermionFlow_impl(std::vector<quda::ColorSpinorField> &fout,
                             std::vector<quda::ColorSpinorField> &fin,
                             std::vector<quda::ColorSpinorField> &t0,
                             std::vector<quda::ColorSpinorField> &t1,
                             std::vector<quda::ColorSpinorField> &t2,
                             std::vector<quda::ColorSpinorField> &t3,
                             std::vector<quda::ColorSpinorField> &t4,
                             plegma::PLEGMA_Gauge<Float>         &gaugeOut,
                             plegma::PLEGMA_Gauge<double>        &gaugeIn,
                             QudaInvertParam                     &inv_param,
                             int                                  n_steps,
                             double                               epsilon,
                             double                               t0_start,
                             bool                                 compute_plaquette,
                             bool                                 compute_qcharge,
                             bool                                 export_gauge,
                             bool                                 keep_smeared_gauge,
                             int                                  rhs_block_user)
{
  using namespace quda;

  if (!gaugePrecise) errorQuda("fermionFlow_impl: gaugePrecise is null");

  pushOutputPrefix("fermionFlow_PLEGMA: ");
  pushVerbosity(inv_param.verbosity);

  // -----------------------------
  // Smear parameters (Wilson flow, RK3)
  // -----------------------------
  QudaGaugeSmearParam smear_param = newQudaGaugeSmearParam();
  smear_param.smear_type    = QUDA_GAUGE_SMEAR_WILSON_FLOW;
  smear_param.n_steps       = std::max(0, n_steps);
  smear_param.epsilon       = epsilon;
  smear_param.t0            = t0_start;
  smear_param.restart       = QUDA_BOOLEAN_FALSE;
  smear_param.meas_interval = 1;
  smear_param.rk_order      = 3;

  std::vector<QudaGaugeObservableParam> obs(smear_param.n_steps + 1);
  for (auto &o : obs) {
    o = newQudaGaugeObservableParam();
    o.compute_plaquette = compute_plaquette ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    o.compute_qcharge   = compute_qcharge   ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    o.su_project        = QUDA_BOOLEAN_FALSE;
  }

  // -----------------------------
  // Communication topology + extended radius Rflow
  // -----------------------------
  int comm_dim[4] = {};
  for (int d = 0; d < 4; d++) comm_dim[d] = comm_dim_partitioned(d);

  lat_dim_t Rflow = {};
  for (int d = 0; d < 4; d++) Rflow[d] = comm_dim[d] ? 1 : 0;
  // for (int d = 0; d < 4; d++) Rflow[d] = 0;
  logQuda(QUDA_SUMMARIZE,
        "[FFMETA][ENTRY] rank=%d size=%d comm_dim_partitioned=%d %d %d %d -> Rflow=%d %d %d %d\n",
        comm_rank(), comm_size(),
        comm_dim[0], comm_dim[1], comm_dim[2], comm_dim[3],
        Rflow[0], Rflow[1], Rflow[2], Rflow[3]);

  // (Re)build the resident smeared gauge if starting a new trajectory
  const bool new_traj = (std::abs(t0_start) < epsilon / 2);
  logQuda(QUDA_SUMMARIZE,
          "[FF] enter: t0_start=%.6e eps=%.6e new_traj=%d gaugeSmeared=%p keep=%d\n",
          t0_start, epsilon, (int)new_traj, (void*)gaugeSmeared, (int)keep_smeared_gauge);

  if (new_traj || gaugeSmeared == nullptr) {
    freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
      dumpGaugeFieldMeta(*gaugePrecise, "gaugePrecise(before createExtendedGauge)");


    auto recon = gaugePrecise->Reconstruct();
    logQuda(QUDA_SUMMARIZE, "[FF] recon(gaugePrecise)=%d\n", (int)recon);

    gaugeSmeared = createExtendedGauge(*gaugePrecise, Rflow, profileFermionFlow, false, recon);
    dumpGaugeFieldMeta(*gaugeSmeared, "gaugeSmeared(after createExtendedGauge)");

    if (!gaugeSmeared) errorQuda("fermionFlow_impl: createExtendedGauge failed");
    logQuda(QUDA_SUMMARIZE, "[FF] recon(gaugeSmeared)=%d\n", (int)gaugeSmeared->Reconstruct());

    // Ensure extended halos are initialized and consistent
    gaugeSmeared->exchangeExtendedGhost(gaugeSmeared->R());
  }

  // -----------------------------
  // Extended gauge buffers: input/output + temporary
  // -----------------------------
  GaugeField *in_ext = gaugeSmeared;

  GaugeFieldParam gParamEx(*gaugeSmeared);
  GaugeField gaugeAux(gParamEx);
  GaugeField *out_ext = &gaugeAux;

  GaugeFieldParam gTempParam(*gaugeSmeared);
  gTempParam.create      = QUDA_NULL_FIELD_CREATE;
  gTempParam.reconstruct = QUDA_RECONSTRUCT_NO; // temp field is not on the SU(3) manifold
  GaugeField gaugeTemp(gTempParam);

  // -----------------------------
  // Body gauge for Laplace (R=0 geometry)
  // We extract/copy the interior links from the extended gauge into this field.
  // Use spinor precision (not gauge precision) for ApplyLaplace compatibility,
  // saving ~50% memory when spinors are single but gauge is double.
  // -----------------------------
    logQuda(QUDA_SUMMARIZE,
        "[FFPREC] inv_param: cuda_prec=%d sloppy=%d precondition=%d ritz=%d\n",
      (int)inv_param.cuda_prec,
      (int)inv_param.cuda_prec_sloppy,
      (int)inv_param.cuda_prec_precondition,
        (int)inv_param.cuda_prec_ritz);

  auto spinor_prec = fin[0].Precision();
    logQuda(QUDA_SUMMARIZE,
      "[FFPREC] field precision: fin=%d gaugePrecise=%d gaugeSmeared=%d\n",
      (int)spinor_prec,
      (int)gaugePrecise->Precision(),
      gaugeSmeared ? (int)gaugeSmeared->Precision() : -1);

  GaugeFieldParam bodyParam(*gaugePrecise);
  bodyParam.create      = QUDA_NULL_FIELD_CREATE;
  bodyParam.reconstruct = gaugePrecise->Reconstruct();
  bodyParam.setPrecision(spinor_prec, true);
  GaugeField body(bodyParam);
    logQuda(QUDA_SUMMARIZE,
      "[FFPREC] Laplace body gauge precision=%d (copied from extended gauge with spinor precision)\n",
      (int)body.Precision());

  auto measure_plaq = [&](GaugeField &g, QudaGaugeObservableParam &o, const char *tag) {
    // For extended fields, we must exchange the extended halos before observables
    g.exchangeExtendedGhost(g.R());
    gaugeObservables(g, o);
    if (compute_plaquette) {
      logQuda(QUDA_SUMMARIZE,
              "[P] %s plaq(total)=%.16e  plaq_t=%.16e  plaq_s=%.16e\n",
              tag, o.plaquette[0], o.plaquette[2], o.plaquette[1]);
    }
  };

  // -----------------------------
  // Fermion parameters / blocking
  // -----------------------------
  const int n_rhs = (int)fin.size();
  if ((int)fout.size() != n_rhs) errorQuda("fermionFlow_impl: fout.size != fin.size");

  const int rhs_block = pick_block(n_rhs, rhs_block_user);
  if ((int)t0.size() != rhs_block || (int)t1.size() != rhs_block || (int)t2.size() != rhs_block ||
      (int)t3.size() != rhs_block || (int)t4.size() != rhs_block) {
    errorQuda("fermionFlow_impl: temp vec sizes must equal rhs_block");
  }

  int parity = 0;
  double a = 1.0;
  double b = -8.0;

  // Initial measurement at flow time t0 (aligned with performWFlowQuda behavior)
  measure_plaq(*in_ext, obs[0], "t=0 (in_ext)");
  logQuda(QUDA_SUMMARIZE, "flow_t = %le\n", smear_param.t0);

  // -----------------------------
  // Main loop (RK3: W1/W2/VT + coupled Laplace updates)
  // -----------------------------
  for (int step = 0; step < smear_param.n_steps; step++) {

    // -------- Stage 0: use g0=in_ext to compute k1 and psi1 --------
    // body <- g0 interior (avoid copyFieldOffset parity constraints)
    copyExtendedGauge(body, *in_ext, QUDA_CUDA_FIELD_LOCATION);
    body.exchangeGhost();

    for (int rhs0 = 0; rhs0 < n_rhs; rhs0 += rhs_block) {

      // Load psi0 -> t1
      for (int i = 0; i < rhs_block; i++) t1[i] = fin[rhs0 + i];

      // k1 = L(psi0; g0) -> t0
      for (int i = 0; i < rhs_block; i++) t0[i] = t1[i];
      ApplyLaplace(t4, t0, body, 4, a, b, t0, parity, comm_dim, profileFermionFlow);
      for (int i = 0; i < rhs_block; i++) t0[i] = t4[i];

      // Store k1 into fout as workspace: fout[rhs] = k1
      for (int i = 0; i < rhs_block; i++) fout[rhs0 + i] = t0[i];

      // psi1 = psi0 + (eps/4) * k1, write back into fin (fin becomes psi1)
      blas::axpy(smear_param.epsilon / 4., t0, t1);
      for (int i = 0; i < rhs_block; i++) fin[rhs0 + i] = t1[i];
    }

    // -------- Gauge W1: out_ext = g1 --------
    GFlowStep(*out_ext, gaugeTemp, *in_ext, smear_param.epsilon, smear_param.smear_type, WFLOW_STEP_W1);

    // -------- Stage 1: use g1=out_ext to compute k2 and psi2 --------
    copyExtendedGauge(body, *out_ext, QUDA_CUDA_FIELD_LOCATION);
    body.exchangeGhost();

    // psi2 = psi1 + (8/9)eps*k2 - (17/36)eps*k1
    const double c2 = smear_param.epsilon * 8. / 9.;
    const double c1 = -smear_param.epsilon * 17. / 36.;

    for (int rhs0 = 0; rhs0 < n_rhs; rhs0 += rhs_block) {

      // Load psi1 -> t2
      for (int i = 0; i < rhs_block; i++) t2[i] = fin[rhs0 + i];

      // k2 = L(psi1; g1) -> t1
      for (int i = 0; i < rhs_block; i++) t1[i] = t2[i];
      ApplyLaplace(t4, t1, body, 4, a, b, t1, parity, comm_dim, profileFermionFlow);
      for (int i = 0; i < rhs_block; i++) t1[i] = t4[i];

      // psi2 starts from psi1 (t2)
      blas::axpy(c2, t1, t2); // psi2 += c2*k2

      // psi2 += c1*k1 (k1 is stored in fout[rhs])
      for (int i = 0; i < rhs_block; i++) t0[i] = fout[rhs0 + i];
      blas::axpy(c1, t0, t2);

      // Store psi2 into fout (overwrites previous k1 workspace)
      for (int i = 0; i < rhs_block; i++) fout[rhs0 + i] = t2[i];
    }

    // -------- Gauge W2: in_ext = g2 --------
    GFlowStep(*in_ext, gaugeTemp, *out_ext, smear_param.epsilon, smear_param.smear_type, WFLOW_STEP_W2);

    // -------- Stage 2: use g2=in_ext to compute k3 and psi_new --------
    copyExtendedGauge(body, *in_ext, QUDA_CUDA_FIELD_LOCATION);
    body.exchangeGhost();

    const double c3 = smear_param.epsilon * 3. / 4.;

    for (int rhs0 = 0; rhs0 < n_rhs; rhs0 += rhs_block) {

      // Load psi2 -> t2 (psi2 is stored in fout)
      for (int i = 0; i < rhs_block; i++) t2[i] = fout[rhs0 + i];

      // k3 = L(psi2; g2) -> t3
      for (int i = 0; i < rhs_block; i++) t3[i] = t2[i];
      ApplyLaplace(t4, t3, body, 4, a, b, t3, parity, comm_dim, profileFermionFlow);
      for (int i = 0; i < rhs_block; i++) t3[i] = t4[i];

      // psi_new = psi1 + c3*k3, where psi1 is stored in fin
      for (int i = 0; i < rhs_block; i++) t1[i] = fin[rhs0 + i];
      blas::axpy(c3, t3, t1);

      // Write back fin <- psi_new
      for (int i = 0; i < rhs_block; i++) fin[rhs0 + i] = t1[i];
    }

    // -------- Gauge VT: out_ext = g(t+eps) --------
    GFlowStep(*out_ext, gaugeTemp, *in_ext, smear_param.epsilon, smear_param.smear_type, WFLOW_STEP_VT);

    // Swap gauge buffers: next step uses g(t+eps) as input
    std::swap(in_ext, out_ext);

    // -------- Measurements --------
    if ((step + 1) % smear_param.meas_interval == 0) {
      const int m = step + 1;

      measure_plaq(*in_ext, obs[m], "measure(in_ext)");
      logQuda(QUDA_SUMMARIZE, "flow_t = %le\n", smear_param.t0 + smear_param.epsilon * (step + 1));

      if (compute_plaquette)
        logQuda(QUDA_SUMMARIZE, "plaquette(total)=%.16e\n", obs[m].plaquette[0]);

      for (int r = 0; r < n_rhs; r++)
        logQuda(QUDA_SUMMARIZE, "spinor[%d] norm2 = %.16e\n", r, blas::norm2(fin[r]));
    }
  }

  // Final output: fout <- fin
  for (int r = 0; r < n_rhs; r++) fout[r] = fin[r];

  // Copy final gauge back into gaugeSmeared (for restart / export) and refresh halos
  if (in_ext != gaugeSmeared) copyExtendedGauge(*gaugeSmeared, *in_ext, QUDA_CUDA_FIELD_LOCATION);
  gaugeSmeared->exchangeExtendedGhost(gaugeSmeared->R());

  if (export_gauge) {
    exportSmearedGaugeToPLEGMA(gaugeOut);
  }

  if (!keep_smeared_gauge) {
    freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
    gaugeSmeared = nullptr;
  }

  logQuda(QUDA_SUMMARIZE, "[FF] exit: gaugeSmeared=%p keep=%d\n",
          (void*)gaugeSmeared, (int)keep_smeared_gauge);

  popVerbosity();
  popOutputPrefix();
}

void fermionFlow_PLEGMA(plegma::PLEGMA_Propagator<float> &propOut,
                                plegma::PLEGMA_Propagator<float> &propIn,
                                plegma::PLEGMA_Propagator<float> &seqOut,
                                plegma::PLEGMA_Propagator<float> &seqIn,
                                plegma::PLEGMA_Gauge<float>     &gaugeOut,
                                plegma::PLEGMA_Gauge<double>     &gaugeIn,
                                int n_steps, double epsilon, double t0_start,
                                bool compute_plaquette, bool compute_qcharge,
                                bool export_gauge, bool keep_smeared_gauge,
                                int rhs_block_user)
{
  using namespace quda;

  if (!gaugePrecise) errorQuda("fermionFlow_PLEGMA: gaugePrecise is null");

  // Double precision for the Laplace kernel: single precision accumulates
  // enough rounding error to bias the fermion flow result noticeably.
  QudaInvertParam inv_flow = make_flow_inv_param(QUDA_DOUBLE_PRECISION);
  const int n_rhs = 24;                 // 12 + 12
  bool isEv = false;

  // -----------------------------
  // Allocate fin/fout full 24
  // -----------------------------
  std::vector<ColorSpinorField> fin, fout;
  auto devParam = make_flow_device_spinor_param(inv_flow);

  devParam.create = QUDA_ZERO_FIELD_CREATE;
  alloc_spinor_vec(fin,  n_rhs, devParam);

  devParam.create = QUDA_NULL_FIELD_CREATE;
  alloc_spinor_vec(fout, n_rhs, devParam);

  // -----------------------------
  // Allocate temps only blocksize
  // -----------------------------
  int rhs_block = pick_block(n_rhs, rhs_block_user);
  std::vector<ColorSpinorField> t0, t1, t2, t3, t4;

  devParam.create = QUDA_NULL_FIELD_CREATE;
  alloc_spinor_vec(t0, rhs_block, devParam);
  alloc_spinor_vec(t1, rhs_block, devParam);
  alloc_spinor_vec(t2, rhs_block, devParam);
  alloc_spinor_vec(t3, rhs_block, devParam);
  alloc_spinor_vec(t4, rhs_block, devParam);

  // -----------------------------
  // Scatter propIn/seqIn -> fin[0..23]
  // -----------------------------
  prop_copyToQUDA_offset(propIn, fin, 0,  isEv);
  prop_copyToQUDA_offset(seqIn,  fin, 12, isEv);

  fermionFlow_impl(fout, fin, t0, t1, t2, t3, t4,
                           gaugeOut, gaugeIn, inv_flow,
                           n_steps, epsilon, t0_start,
                           compute_plaquette, compute_qcharge,
                           export_gauge, keep_smeared_gauge,
                           rhs_block);
  // -----------------------------
  // Gather fout -> propOut/seqOut
  // -----------------------------
  prop_copyFromQUDA_offset(propOut, fout, 0,  isEv);
  prop_copyFromQUDA_offset(seqOut,  fout, 12, isEv);
}

void fermionFlow_PLEGMA(plegma::PLEGMA_Propagator<double> &propOut,
                                plegma::PLEGMA_Propagator<double> &propIn,
                                plegma::PLEGMA_Propagator<double> &seqOut,
                                plegma::PLEGMA_Propagator<double> &seqIn,
                                plegma::PLEGMA_Gauge<double>      &gaugeOut,
                                plegma::PLEGMA_Gauge<double>      &gaugeIn,
                                int n_steps, double epsilon, double t0_start,
                                bool compute_plaquette, bool compute_qcharge,
                                bool export_gauge, bool keep_smeared_gauge,
                                int rhs_block_user)
{
  using namespace quda;

  if (!gaugePrecise) errorQuda("fermionFlow_PLEGMA<double>: gaugePrecise is null");

  QudaInvertParam inv_flow = make_flow_inv_param(QUDA_DOUBLE_PRECISION);
  const int n_rhs = 24;
  bool isEv = false;

  std::vector<ColorSpinorField> fin, fout;
  auto devParam = make_flow_device_spinor_param(inv_flow);

  devParam.create = QUDA_ZERO_FIELD_CREATE;
  alloc_spinor_vec(fin,  n_rhs, devParam);
  devParam.create = QUDA_NULL_FIELD_CREATE;
  alloc_spinor_vec(fout, n_rhs, devParam);

  int rhs_block = pick_block(n_rhs, rhs_block_user);
  std::vector<ColorSpinorField> t0, t1, t2, t3, t4;
  devParam.create = QUDA_NULL_FIELD_CREATE;
  alloc_spinor_vec(t0, rhs_block, devParam);
  alloc_spinor_vec(t1, rhs_block, devParam);
  alloc_spinor_vec(t2, rhs_block, devParam);
  alloc_spinor_vec(t3, rhs_block, devParam);
  alloc_spinor_vec(t4, rhs_block, devParam);

  prop_copyToQUDA_offset(propIn, fin, 0,  isEv);
  prop_copyToQUDA_offset(seqIn,  fin, 12, isEv);

  fermionFlow_impl(fout, fin, t0, t1, t2, t3, t4,
                   gaugeOut, gaugeIn, inv_flow,
                   n_steps, epsilon, t0_start,
                   compute_plaquette, compute_qcharge,
                   export_gauge, keep_smeared_gauge,
                   rhs_block);

  prop_copyFromQUDA_offset(propOut, fout, 0,  isEv);
  prop_copyFromQUDA_offset(seqOut,  fout, 12, isEv);
}

template <typename VecFloat, typename GaugeFloat>
void fermionFlow_PLEGMA(plegma::PLEGMA_Vector<VecFloat> &spinorOut,
                        plegma::PLEGMA_Vector<VecFloat> &spinorIn,
                        plegma::PLEGMA_Gauge<GaugeFloat>  &gaugeOut,
                        plegma::PLEGMA_Gauge<double>  &gaugeIn,
                        int n_steps, double epsilon, double t0_start,
                        bool compute_plaquette, bool compute_qcharge,
                            bool export_gauge,
                          bool keep_smeared_gauge)
{
  using namespace quda;

  if (!gaugePrecise) errorQuda("fermionFlow_PLEGMA(Vector): gaugePrecise is null");

  // Double precision for the Laplace kernel: single precision accumulates
  // enough rounding error to bias the fermion flow result noticeably.
  QudaInvertParam inv_flow = make_flow_inv_param(QUDA_DOUBLE_PRECISION);

  // Local (non-static) spinor buffers
  std::vector<ColorSpinorField> fin, fout, t0, t1, t2, t3, t4;

  // Allocate buffers
  auto devParam = make_flow_device_spinor_param(inv_flow);

  devParam.create = QUDA_ZERO_FIELD_CREATE;
  alloc_spinor_vec(fin,  1, devParam);

  devParam.create = QUDA_NULL_FIELD_CREATE; // no need to memset
  alloc_spinor_vec(fout, 1, devParam);
  alloc_spinor_vec(t0,   1, devParam);
  alloc_spinor_vec(t1,   1, devParam);
  alloc_spinor_vec(t2,   1, devParam);
  alloc_spinor_vec(t3,   1, devParam);
  alloc_spinor_vec(t4,   1, devParam);

  // PLEGMA -> QUDA
  bool isEv = false; // full field
  spinorIn.copyToQUDA(fin, isEv);

  // Run flow
  fermionFlow_impl(fout, fin, t0, t1, t2, t3, t4,
                   gaugeOut, gaugeIn, inv_flow,
                   n_steps, epsilon, t0_start,
                   compute_plaquette, compute_qcharge, export_gauge, keep_smeared_gauge, 1);

  // QUDA -> PLEGMA
  spinorOut.copyFromQUDA(fout, isEv);
}

template <typename VecFloat, typename GaugeFloat>
void fermionFlow_PLEGMA(std::vector<PLEGMA_Vector<VecFloat>*> &out,
                        std::vector<PLEGMA_Vector<VecFloat>*> &in,
                        PLEGMA_Gauge<GaugeFloat>  &gaugeOut,
                        PLEGMA_Gauge<double> &gaugeIn,
                        int n_steps, double epsilon, double t0_start,
                        bool compute_plaquette, bool compute_qcharge,
                        bool export_gauge, bool keep_smeared_gauge,
                        int rhs_block_user)
{
  using namespace quda;
  if (!gaugePrecise) errorQuda("fermionFlow_PLEGMA(batch): gaugePrecise is null");
  if (in.size() != out.size()) errorQuda("fermionFlow_PLEGMA(batch): size mismatch");

  const int n_rhs = (int)in.size();
  if (n_rhs == 0) return;

  // Double precision for the Laplace kernel: single precision accumulates
  // enough rounding error to bias the fermion flow result noticeably.
  QudaInvertParam inv_flow = make_flow_inv_param(QUDA_DOUBLE_PRECISION);
  bool isEv = false;

  std::vector<ColorSpinorField> fin, fout;
  auto devParam = make_flow_device_spinor_param(inv_flow);

  devParam.create = QUDA_ZERO_FIELD_CREATE;
  alloc_spinor_vec(fin, n_rhs, devParam);
  devParam.create = QUDA_NULL_FIELD_CREATE;
  alloc_spinor_vec(fout, n_rhs, devParam);

  int rhs_block = pick_block(n_rhs, rhs_block_user);
  std::vector<ColorSpinorField> t0, t1, t2, t3, t4;
  devParam.create = QUDA_NULL_FIELD_CREATE;
  alloc_spinor_vec(t0, rhs_block, devParam);
  alloc_spinor_vec(t1, rhs_block, devParam);
  alloc_spinor_vec(t2, rhs_block, devParam);
  alloc_spinor_vec(t3, rhs_block, devParam);
  alloc_spinor_vec(t4, rhs_block, devParam);

  // for (int r = 0; r < n_rhs; r++)
    vec_copyToQUDA_offset(in, fin, 0, isEv);

  fermionFlow_impl(fout, fin, t0, t1, t2, t3, t4,
                   gaugeOut, gaugeIn, inv_flow,
                   n_steps, epsilon, t0_start,
                   compute_plaquette, compute_qcharge,
                   export_gauge, keep_smeared_gauge,
                   rhs_block);

  // for (int r = 0; r < n_rhs; r++)
    vec_copyFromQUDA_offset(out, fout, 0, isEv);
}

template void fermionFlow_PLEGMA<double, double>(
    plegma::PLEGMA_Vector<double> &spinorOut,
    plegma::PLEGMA_Vector<double> &spinorIn,
    plegma::PLEGMA_Gauge<double> &gaugeOut,
    plegma::PLEGMA_Gauge<double> &gaugeIn,
    int n_steps, double epsilon, double t0_start,
    bool compute_plaquette, bool compute_qcharge, bool export_gauge,
    bool keep_smeared_gauge);

template void fermionFlow_PLEGMA<float, double>(
    plegma::PLEGMA_Vector<float> &spinorOut,
    plegma::PLEGMA_Vector<float> &spinorIn,
    plegma::PLEGMA_Gauge<double> &gaugeOut,
    plegma::PLEGMA_Gauge<double> &gaugeIn,
    int n_steps, double epsilon, double t0_start,
    bool compute_plaquette, bool compute_qcharge, bool export_gauge,
    bool keep_smeared_gauge);

template void fermionFlow_PLEGMA<float, float>(
    plegma::PLEGMA_Vector<float> &spinorOut,
    plegma::PLEGMA_Vector<float> &spinorIn,
    plegma::PLEGMA_Gauge<float> &gaugeOut,
    plegma::PLEGMA_Gauge<double> &gaugeIn,
    int n_steps, double epsilon, double t0_start,
    bool compute_plaquette, bool compute_qcharge, bool export_gauge,
    bool keep_smeared_gauge);

template void fermionFlow_PLEGMA<double, float>(
    plegma::PLEGMA_Vector<double> &spinorOut,
    plegma::PLEGMA_Vector<double> &spinorIn,
    plegma::PLEGMA_Gauge<float> &gaugeOut,
    plegma::PLEGMA_Gauge<double> &gaugeIn,
    int n_steps, double epsilon, double t0_start,
    bool compute_plaquette, bool compute_qcharge, bool export_gauge,
    bool keep_smeared_gauge);

template void fermionFlow_PLEGMA<double, double>(
    std::vector<plegma::PLEGMA_Vector<double> *> &out,
    std::vector<plegma::PLEGMA_Vector<double> *> &in,
    plegma::PLEGMA_Gauge<double> &gaugeOut,
    plegma::PLEGMA_Gauge<double> &gaugeIn,
    int n_steps, double epsilon, double t0_start,
    bool compute_plaquette, bool compute_qcharge, bool export_gauge,
    bool keep_smeared_gauge, int rhs_block_user);

template void fermionFlow_PLEGMA<double, float>(
    std::vector<plegma::PLEGMA_Vector<double> *> &out,
    std::vector<plegma::PLEGMA_Vector<double> *> &in,
    plegma::PLEGMA_Gauge<float> &gaugeOut,
    plegma::PLEGMA_Gauge<double> &gaugeIn,
    int n_steps, double epsilon, double t0_start,
    bool compute_plaquette, bool compute_qcharge, bool export_gauge,
    bool keep_smeared_gauge, int rhs_block_user);

template void fermionFlow_PLEGMA<float, float>(
    std::vector<plegma::PLEGMA_Vector<float> *> &out,
    std::vector<plegma::PLEGMA_Vector<float> *> &in,
    plegma::PLEGMA_Gauge<float> &gaugeOut,
    plegma::PLEGMA_Gauge<double> &gaugeIn,
    int n_steps, double epsilon, double t0_start,
    bool compute_plaquette, bool compute_qcharge, bool export_gauge,
    bool keep_smeared_gauge, int rhs_block_user);

template void fermionFlow_PLEGMA<float, double>(
    std::vector<plegma::PLEGMA_Vector<float> *> &out,
    std::vector<plegma::PLEGMA_Vector<float> *> &in,
    plegma::PLEGMA_Gauge<double> &gaugeOut,
    plegma::PLEGMA_Gauge<double> &gaugeIn,
    int n_steps, double epsilon, double t0_start,
    bool compute_plaquette, bool compute_qcharge, bool export_gauge,
    bool keep_smeared_gauge, int rhs_block_user);

static void fermionAdjointFlow_impl(std::vector<quda::ColorSpinorField> &fout,
                                    std::vector<quda::ColorSpinorField> &fin,
                                    std::vector<quda::ColorSpinorField> &f_temp0,
                                    std::vector<quda::ColorSpinorField> &f_temp1,
                                    std::vector<quda::ColorSpinorField> &f_temp2,
                                    std::vector<quda::ColorSpinorField> &f_temp3,
                                    std::vector<quda::ColorSpinorField> &f_temp4,
                                    plegma::PLEGMA_Gauge<double>        &gaugeIn,
                                    QudaInvertParam                     &inv_param,
                                    int                                  n_steps,
                                    double                               epsilon,
                                    bool                                 antiperiodic)
{
  using namespace quda;

  if (!gaugePrecise) errorQuda("fermionAdjointFlow_impl: gaugePrecise is null");

  const int nSteps = std::max(0, n_steps);
  const size_t nSpinors = fin.size();

  pushOutputPrefix("fermionAdjointFlow_PLEGMA: ");
  pushVerbosity(inv_param.verbosity);

  // ------------------------------------------------------------
  // Build comm mask and an Rflow that is safe for multi-GPU:
  // - If a dimension is partitioned, R must be >= 1 to avoid
  //   extended-ghost corner cases (e.g. divide-by-zero in tuning).
  // - Otherwise R can be 0.
  // ------------------------------------------------------------
  int comm_dim[4] = {};
  for (int d = 0; d < 4; d++) comm_dim[d] = comm_dim_partitioned(d);

  lat_dim_t Rflow = {};
  for (int d = 0; d < 4; d++) Rflow[d] = comm_dim[d] ? 1 : 0;

  logQuda(QUDA_SUMMARIZE,
          "[ADJMETA][ENTRY] n_steps=%d eps=%.8e rank=%d size=%d comm_dim_partitioned=%d %d %d %d -> Rflow=%d %d %d %d\n",
          nSteps, epsilon, comm_rank(), comm_size(),
          comm_dim[0], comm_dim[1], comm_dim[2], comm_dim[3],
          Rflow[0], Rflow[1], Rflow[2], Rflow[3]);

  logQuda(QUDA_SUMMARIZE,
          "Adjoint fermion flow: n_steps = %d, epsilon = %.8e\n",
          nSteps, epsilon);

  // ------------------------------------------------------------
  // (1) Rebuild an extended gauge field for the adjoint flow.
  // We always rebuild from gaugePrecise here (conservative/safe).
  // ------------------------------------------------------------
  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);

  auto recon = gaugePrecise->Reconstruct();
  logQuda(QUDA_SUMMARIZE, "[ADJ] recon(gaugePrecise)=%d\n", (int)recon);

  gaugeSmeared = createExtendedGauge(*gaugePrecise, Rflow, profileAdjointFermionFlow,
                                     /*redundant_comms=*/false,
                                     /*recon=*/recon);
  if (!gaugeSmeared) errorQuda("fermionAdjointFlow_impl: createExtendedGauge failed");

  // Make sure extended halos are consistent before any gauge observables/flow steps.
  gaugeSmeared->exchangeExtendedGhost(gaugeSmeared->R());

  // ------------------------------------------------------------
  // (2) Allocate extended gauge work buffers (W1/W2/VT) and temp gauge.
  // All flow steps are done on extended gauge fields.
  // ------------------------------------------------------------
  GaugeFieldParam gParamEx(*gaugeSmeared);
  GaugeField gaugeW1(gParamEx);
  GaugeField gaugeW2(gParamEx);
  GaugeField gaugeVT(gParamEx);

  // Temporary gauge must not use reconstruct (not on the manifold).
  GaugeFieldParam gTempParam(*gaugeSmeared);
  gTempParam.create      = QUDA_NULL_FIELD_CREATE;
  gTempParam.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeField gaugeTemp(gTempParam);

  // Snapshot of the starting extended gauge, used to reset W0 for each inner rebuild.
  GaugeField gin(*gaugeSmeared);

  // Reused buffer aliases (extended)
  GaugeField &g_W0 = *gaugeSmeared; // overwritten during reconstruction
  GaugeField &g_W1 = gaugeW1;
  GaugeField &g_W2 = gaugeW2;
  GaugeField &g_VT = gaugeVT;

  // ------------------------------------------------------------
  // (3) Helper gauge field for Laplace: "normal ghost" field.
  // We will copy an extended gauge into it and then call exchangeGhost().
  // ------------------------------------------------------------
  GaugeField precise;
  {
    GaugeFieldParam helper(*gaugePrecise);
    helper.create = QUDA_NULL_FIELD_CREATE;
    precise = GaugeField(helper);
  }

  // Laplace coefficients and parity (match forward flow conventions)
  const int parity = 0;
  const double a = 1.0;
  const double b = -8.0;

  // ------------------------------------------------------------
  // (4) Optional debug: plaquette at forward t=0 (gaugePrecise).
  // ------------------------------------------------------------
  QudaGaugeObservableParam obs = newQudaGaugeObservableParam();
  obs.compute_plaquette = QUDA_BOOLEAN_TRUE;
  obs.compute_qcharge   = QUDA_BOOLEAN_FALSE;
  obs.su_project        = QUDA_BOOLEAN_FALSE;

  gaugeObservables(*gaugePrecise, obs);
  logQuda(QUDA_SUMMARIZE, "[ADJ] forward t=0 plaquette(total)=%.16e\n", obs.plaquette[0]);

  // ------------------------------------------------------------
  // (5) Fermion initialization
  // ------------------------------------------------------------
  f_temp3 = fin;   // current adjoint state
  fout    = fin;   // nSteps==0 => identity

  for (size_t k = 0; k < nSpinors; k++) {
    logQuda(QUDA_SUMMARIZE,
            "[ADJ] start: spinor[%lu] norm2 = %.16e\n",
            (unsigned long)k, blas::norm2(fin[k]));
  }

  // ------------------------------------------------------------
  // (6) Main adjoint-flow loop
  // We rebuild the gauge path up to t = (nSteps - j) * eps each outer step j.
  // ------------------------------------------------------------
  for (int j = 0; j < nSteps; j++) {

    const int n_forward = nSteps - j;

    // Rebuild W0/W1/W2/VT along the forward flow trajectory.
    // NOTE: GFlowStep() already calls exchangeExtendedGhost(out.R()) internally.
    for (int i = 0; i < n_forward; i++) {

      if (i == 0) {
        g_W0 = gin; // reset W0 to the starting extended gauge
        g_W0.exchangeExtendedGhost(g_W0.R()); // ensure halos consistent after assignment
      } else {
        std::swap(g_W0, g_VT); // previous VT becomes next W0
      }

      GFlowStep(g_W1, gaugeTemp, g_W0, epsilon, QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_STEP_W1);
      GFlowStep(g_W2, gaugeTemp, g_W1, epsilon, QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_STEP_W2);
      GFlowStep(g_VT, gaugeTemp, g_W2, epsilon, QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_STEP_VT);
    }

    // Measure plaquette on the extended gauge used at this level (optional).
    g_VT.exchangeExtendedGhost(g_VT.R());
    gaugeObservables(g_VT, obs);

    const double t_forward = double(n_forward) * epsilon;
    logQuda(QUDA_SUMMARIZE,
            "[ADJ] j=%d forward_t=%.8e plaquette(total)=%.16e\n",
            j, t_forward, obs.plaquette[0]);

    // Initialize auxiliaries from the current adjoint state
    f_temp0 = f_temp3;
    f_temp1 = f_temp3;
    f_temp2 = f_temp3;

    // ---- Laplace with W2 ----
    copyExtendedGauge(precise, g_W2, QUDA_CUDA_FIELD_LOCATION);
    precise.exchangeGhost();
    ApplyLaplace(f_temp4, f_temp0, precise, 4, a, b, f_temp0, parity, comm_dim, profileAdjointFermionFlow);

    // f_temp2 = (3/4) eps * Laplace_W2(f_temp0)
    for (size_t k = 0; k < nSpinors; k++) blas::ax(epsilon * 3.0 / 4.0, f_temp4[k]);
    f_temp2 = f_temp4;

    // ---- Laplace with W1 ----
    copyExtendedGauge(precise, g_W1, QUDA_CUDA_FIELD_LOCATION);
    precise.exchangeGhost();
    ApplyLaplace(f_temp4, f_temp2, precise, 4, a, b, f_temp2, parity, comm_dim, profileAdjointFermionFlow);

    // f_temp3 += (8/9) eps * Laplace_W1(f_temp2)
    for (size_t k = 0; k < nSpinors; k++) blas::axpy(epsilon * 8.0 / 9.0, f_temp4[k], f_temp3[k]);

    // build f_temp4 = f_temp3 - (8/9) f_temp2   (matches your original algebra)
    f_temp1 = f_temp3;
    f_temp4 = f_temp1;
    for (size_t k = 0; k < nSpinors; k++) blas::axpy(-8.0 / 9.0, f_temp2[k], f_temp4[k]);

    // ---- Laplace with W0 ----
    copyExtendedGauge(precise, g_W0, QUDA_CUDA_FIELD_LOCATION);
    precise.exchangeGhost();
    ApplyLaplace(f_temp0, f_temp4, precise, 4, a, b, f_temp4, parity, comm_dim, profileAdjointFermionFlow);

    // Combine terms (same structure as your original)
    for (size_t k = 0; k < nSpinors; k++) {
      blas::ax(epsilon * 1.0 / 4.0, f_temp0[k]);
      blas::axpy(1.0, f_temp2[k], f_temp0[k]);
      blas::axpy(1.0, f_temp1[k], f_temp0[k]);
    }

    // Update state/output
    fout    = f_temp0;
    f_temp3 = f_temp0;

    const double t_adj = double(j + 1) * epsilon;
    for (size_t k = 0; k < nSpinors; k++) {
      logQuda(QUDA_SUMMARIZE,
              "[ADJ] progress t_adj=%.8e step=%d spinor[%lu] norm2=%.16e\n",
              t_adj, j + 1, (unsigned long)k, blas::norm2(fout[k]));
    }
  }

  // ------------------------------------------------------------
  // (7) Cleanup: release smeared gauge to avoid lifetime issues.
  // ------------------------------------------------------------
  freeUniqueGaugeQuda(QUDA_SMEARED_LINKS);
  gaugeSmeared = nullptr;

  popVerbosity();
  popOutputPrefix();
}

void fermionAdjointFlow_PLEGMA(plegma::PLEGMA_Vector<float> &spinorOut,
                               plegma::PLEGMA_Vector<float> &spinorIn,
                               plegma::PLEGMA_Gauge<double>  &gaugeIn,
                               QudaInvertParam               &inv_param,
                               int                            n_steps,
                               double                         epsilon,
                               bool                           antiperiodic)
{
  using namespace quda;

  updateGaugeQuda(gaugeIn, antiperiodic, QUDA_WILSON_LINKS);
  if (!gaugePrecise) errorQuda("fermionAdjointFlow_PLEGMA(Vector): gaugePrecise is null");

  std::vector<ColorSpinorField> fin, fout, t0, t1, t2, t3, t4;

  auto devParam = make_flow_device_spinor_param(inv_param);

  devParam.create = QUDA_ZERO_FIELD_CREATE;
  alloc_spinor_vec(fin,  1, devParam);

  devParam.create = QUDA_NULL_FIELD_CREATE;
  alloc_spinor_vec(fout, 1, devParam);
  alloc_spinor_vec(t0,   1, devParam);
  alloc_spinor_vec(t1,   1, devParam);
  alloc_spinor_vec(t2,   1, devParam);
  alloc_spinor_vec(t3,   1, devParam);
  alloc_spinor_vec(t4,   1, devParam);

  bool isEv = false;
  spinorIn.copyToQUDA(fin, isEv);

  fermionAdjointFlow_impl(fout, fin, t0, t1, t2, t3, t4,
                          gaugeIn, inv_param,
                          n_steps, epsilon, antiperiodic);

  spinorOut.copyFromQUDA(fout, isEv);
}

void fermionAdjointFlow_PLEGMA(plegma::PLEGMA_Propagator<float> &propOut,
                               plegma::PLEGMA_Propagator<float> &propIn,
                               plegma::PLEGMA_Gauge<double>      &gaugeIn,
                               QudaInvertParam                   &inv_param,
                               int                                n_steps,
                               double                             epsilon,
                               bool                               antiperiodic)
{
  using namespace quda;

  updateGaugeQuda(gaugeIn, antiperiodic, QUDA_WILSON_LINKS);
  if (!gaugePrecise) errorQuda("fermionAdjointFlow_PLEGMA(Propagator): gaugePrecise is null");

  std::vector<ColorSpinorField> fin, fout, t0, t1, t2, t3, t4;

  const int n_rhs = N_SPINS * N_COLS; // 12

  auto devParam = make_flow_device_spinor_param(inv_param);

  devParam.create = QUDA_ZERO_FIELD_CREATE;
  alloc_spinor_vec(fin,  n_rhs, devParam);

  devParam.create = QUDA_NULL_FIELD_CREATE;
  alloc_spinor_vec(fout, n_rhs, devParam);
  alloc_spinor_vec(t0,   n_rhs, devParam);
  alloc_spinor_vec(t1,   n_rhs, devParam);
  alloc_spinor_vec(t2,   n_rhs, devParam);
  alloc_spinor_vec(t3,   n_rhs, devParam);
  alloc_spinor_vec(t4,   n_rhs, devParam);

  bool isEv = false;
  propIn.copyToQUDA(fin, isEv);

  fermionAdjointFlow_impl(fout, fin, t0, t1, t2, t3, t4,
                          gaugeIn, inv_param,
                          n_steps, epsilon, antiperiodic);

  propOut.copyFromQUDA(fout, isEv);
}
