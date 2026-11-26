#include <PLEGMA.h>
#include <PLEGMA_utils.h>

QudaInvertParam make_flow_inv_param(QudaPrecision prec = QUDA_DOUBLE_PRECISION);
void resetFermionFlowSmearedGauge();

void wilsonFlow_QUDA(PLEGMA_Gauge<double> &gaugeOut,
                     PLEGMA_Gauge<double> &gaugeIn, int n_steps, double epsilon,
                     double t0_start, bool compute_plaquette,
                     bool compute_qcharge);
// void gSmear_QUDA(PLEGMA_Gauge<double> &gaugeOut, PLEGMA_Gauge<double>
// &gaugeIn,
//                   bool antiperiodic);

void fermionFlow_QUDA(plegma::PLEGMA_Vector<double> &spinorOut,
                      plegma::PLEGMA_Vector<double> &spinorIn,
                      plegma::PLEGMA_Gauge<double> &gaugeOut,
                      plegma::PLEGMA_Gauge<double> &gaugeIn,
                      QudaInvertParam &inv_param, int n_steps, double epsilon,
                      double t0_start, bool antiperiodic,
                      bool compute_plaquette, bool compute_qcharge);

void adjointFermionFlow_QUDA(PLEGMA_Vector<double> &spinorOut,
                             PLEGMA_Vector<double> &spinorIn,
                             QudaInvertParam &inv_param, int n_steps,
                             double epsilon, double t0_start,
                             int adj_n_save = 0);

void applyLaplace_PLEGMA(PLEGMA_Vector<double> &spinorOut,
                         PLEGMA_Vector<double> &spinorIn,
                         QudaInvertParam &inv_param,
                         int dir, // 要省略的方向
                         double a, double b);
// fermionFlow_PLEGMA: fermion flow via coupled Wilson flow + Laplace steps.
// Internal computation always uses double precision; single-precision Laplace
// accumulates enough rounding error to bias the result noticeably.
// antiperiodic boundary conditions are irrelevant for the forward flow
// (the gauge is already resident in QUDA; use initGaugeQuda/updateGaugeQuda
// with antiperiodic=true before calling if needed).
template <typename VecFloat, typename GaugeFloat>
void fermionFlow_PLEGMA(plegma::PLEGMA_Vector<VecFloat> &spinorOut,
                        plegma::PLEGMA_Vector<VecFloat> &spinorIn,
                        plegma::PLEGMA_Gauge<GaugeFloat> &gaugeOut,
                        plegma::PLEGMA_Gauge<double> &gaugeIn,
                        int n_steps, double epsilon,
                        double t0_start,
                        bool compute_plaquette, bool compute_qcharge,
                        bool export_gauge, bool keep_smeared_gauge);
template <typename VecFloat, typename GaugeFloat>
void fermionFlow_PLEGMA(std::vector<plegma::PLEGMA_Vector<VecFloat>*> &out,
                        std::vector<plegma::PLEGMA_Vector<VecFloat>*> &in,
                        plegma::PLEGMA_Gauge<GaugeFloat>              &gaugeOut,
                        plegma::PLEGMA_Gauge<double>                  &gaugeIn,
                        int n_steps, double epsilon, double t0_start,
                        bool compute_plaquette, bool compute_qcharge,
                        bool export_gauge, bool keep_smeared_gauge,
                        int rhs_block_user);
void fermionFlow_PLEGMA(plegma::PLEGMA_Propagator<float> &propOut,
                        plegma::PLEGMA_Propagator<float> &propIn,
                        plegma::PLEGMA_Propagator<float> &seqOut,
                        plegma::PLEGMA_Propagator<float> &seqIn,
                        plegma::PLEGMA_Gauge<float>     &gaugeOut,
                        plegma::PLEGMA_Gauge<double>     &gaugeIn,
                        int n_steps, double epsilon, double t0_start,
                        bool compute_plaquette, bool compute_qcharge,
                        bool export_gauge, bool keep_smeared_gauge, int rhs_block_user);
void fermionFlow_PLEGMA(plegma::PLEGMA_Propagator<double> &propOut,
                        plegma::PLEGMA_Propagator<double> &propIn,
                        plegma::PLEGMA_Propagator<double> &seqOut,
                        plegma::PLEGMA_Propagator<double> &seqIn,
                        plegma::PLEGMA_Gauge<double>      &gaugeOut,
                        plegma::PLEGMA_Gauge<double>      &gaugeIn,
                        int n_steps, double epsilon, double t0_start,
                        bool compute_plaquette, bool compute_qcharge,
                        bool export_gauge, bool keep_smeared_gauge, int rhs_block_user);

void fermionAdjointFlow_PLEGMA(plegma::PLEGMA_Propagator<float> &propOut,
                               plegma::PLEGMA_Propagator<float> &propIn,
                               plegma::PLEGMA_Gauge<double> &gaugeIn,
                               QudaInvertParam &inv_param, int n_steps,
                               double epsilon, bool antiperiodic);
void fermionAdjointFlow_PLEGMA(plegma::PLEGMA_Vector<float> &spinorOut,
                               plegma::PLEGMA_Vector<float> &spinorIn,
                               plegma::PLEGMA_Gauge<double> &gaugeIn,
                               QudaInvertParam &inv_param, int n_steps,
                               double epsilon, bool antiperiodic);