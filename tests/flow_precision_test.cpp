#include "PLEGMA_global.h"
#include "QUDA_wilson_flow.h"
#include "enum_quda.h"
#include "quda.h"

#include <PLEGMA.h>
#include <PLEGMA_utils.h>

#include <mpi.h>

#include <complex>
#include <string>
#include <vector>

using namespace plegma;
using namespace quda;

static std::vector<double> runtime;
#define TIME(fnc)                                                              \
  runtime.push_back(MPI_Wtime());                                              \
  fnc;                                                                         \
  PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back()); \
  runtime.pop_back()

struct FlowMetrics {
  double ref_norm = 0.0;
  double test_norm = 0.0;
  double diff_norm = 0.0;
  double rel_diff = 0.0;
};

static const char *precision_name(QudaPrecision prec)
{
  switch (prec) {
  case QUDA_DOUBLE_PRECISION:
    return "double";
  case QUDA_SINGLE_PRECISION:
    return "float";
  case QUDA_HALF_PRECISION:
    return "half";
  default:
    return "unknown";
  }
}

static FlowMetrics compare_vectors(PLEGMA_Vector<double> &reference,
                                   PLEGMA_Vector<double> &candidate)
{
  PLEGMA_Vector<double> diff(BOTH);
  diff.copy(reference, DEVICE);
  diff.add(candidate, std::complex<double>(-1.0, 0.0));

  FlowMetrics metrics;
  metrics.ref_norm = reference.norm();
  metrics.test_norm = candidate.norm();
  metrics.diff_norm = diff.norm();
  metrics.rel_diff = (metrics.ref_norm > 0.0) ? metrics.diff_norm / metrics.ref_norm
                                              : metrics.diff_norm;
  return metrics;
}

static void print_metrics(const char *label, const FlowMetrics &metrics)
{
  PLEGMA_printf("[%s]\n", label);
  PLEGMA_printf("  reference norm = %.16e\n", metrics.ref_norm);
  PLEGMA_printf("  candidate norm = %.16e\n", metrics.test_norm);
  PLEGMA_printf("  diff norm      = %.16e\n", metrics.diff_norm);
  PLEGMA_printf("  relative diff  = %.16e\n", metrics.rel_diff);
}

template <typename VecFloat, typename GaugeFloat>
static void run_fermion_flow(PLEGMA_Vector<VecFloat> &out,
                             PLEGMA_Vector<VecFloat> &in,
                             PLEGMA_Gauge<GaugeFloat> &gauge_out,
                             PLEGMA_Gauge<double> &gauge_in,
                             int n_steps,
                             double epsilon,
                             double t0_start,
                             bool compute_plaquette,
                             bool compute_qcharge,
                             bool export_gauge,
                             bool keep_smeared_gauge)
{
  TIME(fermionFlow_PLEGMA(out, in,
                          gauge_out, gauge_in,
                          n_steps, epsilon, t0_start,
                          compute_plaquette, compute_qcharge,
                          export_gauge, keep_smeared_gauge));
}

int main(int argc, char **argv)
{
  std::vector<std::string> listOpt = {
      "verbosity",
      "load-gauge"
  };

  initializeOptions(argc, argv, true, listOpt);

  int flow_steps = 10;
  double flow_epsilon = 0.01;
  double flow_t0 = 0.0;
  int stoc_seed = rng_seed;
  bool antiperiodic = false;
  bool compute_plaquette = false;
  bool compute_qcharge = false;
  bool export_gauge = false;
  bool keep_smeared_gauge = false;
  bool dump_hdf5 = false;

  HGC_options->set("flow-steps", "Number of Wilson flow steps", verbosity, flow_steps);
  HGC_options->set("flow-epsilon", "Wilson flow epsilon", verbosity, flow_epsilon);
  HGC_options->set("flow-t0", "Initial flow time", verbosity, flow_t0);
  HGC_options->set("stoc-seed", "Seed used to build the stochastic source", verbosity, stoc_seed);
  HGC_options->set("flow-antiperiodic", "Use anti-periodic temporal boundary during flow", verbosity, antiperiodic);
  HGC_options->set("flow-compute-plaquette", "Compute plaquette during flow", verbosity, compute_plaquette);
  HGC_options->set("flow-compute-qcharge", "Compute topological charge during flow", verbosity, compute_qcharge);
  HGC_options->set("flow-export-gauge", "Export flowed gauge from fermionFlow_PLEGMA", verbosity, export_gauge);
  HGC_options->set("flow-keep-smeared-gauge", "Keep flowed resident gauge after fermionFlow_PLEGMA", verbosity, keep_smeared_gauge);
  HGC_options->set("dump-hdf5", "Dump source and flowed vectors to HDF5", verbosity, dump_hdf5);

  initializePLEGMA();

  {
    PLEGMA_Gauge<double> gauge(BOTH);
    TIME(gauge.readFile(latfile, LIME_FORMAT));
    gauge.calculatePlaq();

    initGaugeQuda(gauge, antiperiodic);
    plaqQuda();

    updateOptions(LIGHT);

    // fermionFlow_PLEGMA uses double precision internally; float-precision
    // Laplace accumulates enough rounding error to bias the result noticeably.
    PLEGMA_printf("Wilson flow stochastic-source precision test\n");
    PLEGMA_printf("  gauge file      = %s\n", latfile.c_str());
    PLEGMA_printf("  n_steps         = %d\n", flow_steps);
    PLEGMA_printf("  epsilon         = %.16e\n", flow_epsilon);
    PLEGMA_printf("  t0_start        = %.16e\n", flow_t0);
    PLEGMA_printf("  stochastic seed = %d\n", stoc_seed);
    PLEGMA_printf("  spinor run A    = double\n");
    PLEGMA_printf("  spinor run B    = float\n");

    PLEGMA_Vector<double> src_double(BOTH);
    PLEGMA_Vector<float> src_float(BOTH);
    src_double.randInit(stoc_seed);
    src_double.stochastic_Z(4);
    src_float.copy(src_double);

    if (dump_hdf5) {
      src_double.writeHDF5("stoc_src_double");
      PLEGMA_Vector<double> src_float_as_double(BOTH);
      src_float_as_double.copy(src_float);
      src_float_as_double.writeHDF5("stoc_src_float_cast_to_double");
    }

    PLEGMA_Gauge<double> gauge_flowed_double(BOTH);
    PLEGMA_Gauge<double> gauge_flowed_double_from_float_src(BOTH);
    PLEGMA_Gauge<float> gauge_flowed_float(BOTH);
    PLEGMA_Vector<double> out_double(BOTH);
    PLEGMA_Vector<double> out_double_from_float_src(BOTH);
    PLEGMA_Vector<float> out_float(BOTH);
    PLEGMA_Vector<double> out_float_as_double(BOTH);
    PLEGMA_Vector<float> out_double_as_float(BOTH);
    PLEGMA_Vector<double> out_double_roundtrip(BOTH);
    PLEGMA_Vector<double> src_float_as_double(BOTH);
    src_float_as_double.copy(src_float);

    // Restore the same resident gauge state before each run.
    updateGaugeQuda(gauge, antiperiodic, QUDA_WILSON_LINKS);
    run_fermion_flow(out_double, src_double,
                     gauge_flowed_double, gauge,
                     flow_steps, flow_epsilon, flow_t0,
                     compute_plaquette, compute_qcharge,
                     export_gauge, keep_smeared_gauge);

    updateGaugeQuda(gauge, antiperiodic, QUDA_WILSON_LINKS);
    run_fermion_flow(out_float, src_float,
                     gauge_flowed_float, gauge,
                     flow_steps, flow_epsilon, flow_t0,
                     compute_plaquette, compute_qcharge,
                     export_gauge, keep_smeared_gauge);

    // Diagnostic A: isolate source precision effect in the double flow path.
    updateGaugeQuda(gauge, antiperiodic, QUDA_WILSON_LINKS);
    run_fermion_flow(out_double_from_float_src, src_float_as_double,
                     gauge_flowed_double_from_float_src, gauge,
                     flow_steps, flow_epsilon, flow_t0,
                     compute_plaquette, compute_qcharge,
                     export_gauge, keep_smeared_gauge);

    out_float_as_double.copy(out_float);
    out_double_as_float.copy(out_double);
    out_double_roundtrip.copy(out_double_as_float);

    const FlowMetrics self_metrics            = compare_vectors(out_double, out_double);
    const FlowMetrics cast_metrics            = compare_vectors(out_double, out_double_roundtrip);
    const FlowMetrics src_precision_metrics   = compare_vectors(out_double, out_double_from_float_src);
    const FlowMetrics flow_metrics            = compare_vectors(out_double, out_float_as_double);

    PLEGMA_printf("\nPrecision summary\n");
    PLEGMA_printf("  source norm (double) = %.16e\n", src_double.norm());
    PLEGMA_printf("  flowed norm (double) = %.16e\n", out_double.norm());
    PLEGMA_printf("  flowed norm (float)  = %.16e\n", out_float_as_double.norm());
    print_metrics("sanity: double-flow vs itself", self_metrics);
    print_metrics("cast-only: double -> float -> double", cast_metrics);
    print_metrics("source-precision effect: double(src_f32_cast) vs double(src_f64)", src_precision_metrics);
    print_metrics("double-flow vs float-flow (spinor storage only)", flow_metrics);

    if (dump_hdf5) {
      out_double.writeHDF5("flowed_stoc_double");
      out_float_as_double.writeHDF5("flowed_stoc_float_cast_to_double");
    }

    PLEGMA_printf("\nInterpretation\n");
    PLEGMA_printf("  Both runs use the same Z4 stochastic source.\n");
    PLEGMA_printf("  The Laplace kernel always runs in double; the reported diff\n");
    PLEGMA_printf("  measures only the effect of float vs double spinor storage.\n");

    finalizeGaugeQuda();
  }

  finalize();
  return 0;
}