#include "PLEGMA_global.h"
#include "PLEGMA_oneD.cuh"
#include "enum_quda.h"
#include "quda.h"
#include "QUDA_wilson_flow.h"

#include <PLEGMA.h>
#include <PLEGMA_utils.h>

#include <mpi.h>
#include <string>
#include <vector>

using namespace plegma;
using namespace quda;

// Simple scoped timer
static std::vector<double> runtime;
#define TIME(fnc)                                                              \
  runtime.push_back(MPI_Wtime());                                              \
  fnc;                                                                         \
  PLEGMA_printf("TIME for " #fnc " %f sec\n", MPI_Wtime() - runtime.back());   \
  runtime.pop_back()

extern quda::GaugeField *gaugePrecise;
extern quda::GaugeField *gaugeSmeared;

static double rel_diff(PLEGMA_Vector<double> &a, PLEGMA_Vector<double> &b)
{
  PLEGMA_Vector<double> tmp(BOTH);
  tmp.copy(a, DEVICE);
  tmp.add(b, std::complex<double>(-1.0, 0.0)); // tmp = a - b

  double na = a.norm();
  double nd = tmp.norm();
  return (na > 0.0) ? nd / na : nd;
}

static double rel_diff_prop(PLEGMA_Propagator<double> &a,
                            PLEGMA_Propagator<double> &b)
{
  PLEGMA_Propagator<double> tmp(BOTH);
  tmp.copy(a, DEVICE);
  tmp.add(b, std::complex<double>(-1.0, 0.0)); // tmp = a - b

  double na = a.norm();
  double nd = tmp.norm();
  return (na > 0.0) ? nd / na : nd;
}

static void print_norm_prop(const char *name, PLEGMA_Propagator<double> &p)
{
  PLEGMA_printf("%s norm = %.16e\n", name, p.norm());
}

static void print_norm(const char *name, PLEGMA_Vector<double> &v)
{
  PLEGMA_printf("%s norm = %.16e\n", name, v.norm());
}

int main(int argc, char **argv)
{
  std::vector<std::string> listOpt = {
      "verbosity",
      "load-gauge",
      "corr-file-format"
  };

  initializeOptions(argc, argv, true, listOpt);
  initializePLEGMA();

  {
    const int    n_steps      = 10;
    const double epsilon      = 0.01;
    const double t0_start     = 0.0;
    const bool   antiperiodic = false;

    // -----------------------------
    // Gauge
    // -----------------------------
    PLEGMA_Gauge<double> gauge(BOTH);
    PLEGMA_Gauge<double> gauge_flowed(BOTH);

    gauge.readFile(latfile, LIME_FORMAT);
    gauge.calculatePlaq();

    initGaugeQuda(gauge, antiperiodic);
    plaqQuda();

    PLEGMA_printf("After initGaugeQuda: gaugePrecise=%p gaugeSmeared=%p\n",
                  (void *)gaugePrecise, (void *)gaugeSmeared);

    // -----------------------------
    // Dirac operator
    // -----------------------------
    updateOptions(LIGHT);

    const QudaDslashType test_dslash = QUDA_TWISTED_MASS_DSLASH;

    QUDA_dirac *D = new QUDA_dirac(test_dslash);
    D->print();

    // Separate invert param for fermionFlow_PLEGMA
    QudaInvertParam inv_flow = newQudaInvertParam();
    setInvertParam(inv_flow);
    inv_flow.dslash_type = test_dslash;

    // -----------------------------
    // One stochastic source
    // -----------------------------
    PLEGMA_Vector<double> src, out;
    PLEGMA_Vector<double> src_flowed(BOTH);
    PLEGMA_Vector<double> out_before(BOTH);
    PLEGMA_Vector<double> out_after(BOTH);
    PLEGMA_Vector<double> out_reload(BOTH);
    PLEGMA_Propagator<double> prop_q;
    PLEGMA_Propagator<double> prop_q_s;

    src.randInit(rng_seed);
    src.stochastic_Z(4);
    src.communicateSideGhost(-1, DIR_BOTH, DO_ALL);
    src.writeHDF5("src");

    for (int isc = 0; isc < 12; ++isc) {
      prop_q.absorb(src, isc / 3, isc % 3);
    }

    prop_q.communicateSideGhost(-1, DIR_BOTH, DO_ALL);
    // TIME(prop_q_s.gaussianSmearing(prop_q, gauge, 50, 0.5));
    // TIME(covariant_derivative(out,src,gauge,0));
        PLEGMA_printf("\n==== Test 1: basic run in all directions ====\n");

    for (int dir = 0; dir < 4; ++dir) {
      out.zero_device();
      if (dir < 3) {
        TIME(covariant_derivative(out, src, gauge, dir));
        print_norm("custom spatial oneD", out);
      } else {
        TIME(covariant_derivative(out, src, gauge, dir, TIME_PLUS));
        print_norm("custom temporal +", out);

        out.zero_device();
        TIME(covariant_derivative(out, src, gauge, dir, TIME_MINUS));
        print_norm("custom temporal -", out);
      }
    }

    PLEGMA_printf("\n==== Test 2: compare temporal branch with built-in covD ====\n");

{
  PLEGMA_Vector<double> ref_plus(BOTH), ref_minus(BOTH);
  PLEGMA_Vector<double> my_plus(BOTH),  my_minus(BOTH);

  TIME(ref_plus.covD(src, gauge, 3));
  TIME(covariant_derivative(my_plus, src, gauge, 3, TIME_PLUS));

  TIME(ref_minus.covD(src, gauge, 7));
  TIME(covariant_derivative(my_minus, src, gauge, 3, TIME_MINUS));

  double err_plus  = rel_diff(ref_plus,  my_plus);
  double err_minus = rel_diff(ref_minus, my_minus);

  PLEGMA_printf("relative diff (time +) = %.16e\n", err_plus);
  PLEGMA_printf("relative diff (time -) = %.16e\n", err_minus);
}
PLEGMA_printf("\n==== Test 3: compare spatial branch with built-in forward-backward ====\n");

for (int dir = 0; dir < 3; ++dir) {
  PLEGMA_Vector<double> ref_plus(BOTH), ref_minus(BOTH), ref_diff(BOTH);
  PLEGMA_Vector<double> my_diff(BOTH);

  TIME(ref_plus.covD(src, gauge, dir));
  TIME(ref_minus.covD(src, gauge, dir + 4));

  ref_diff.copy(ref_plus, DEVICE);
  ref_diff.add(ref_minus, std::complex<double>(-1.0, 0.0));

  TIME(covariant_derivative(my_diff, src, gauge, dir));

  double err = rel_diff(ref_diff, my_diff);
  PLEGMA_printf("dir = %d, relative diff = %.16e\n", dir, err);
}


    PLEGMA_printf("\n==== Test 4: zero source ====\n");

{
  PLEGMA_Vector<double> zero_src(BOTH), zero_out(BOTH);
  zero_src.zero_device();

  for (int dir = 0; dir < 3; ++dir) {
    zero_out.zero_device();
    TIME(covariant_derivative(zero_out, zero_src, gauge, dir));
    PLEGMA_printf("zero test dir=%d, norm=%.16e\n", dir, zero_out.norm());
  }

  zero_out.zero_device();
  TIME(covariant_derivative(zero_out, zero_src, gauge, 3, TIME_PLUS));
  PLEGMA_printf("zero test time+, norm=%.16e\n", zero_out.norm());

  zero_out.zero_device();
  TIME(covariant_derivative(zero_out, zero_src, gauge, 3, TIME_MINUS));
  PLEGMA_printf("zero test time-, norm=%.16e\n", zero_out.norm());
}

PLEGMA_printf("\n==== Test 5: propagator basic run in all directions ====\n");

{
  PLEGMA_Propagator<double> prop_out(BOTH);

  for (int dir = 0; dir < 4; ++dir) {
    prop_out.zero_device();

    if (dir < 3) {
      TIME(covariant_derivative(prop_out, prop_q, gauge, dir));
      print_norm_prop("custom propagator spatial oneD", prop_out);
    } else {
      TIME(covariant_derivative(prop_out, prop_q, gauge, dir, TIME_PLUS));
      print_norm_prop("custom propagator temporal +", prop_out);

      prop_out.zero_device();
      TIME(covariant_derivative(prop_out, prop_q, gauge, dir, TIME_MINUS));
      print_norm_prop("custom propagator temporal -", prop_out);
    }
  }
}

PLEGMA_printf("\n==== Test 6: propagator should equal 12 repeated vector results ====\n");

for (int dir = 0; dir < 4; ++dir) {
  PLEGMA_Vector<double>      vec_out(BOTH);
  PLEGMA_Propagator<double>  prop_out(BOTH), prop_ref(BOTH);

  if (dir < 3) {
    TIME(covariant_derivative(vec_out,  src,   gauge, dir));
    TIME(covariant_derivative(prop_out, prop_q, gauge, dir));
  } else {
    TIME(covariant_derivative(vec_out,  src,   gauge, dir, TIME_PLUS));
    TIME(covariant_derivative(prop_out, prop_q, gauge, dir, TIME_PLUS));
  }

  prop_ref.zero_device();
  for (int isc = 0; isc < 12; ++isc) {
    prop_ref.absorb(vec_out, isc / 3, isc % 3);
  }

  double err = rel_diff_prop(prop_out, prop_ref);

  if (dir < 3) {
    PLEGMA_printf("prop repeated-copy check dir=%d, relative diff = %.16e\n", dir, err);
  } else {
    PLEGMA_printf("prop repeated-copy check time+, relative diff = %.16e\n", err);
  }
}

PLEGMA_printf("\n==== Test 7: propagator repeated-copy check for time- ====\n");

{
  PLEGMA_Vector<double>      vec_out(BOTH);
  PLEGMA_Propagator<double>  prop_out(BOTH), prop_ref(BOTH);

  TIME(covariant_derivative(vec_out,  src,   gauge, 3, TIME_MINUS));
  TIME(covariant_derivative(prop_out, prop_q, gauge, 3, TIME_MINUS));

  prop_ref.zero_device();
  for (int isc = 0; isc < 12; ++isc) {
    prop_ref.absorb(vec_out, isc / 3, isc % 3);
  }

  double err = rel_diff_prop(prop_out, prop_ref);
  PLEGMA_printf("prop repeated-copy check time-, relative diff = %.16e\n", err);
}

PLEGMA_printf("\n==== Test 8: zero propagator ====\n");

{
  PLEGMA_Propagator<double> zero_prop(BOTH), zero_prop_out(BOTH);
  zero_prop.zero_device();

  for (int dir = 0; dir < 3; ++dir) {
    zero_prop_out.zero_device();
    TIME(covariant_derivative(zero_prop_out, zero_prop, gauge, dir));
    PLEGMA_printf("zero prop test dir=%d, norm=%.16e\n", dir, zero_prop_out.norm());
  }

  zero_prop_out.zero_device();
  TIME(covariant_derivative(zero_prop_out, zero_prop, gauge, 3, TIME_PLUS));
  PLEGMA_printf("zero prop test time+, norm=%.16e\n", zero_prop_out.norm());

  zero_prop_out.zero_device();
  TIME(covariant_derivative(zero_prop_out, zero_prop, gauge, 3, TIME_MINUS));
  PLEGMA_printf("zero prop test time-, norm=%.16e\n", zero_prop_out.norm());
}

PLEGMA_printf("\n==== Test 9: norm ratio check ====\n");

{
  PLEGMA_Vector<double>      vec_out(BOTH);
  PLEGMA_Propagator<double>  prop_out(BOTH);

  TIME(covariant_derivative(vec_out,  src,   gauge, 0));
  TIME(covariant_derivative(prop_out, prop_q, gauge, 0));

  double nv = vec_out.norm();
  double np = prop_out.norm();

  PLEGMA_printf("vector norm = %.16e\n", nv);
  PLEGMA_printf("prop   norm = %.16e\n", np);
  PLEGMA_printf("prop/vector = %.16e, expected sqrt(12)=%.16e\n",
                np / nv, std::sqrt(12.0));
}
    // // -----------------------------
    // // applyD before flow
    // // -----------------------------
    // TIME(
    //   D->apply<M>(out_before.D_elem(), src.D_elem(),
    //   QUDA_MASS_NORMALIZATION)
    // );
    // out_before.writeHDF5("applyD_before");
    //   PLEGMA_printf("D * src  %f src %f \n",out_before.norm(), src.norm());

    // // -----------------------------
    // // ordinary fermion flow
    // // IMPORTANT:
    // // applyD after this is still tested on the SAME original src
    // // -----------------------------
    // TIME(
    //   fermionFlow_PLEGMA(src_flowed, src,
    //                      gauge_flowed, gauge,
    //                      inv_flow,
    //                      n_steps, epsilon, t0_start,
    //                      antiperiodic,
    //                      true, true,   // compute_plaquette, compute_qcharge
    //                      true, false)    // export_gauge, keep_smeared_gauge
    // );

    // src_flowed.writeHDF5("src_flowed");
    // gauge_flowed.writeHDF5("gauge_flowed");

    // PLEGMA_printf("After fermionFlow_PLEGMA: gaugePrecise=%p
    // gaugeSmeared=%p\n",
    //               (void *)gaugePrecise, (void *)gaugeSmeared);

    // // -----------------------------
    // // applyD after flow, on the SAME original src
    // // -----------------------------
    // TIME(
    //   D->apply<M>(out_after.D_elem(), src.D_elem(), QUDA_MASS_NORMALIZATION)
    // );
    // out_after.writeHDF5("applyD_after");
    //   PLEGMA_printf("D * src  %f src %f \n",out_after.norm(), src.norm());

    // // -----------------------------
    // // Reload original gauge and rebuild D
    // // -----------------------------
    // updateGaugeQuda(gauge, antiperiodic, QUDA_WILSON_LINKS);

    // PLEGMA_printf("After updateGaugeQuda(original): gaugePrecise=%p
    // gaugeSmeared=%p\n",
    //               (void *)gaugePrecise, (void *)gaugeSmeared);

    // delete D;
    // D = new QUDA_dirac(test_dslash);

    // TIME(
    //   D->apply<M>(out_reload.D_elem(), src.D_elem(),
    //   QUDA_MASS_NORMALIZATION)
    // );
    // out_reload.writeHDF5("applyD_after_reload");
    //   PLEGMA_printf("D * src  %f src %f \n",out_reload.norm(), src.norm());

    // delete D;
    finalizeGaugeQuda();
  }

  finalize();
  return 0;
}

