#include <PLEGMA.h>
//#include <quda.h>
//#include <quda_gauge.h>
//#include <quda_gauge_tools.h>
#include <PLEGMA_utils.h>
#include <cuda_profiler_api.h>

using namespace plegma;
using namespace quda;

int main(int argc, char **argv)
{
  initializeOptions(argc, argv); // Put list of options later
  //================ Add your options in this between initializeOptions and initializePLEGMA ================//

  //=========================================================================================================//
  initializePLEGMA();

  // Allocation done on BOTH, DEVICE and HOST
  {
    cudaProfilerStart();
    PLEGMA_Gauge<float> gauge(BOTH);
    // Reading from Lime file and loading to device
    gauge.readFile(latfile, LIME_FORMAT);
    PLEGMA_Gauge<double> gauged(BOTH);
    PLEGMA_Gauge<double> gaugedout(BOTH);


    // Computing plaquette on device in three different way for crosschecking
    gauge.calculatePlaq();
    gauge.calculatePlaqCorners();
    gauge.calculatePlaqShifts();
    gauge.calculatePlaqClover();
    gauge.calculatePlaqStaples();
 
    gauged.copy(gauge);

    // Loading to QUDA and computing plaquette also there
    initGaugeQuda(gauged, true);
    plaqQuda();
    gSmear_QUDA( gaugedout, gauged, false);
    gaugedout.calculatePlaq();
    gaugedout.unload();
    gaugedout.writeLIME("gaugetest_check3");
//    gauge.copy(gaugedout);
//    gauge.unload();
//    gauge.writeLIME("gaugetest_check2");
/*    QudaGaugeSmearParam smear_param = newQudaGaugeSmearParam();
    smear_param.smear_type = QUDA_GAUGE_SMEAR_HYP;
    PLEGMA_printf("smear type %d %d\n",smear_param.smear_type, QUDA_GAUGE_SMEAR_HYP); 
    fflush(stdout);
    smear_param.alpha1 = 0.75;  // typical HYP values
    smear_param.alpha2 = 0.60;
    smear_param.alpha3 = 0.30;
    smear_param.n_steps = 1;    // number of iterations
    smear_param.meas_interval = 1;

    //smear_param.gauge_order = gauge_param.gauge_order;
    //smear_param.precision = gauge_param.cuda_prec;

    int gauge_smear_step=1;
    int measurement_interval=1;
    bool su_project = false;
    QudaGaugeObservableParam *obs_param = new QudaGaugeObservableParam[gauge_smear_step / measurement_interval + 1];
    for (int i = 0; i < gauge_smear_step / measurement_interval + 1; i++) {
      obs_param[i] = newQudaGaugeObservableParam();
      obs_param[i].compute_plaquette = QUDA_BOOLEAN_TRUE;
      obs_param[i].compute_qcharge = QUDA_BOOLEAN_TRUE;
      obs_param[i].su_project = su_project ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    } 

    // ------------------------------
    // Apply HYP smearing
    // ------------------------------
    performGaugeSmearQuda(&smear_param, obs_param);
    QudaGaugeParam gauge_param = newQudaGaugeParam();
    setGaugeParam(gauge_param);
    gauge_param.type = QUDA_WILSON_LINKS;
    gauge_param.make_resident_gauge = 0;
    int volume = gauge_param.X[0] * gauge_param.X[1] *
             gauge_param.X[2] * gauge_param.X[3];
    size_t bytes_per_dir = volume * gauge_param.site_size *((gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float));

    PLEGMA_printf("Lattice %dx%dx%dx%d\n", gauge_param.X[0], gauge_param.X[1], gauge_param.X[2], gauge_param.X[3]);
    PLEGMA_printf("site_size = %d, cpu_prec = %d\n", gauge_param.site_size, gauge_param.cpu_prec);
    PLEGMA_printf("bytes_per_dir = %zu (%.3f MB)\n", bytes_per_dir, bytes_per_dir / (1024.0*1024.0));




//   size_t bytes_per_dir = gauge_param.gauge_site_size * gauge_param.volume * 
//                       ((gauge_param.precision == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float));

    double* buf[N_DIMS];
    for(int i=0; i<N_DIMS; i++) hostMalloc(buf[i], bytes_per_dir);
    gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
    saveGaugeQuda(buf, &gauge_param);
 //   packGaugeToNormal(gauge,buf);
 //   gauge.load();
 //   gauge.calculatePlaq();
 //   gauge.writeLIME("gaugetest");
    for(int i=0; i<N_DIMS; i++) hostFree(buf[i], bytes_per_dir);*/

  }
  finalize();
 
  return 0;
}
