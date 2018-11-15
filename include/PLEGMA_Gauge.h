#include <PLEGMA_Field.h>

#ifndef _PLEGMA_GAUGE_H
#define _PLEGMA_GAUGE_H

namespace plegma {
  ////////////////////////
  // CLASS: PLEGMA_Gauge //
  ////////////////////////
  
  template<typename Float>
    class PLEGMA_Gauge : public PLEGMA_Field<Float> {
  public:
    PLEGMA_Gauge(ALLOCATION_FLAG alloc_flag);
    ~PLEGMA_Gauge(){;}
    
    void packGauge(double **gauge);
    void packGaugeToBackup(void **gauge);
    void loadGaugeFromBackup();
    void justDownloadGauge();
    void loadGauge();

    void calculatePlaq();
    void calculatePlaqShifts();
  };
}

#endif
