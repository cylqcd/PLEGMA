#include <PLEGMA_global.h>
#include <PLEGMA_Field.h>
#include <PLEGMA_Gauge.h>
#include <PLEGMA_Su3field.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Propagator.h>
#include <PLEGMA_Correlator.h>
#include <PLEGMA_QLoops.h>
#include <PLEGMA_FT.h>

#ifndef _PLEGMA__H
#define _PLEGMA__H

namespace plegma {
  
  void PLEGMA_init();

  void print_status();

  void PLEGMA_end();
  
}
#endif
