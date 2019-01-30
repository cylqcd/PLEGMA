#include <PLEGMA_Field.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Gauge.h>
#include <string>

#ifndef _PLEGMA_QLOOPS
#define _PLEGMA_QLOOPS

namespace plegma{
    
  template<typename Float>
    class PLEGMA_QLoops : public PLEGMA_Field<Float> {
  private:
    Float *h_loc;
    Float *h_oneD[N_DIMS];
    Float *h_oneDC[N_DIMS];

    bool isOneD;
  public:
    PLEGMA_QLoops(ALLOCATION_FLAG alloc_flag=BOTH, bool isOneD=false);
    ~PLEGMA_QLoops();

    Float* H_loc() const{return h_loc;}
    Float** H_oneD() const{if(!isOneD) errorQuda("oneD is not enabled");  return (Float**)h_oneD;}
    Float** H_oneDC() const{if(!isOneD) errorQuda("oneD is not enabled");  return (Float**)h_oneDC;}

    void oneEnd_trick(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r,
		      Float val , bool accum );
    void oneEnd_trick(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r,
		      PLEGMA_Vector<Float> &tmp, PLEGMA_Gauge<Float> &gauge, Float val , bool accum );
    void contractG5(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r, ACCUM_TYPE acc_type);
    void contractG5(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r);

    void write_ASCII(std::string filename_local);
    void write_ASCII(std::string filename_local, std::string filename_oneD, std::string filename_oneDC);
    
    void load(Float* h_ptr);
  };

}
#endif
