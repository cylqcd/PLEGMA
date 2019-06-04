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
    Float *h_twoD[N_DIMS*(N_DIMS-1)]; // we are interested only for off diagonal elements

    bool isOneD;
    bool isTwoD;
    std::vector<std::pair<int,int>> twoD_index;
  public:
    PLEGMA_QLoops(ALLOCATION_FLAG alloc_flag=BOTH, GHOST_FLAG ghost_flag = NO_GHOSTS, bool isPinnedHost = true, bool isOneD=false, bool isTwoD=false);
    ~PLEGMA_QLoops();

    Float* H_loc() const{return h_loc;}
    Float** H_oneD() const{if(!isOneD) PLEGMA_error("oneD is not enabled");  return (Float**)h_oneD;}
    Float** H_oneDC() const{if(!isOneD) PLEGMA_error("oneD is not enabled");  return (Float**)h_oneDC;}
    Float** H_twoD() const{if(!isTwoD) PLEGMA_error("twoD is not enabled");  return (Float**)h_twoD;}
    bool IsOneD() const{return isOneD;}
    bool IsTwoD() const{return isTwoD;}
    std::vector<std::pair<int,int>> get_twoD_index() const{return twoD_index;}
    
    /**
       @brief Computes the one-end trick (only ultralocal case) for disconnected quark loops as x_l^dag \gamma_5 \Gamma c_r.
       In case of standard one-end trick x_l = x_r = x while for generalized one-end trick x_l = x and x_r = \gamma_5 D_C x
       @param PLEGMA_Vector<Float> &x_l, left solution vector
       @param PLEGMA_Vector<Float> &x_r, right solution vector
       @param Float val, Used to scale the results
       @param bool accum, A flag to choose if we want to accumulate or clear
     **/
    void oneEnd_trick(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r,
		      Float val , bool accum );

    /**
       @brief Computes the one-end trick (can do up to two derivatives) for disconnected quark loops as x_l^dag \gamma_5 \Gamma c_r.
       In case of standard one-end trick x_l = x_r = x while for generalized one-end trick x_l = x and x_r = \gamma_5 D_C x
       @param PLEGMA_Vector<Float> &x_l, left solution vector
       @param PLEGMA_Vector<Float> &x_r, right solution vector
       @param PLEGMA_Vector<Float> *tmp[16], 16 temp vectors to be used for the derivatives,one for one-Der and 16 for two-Der are used
       @param PLEGMA_QLoops<Float> *qLtmp, needed for shifting quark loops, Needed only for two derivatives 
       @param PLEGMA_Gauge<Float> &gauge, gauge field needed for the derivatives
       @param Float val, Used to scale the results
       @param bool accum, A flag to choose if we want to accumulate or clear
     **/
    void oneEnd_trick(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r,
		      PLEGMA_Vector<Float> *tmp[16], PLEGMA_QLoops<Float> *qLtmp, PLEGMA_Gauge<Float> &gauge, Float val , bool accum );
    /**
       @brief contracts two vectors over color space with gamma5 between allowing open the general gamma structure.
       @param PLEGMA_Vector<Float> &x_l, left solution vector
       @param PLEGMA_Vector<Float> &x_r, right solution vector
       @param bool accum, A flag to choose if we want to accumulate or clear
     **/
    void contractG5(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r, ACCUM_TYPE acc_type);
    /**
       @brief contracts two vectors over color space with gamma5 between allowing open the general gamma structure. No accumulation case
       @param PLEGMA_Vector<Float> &x_l, left solution vector
       @param PLEGMA_Vector<Float> &x_r, right solution vector
     **/
    void contractG5(PLEGMA_Vector<Float> &x_l, PLEGMA_Vector<Float> &x_r);

    void write_ASCII(std::string filename_local);
    void write_ASCII(std::string filename_local, std::string filename_oneD, std::string filename_oneDC);
    
    void load(Float* h_ptr);
    void clearAccumBuffs();

  };

}
#endif
