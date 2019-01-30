#include <PLEGMA_global.h>
#include <PLEGMA_Random.h>
#include <vector>
#ifndef _PLEGMA_FIELD_H
#define _PLEGMA_FIELD_H

namespace plegma {
  ////////////////////////
  // CLASS: PLEGMA_Field //
  ////////////////////////
  
  template<typename Float>
  class PLEGMA_Field {
  protected:
    
    int field_length;
    int total_length;        
    int ghost_length;
    int ghost_corner_length;
    int total_plus_ghost_length;

    size_t bytes_total_length;
    size_t bytes_ghost_length;
    size_t bytes_ghost_corner_length;
    size_t bytes_total_plus_ghost_length;
    
    Float *h_elem;
    Float *d_elem;
    Float *h_ext_ghost_r;
    Float *h_ext_ghost_s;
    Float *h_ext_ghost_corner_r;
    Float *h_ext_ghost_corner_s;
    PLEGMA_RNG *randstate_ptr;
    
    GHOST_FLAG ghost_flag;
    ALLOCATION_FLAG allocation;
    bool isAllocHost;
    bool isAllocDevice;

    
    void create_host();
    void destroy_host();
    void create_device();
    void destroy_device();

  public:
    PLEGMA_Field(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT, GHOST_FLAG ghost_flag=NO_GHOSTS);
    virtual ~PLEGMA_Field();
    void zero_host();
    void zero_host_backup();
    void zero_device();
    void zero_where(ALLOCATION_FLAG alloc_flag);
    cudaTextureObject_t createTexObject();
    void destroyTexObject(cudaTextureObject_t tex);
    
    Float* H_elem() const { return h_elem; }
    Float* D_elem() const { return d_elem; }

    size_t Bytes_total() const { return bytes_total_length; }
    size_t Bytes_ghost() const { return bytes_ghost_length; }
    size_t Bytes_total_plus_ghost() const { return bytes_total_plus_ghost_length; }

    int Field_length() const { return field_length;} // degrees of freedom per lattice point
    int Total_length() const { return total_length;} // the length of the field (local)
    int Ghost_length() const { return ghost_length;} // the length of the ghost
    int TotalGhost_length() const { return total_plus_ghost_length;} // total + ghost
    
    int Precision() const{
      if( typeid(Float) == typeid(float) )
	return 4;
      else if( typeid(Float) == typeid(double) )
	return 8;
      else
	return 0;
    } 
    void printInfo();
    void communicateSideGhost(int dirOr=-1);
    void communicateCornerGhost(int dirOr=-1);
    void communicateGhost(int dirOr, GHOST_FLAG which_ghost);
    void communicateGhost(int dirOr=-1);
    
    void pack(Float *topack);
    void unpack(Float *out);

    void load();
    void unload();
    
    void shift(PLEGMA_Field &Fin, int dirOr);
    void randInit(int seed);
    void destroy_randstate();
    void stochastic_Z(int n=2);
    void random(DIST sampling=Uniform);
    void setUnit(std::vector<int> indDiag);
    void mulMomentumPhases(std::vector<int> mom, int sign=-1);
    void dot(Float res[2], PLEGMA_Field<Float> &FieldIn);    
  };
}
#endif
