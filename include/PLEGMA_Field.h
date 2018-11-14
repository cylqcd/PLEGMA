#include <PLEGMA_global.h>

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
    int total_plus_ghost_length;

    size_t bytes_total_length;
    size_t bytes_ghost_length;
    size_t bytes_total_plus_ghost_length;

    Float *h_elem;
    Float *d_elem;
    Float *h_ext_ghost;
    Float *h_elem_backup;

    bool isAllocHost;
    bool isAllocDevice;
    bool isAllocHostBackup;

    void create_host();
    void create_host_backup();
    void destroy_host();
    void destroy_host_backup();
    void create_device();
    void destroy_device();

  public:
    PLEGMA_Field(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT);
    virtual ~PLEGMA_Field();
    void zero_host();
    void zero_host_backup();
    void zero_device();
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
    void ghostToHost(int dirOr=-1);
    void cpuExchangeGhost(int dirOr=-1);
    void ghostToDevice();

    void shift(PLEGMA_Field &Fin, int dirOr);
  };
}
#endif
