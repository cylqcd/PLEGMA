#include <PLEGMA_Field.h>
#include <PLEGMA_Vector.h>
#include <PLEGMA_Propagator.h>

#pragma once

namespace plegma {
  /////////////////
  // This PLEGMA_Tensor Class allows to store a generic number of d.o.f
  // (divided into spin and color index) per site. It  
  // will contain all the needed contractions V1, V2, V3, V4, T1, T2 plus
  // other functionalities ( product of tensors along given axis, etc )
  ////////////////
  template<typename Float>
  class PLEGMA_Tensor : public PLEGMA_Field<Float>  {
  protected:
    // List not completed
    // For example, imagine to have T^a_{alpha,beta,gamma}. Then
    std::string index_struct;    //     index_struct = "sssc" (because spin first)

    // probably already present in PLEGMA_Fields
    std::vector<int> index_size; //     index_size   = {4,4,4,3}
    int site_size;               //     site_size    = 4*4*4*3
    //
    
    int n_index;                 //     n_index      = 4

    bool isAllocated;
    
  public:
    // this constructor ALLOCATES the PLEGMA_Tensor here. Could be useful if
    // one needs to absorb from other objects. 
    PLEGMA_Tensor(ALLOCATION_FLAG alloc_flag, std::string index_struct);

    // this constructor does NOT ALLOCATE the memory PLEGMA_Tensor here, because
    // the dimension is not provided. It will be allocated when used.
    PLEGMA_Tensor(ALLOCATION_FLAG alloc_flag);

    ~PLEGMA_Tensor();

    void checkAllocation(int newDof);

    //functions that return values of protected variables
    std::string Index_struct() const{ return index_struct;}
    int N_index() const{ return n_index;}
    bool IsAllocated() const{ return isAllocated}

    void V1( PLEGMA_Vector<Float> Phi, PLEGMA_Propagator<Float> S);
    void V2( PLEGMA_Tensor<Float> V1, PLEGMA_Propagator<Float> S);
    void V2( PLEGMA_Vector<Float> Phi, PLEGMA_Propagator<Float> S1,  PLEGMA_Propagator<Float> S2);
    void V3( PLEGMA_Vector<Float> Phi, PLEGMA_Propagator<Float> S);
    void V4( PLEGMA_Vector<Float> Phi, PLEGMA_Propagator<Float> S1,  PLEGMA_Propagator<Float> S2 );


  };
}
