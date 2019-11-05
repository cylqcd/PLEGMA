#include <PLEGMA_Field.h>
#pragma once

namespace plegma {
  /////////////////
  // This Class will be reponsible to momentum transfer fields
  //and write data in ASCII and HDF5 
  ////////////////
  enum FT_TYPE{FT_NAIVE,FT_GEMV,FT_FFT};
  template<typename Float>
  class PLEGMA_FT : public IO<void,int>  {
    using Vint = std::vector<int>;
    using VVint = std::vector<Vint>;
    using VFloat = std::vector<Float>;
    using VVFloat = std::vector<VFloat>;
  protected:
    int Q2_max;
    VVFloat momList;
    int dof; // degrees of freedom the field has
    std::shared_ptr<Float> h_elem; // memory to hold the transformed data
    int sizeN; // size of the elements array include real,imag
    int dims; // the dimensionality of the transformation either 3 or 4
    int dimT; // if dims = 3, dimT = (dims ==3) ? HGC_localL[DIM_T] : 1; 
    bool accum;
    std::string field_name;
    std::vector<int> site_shape;

    /**
       @brief Creates a list momenta which have the p^2 up to a specific value 
     **/
    void createMom();
    /**
       @brief performs the discrete fourier transform in the naive way using a simple custom kernel for reduction
       @param const PLEGMA_Field<Float> &f: the field we want to tranform in the momentum space
       @param int sign=-1: The sign of the FT, with default value the forward transformation
     **/
    void applyNaive(const PLEGMA_Field<Float> &f, int sign=-1);
    /**
       @brief performs the discrete fourier transform using Thrust to build a scalar field with momentum phases and then use cuBLAS to multiply the input field with field with the momentum phases.
       @param const PLEGMA_Field<Float> &f: the field we want to tranform in the momentum space
       @param int sign=-1: The sign of the FT, with default value the forward transformation
     **/
    void applyGEMV(const PLEGMA_Field<Float> &f, int sign=-1);      // transformation using THRUST for the momentum field and cuBLAS gemv for reduction
    /**
       @brief performs the Fast Fourier Transform. Not implemented yet but one must check if http://accfft.org/about/ can be used as external library
       @param const PLEGMA_Field<Float> &f: the field we want to tranform in the momentum space
       @param int sign=-1: The sign of the FT, with default value the forward transformation
     **/
    void applyFFT(const PLEGMA_Field<Float> &f, int sign=-1);  // use FFT in case in the future is implemented
  public:
    /**
       @brief Constructor of the FT class with max momentum value
       @params int Q2_max: Up to which momentum square we want to do the transformation
       @params int D3D4 = 3: The dimensionality of the FT, either 3 or 4 dimensions are supported
       @params bool accum = false: In case we want to accumulation results from each transformation on the class buffer
       @params bool dimT = HGC_localL[DIM_T]: Size of the time dimension in case we want to transform only part of the vector
     **/
    PLEGMA_FT(int Q2_max, int D3D4 = 3, bool accum = false, int dimT = HGC_localL[DIM_T]);
    /**
       @brief Constructor of the FT class with specific momentum vector
       @params std::vector<int> mom: Momentum vector, either 3 or 4 components based on the choice of D3D4
       @params int D3D4 = 3: The dimensionality of the FT, either 3 or 4 dimensions are supported
       @params bool accum = false: In case we want to accumulation results from each transformation on the class buffer
       @params bool dimT = HGC_localL[DIM_T]: Size of the time dimension in case we want to transform only part of the vector
     **/
    template<typename T>
    PLEGMA_FT(std::vector<T> mom, int D3D4 = 3, bool accum = false, int dimT = HGC_localL[DIM_T]);
    
    ~PLEGMA_FT() {};
    /**
       @brief First time a field is provided for transformation the FT object allocates memory. If field with same dof is provided then uses the same buffer otherwise has to reallocate memory for the new field.
       @params int newDof: The dof of the field we want to transform
     **/
    void checkAllocation(int newDof);
    /**
       @brief Accessor to the number of momenta the FT will do
     **/
    int Nmoms() const{ return momList.size();}
    /**
       @brief Accessor to the list where the components of each momentun are stored
     **/
    VVFloat MomList() const{ return momList;}
    /**
       @brief Clean the buffer of the class
     **/
    void zero();
    /**
       @brief Accessor to the dimensionality of the FT
     **/
    int Dims() const{return dims;}
    /**
       @brief Accessor. If field is 3D field dimT=1. If is 4D and the transformation is on 3D then dimT=localL[DIM_T], if it is a 4D transformation dimT=1
     **/
    int DimT() const{return dimT;}

    bool IsAccum() const{return accum;}
    
    Float* H_elem() const{return h_elem.get();}
    tex_mom_list getTexMomList();

    
    void apply(const PLEGMA_Field<Float> &f, FT_TYPE type = FT_GEMV, int sign = -1);

    /**
       @brief Put additional phases to the FT
       @params src: (\vec{src} \cdot \vec{p}) *(2*PI/L) where src is the source and p the momentum without (2*PI/L)
       @params sign: the appropriate size you want to put
     **/
    void mulConstMomentumPhases(Vint src, int sign);
    void scale(Float a);

    void writeFile(std::string filename, FILE_FORMAT format, int timeshift = 0) {
      return IO<void,int>::writeFile(filename,format,timeshift);
    }

    std::string Field_name() const {return field_name;}
    std::vector<int> getSiteShape() {return site_shape;}
    void setSiteShape(std::vector<int> new_shape) {
      int current_size=1, new_size = 1;
      std::for_each(site_shape.begin(), site_shape.end(), [&] (int n) {current_size *= n;});
      std::for_each(new_shape.begin(), new_shape.end(), [&] (int n) {new_size *= n;});
      assert(current_size==new_size);
      site_shape = new_shape;
    }
    std::string fill_H5_shapes(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start, int timeshift = 0);

    virtual void writeASCII(std::string filename, int timeshift = 0);
    virtual void writeHDF5(std::string filename, int timeshift = 0);
};
}
