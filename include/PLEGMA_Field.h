#include <PLEGMA_global.h>
#include <PLEGMA_Random.h>
#include <PLEGMA_Hprobing.h>
#include <PLEGMA_io.h>
#include <vector>
#ifndef _PLEGMA_FIELD_H
#define _PLEGMA_FIELD_H

namespace plegma {
  template<typename Float>  class PLEGMA_Field3D;
  template<typename Float>  class PLEGMA_Fmunu;
  template<typename Float>  class PLEGMA_Su3field;
  template<typename Float>  class PLEGMA_Gauge;

  ////////////////////////
  // CLASS: PLEGMA_Field //
  ////////////////////////
  /**
     @brief The parent class of all PLEGMA fields
   **/
  template<typename Float>
  class PLEGMA_Field : virtual public IO<void,bool> {
  protected:
    
    int field_length; /*!< Member variable to hold the degrees of freedom of a field eg. (spin,color,...) */
    size_t total_length; /*!< Member variable to hold the size of a field only lattice points excluding the d.o.f per lattice point */
    size_t ghost_length; /*!< Member variable to hold the size of ghosts that involve in the communication (only side ghosts) */
    size_t ghost_corner_length; /*!< Member variable to hold the size of ghosts that involve in the communication (ghosts which are on the corners) */
    size_t ghost_vertex_length;
    
    Float *h_elem; /*!< Member variable pointer to the elements of the field on CPU */
    Float *d_elem; /*!< Member variable pointer to the elements of the field on GPU */
    Float *h_ext_ghost_r; /*!< Member variable pointer to the side ghost elements of the field on CPU (receive version)*/
    Float *h_ext_ghost_s; /*!< Member variable pointer to the side ghost elements of the field on CPU (send version)*/
    Float *h_ext_ghost_corner_r; /*!< Member variable pointer to the corner ghost elements of the field on CPU (receive version)*/
    Float *h_ext_ghost_corner_s; /*!< Member variable pointer to the corner ghost elements of the field on CPU (send version)*/
    Float *h_ext_ghost_vertex_r;
    Float *h_ext_ghost_vertex_s;
    PLEGMA_RNG *randstate_ptr; /*!< Member variable a PLEGMA_RNG allowing a field to create random numbers */
    
    GHOST_FLAG ghost_flag; /*!< Choose what kind of ghosts the field has, options (NO_GHOSTS,FIRST_SIDE,FIRST_CORNER)*/
    ALLOCATION_FLAG allocation; /*!< Choose what kind of memory allocation the field has, options (HOST,DEVICE,BOTH,NONE)*/
    bool isPinnedHost; /*!< Allows for pinned memory allocation on CPU memory for fast memory exhange with GPU*/
    bool isAllocHost; /*!< Flag to determine if allocation is done on the CPU */
    bool isAllocDevice; /*!< Flag to determine if allocation is done on the GPU */
    bool checkErr; /*!< Controls if we want to check for errors in the PLEGMA_Fields, default is true */
    std::vector<MsgHandle*> messages;

    CLASS_ENUM field_type; /*!< Enum for the names of fields */
    std::string field_name; /*!< Hold in string the name of the field */
    std::vector<int> site_shape; /*!< A list of sites of a field. May be dofs, time slices,.... */

    /**
       @brief allocates memory on host
     */
    void create_host();
    /**
       @brief deallocates memory on host
     */
    void destroy_host();
    /**
       @brief allocates memory on device
     */
    void create_device();
    /**
       @brief deallocates memory on device
     */
    void destroy_device();
    /**
       @brief Called from the constructor to create a field, by initializing member variables and handle allocation
       @param alloc_flag: see "allocation"
       @param field_l: see "field_length"
       @param vol_l: see "total_length"
     */
    void initialize(ALLOCATION_FLAG alloc_flag, int field_l, size_t vol_l);
  public:
    /**
       @brief Constructor to create a standard field by calling "initialize(...)"
       @param alloc_flag: see "allocation"
       @param classT: What kind of field to create based on enum name
       @param ghost_flag: Controls what kind of ghosts can be exchanged
       @param isPinnedHost: see "isPinnedHost"
     */
    PLEGMA_Field(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT, GHOST_FLAG ghost_flag=NO_GHOSTS, bool isPinnedHost = false, bool checkErr = true);
    /**
       @brief Overloaded constructor to create a general field by calling "initialize(...)"
       @param alloc_flag: see "allocation"
       @param site_size: Provide d.o.fs of the field
       @param ghost_flag: Controls what kind of ghosts can be exchanged
       @param isPinnedHost: see "isPinnedHost"
       @param checkErr: see "checkErr"
     */
    PLEGMA_Field(ALLOCATION_FLAG alloc_flag, int site_size, size_t localV = HGC_localVolume, GHOST_FLAG ghost_flag=NO_GHOSTS, bool isPinnedHost = false, bool checkErr = true);
    /**
       @brief virtual destructor responsible for freeing memory. Virtual because it could be called from an instance of a derived class through a pointer to base class
     */
    ~PLEGMA_Field();
    // Deleting copy contructor at the moment. This would cause seg fault due to the fields allocated
    PLEGMA_Field(const PLEGMA_Field<Float>&) = delete;
    /**
       @brief sets to zero all element of the field on CPU
     */
    void zero_host();
    void zero_host_backup();
    /**
       @brief sets to zero all element of the field on GPU
     */
    void zero_device();
    /**
       @brief sets to zero all element of the field on the chosen location
     */
    void zero_where(ALLOCATION_FLAG alloc_flag);
    /**
       @brief Creates a texture object which binds on field elements on GPU
     */
#ifdef __NVCC__
    cudaTextureObject_t createTexObject() const;
#elif defined (__HIP__)
    hipTextureObject_t createTexObject() const;
#endif
    /**
       @brief Destroys the texture object which binds on field elements on GPU
     */
#ifdef __NVCC__
    void destroyTexObject(cudaTextureObject_t tex) const;
#elif defined (__HIP__)
    void destroyTexObject(hipTextureObject_t tex) const;
#endif

    /**
     * @return a pointer to access Host elements of the field
     */
    Float* H_elem() const { return h_elem; }
    /**
     * @return a pointer to access device elements of the field
     */
    Float* D_elem() const { return d_elem; }
    /**
     * @return a boolean if the host memory is allocated
     */
    bool IsAllocHost() const { return isAllocHost;}
    /**
     * @return a boolean if the device memory is allocated
     */
    bool IsAllocDevice() const { return isAllocDevice;}
    /**
     * @return the kind of allocation we have for the field
     */
    ALLOCATION_FLAG getAllocation() const { return allocation; }

    /**
     * @return degrees of freedom per lattice point
     */
    int Field_length() const { return field_length;} // degrees of freedom per lattice point
    /**
     * @return the length of the field in lattice points (local)
     */
    size_t Total_length() const { return total_length;} // the length of the field (local)
    /**
     * @return the length of the ghost part of the field in lattice points (local)
     */
    size_t Ghost_length() const { return ghost_length;} // the length of the ghost
    size_t GhostCorner_length() const { return ghost_corner_length;} // the length of the ghost for corners
    size_t GhostVertex_length() const { return ghost_vertex_length;} // the length of the ghost for vertex
    /**
     * @return the length of the field + ghost in lattice points (local)
     */
    size_t TotalPlusGhost_length() const { return Total_length()+Ghost_length()+GhostCorner_length()+GhostVertex_length();} // total + ghost

    /**
     * @return the bytes the length of the field including d.o.f
     */
    size_t Bytes_total() const { return this->Total_length()*this->Field_length()*2*sizeof(Float); }
    /**
     * @return the bytes the length of the ghost including d.o.f
     */
    size_t Bytes_ghost() const { return this->Ghost_length()*this->Field_length()*2*sizeof(Float); }
    size_t Bytes_ghostCorner() const { return this->GhostCorner_length()*this->Field_length()*2*sizeof(Float); }
    size_t Bytes_ghostVertex() const { return this->GhostVertex_length()*this->Field_length()*2*sizeof(Float); }
    /**
     * @return the bytes the length of the field including ghost including d.o.f
     */
    size_t Bytes_total_plus_ghost() const { return this->TotalPlusGhost_length()*this->Field_length()*2*sizeof(Float); }

    template<class... Args>
    bool checkVolume(const Args&... fields) {
      std::vector<bool> checks = {this->Total_length() == fields.Total_length() ...};
      return std::all_of(checks.begin(), checks.end(), [](bool i){return i;});
    }
    
    /**
     * @return a string with the name of the field
     */
    std::string Field_name() const {return field_name;}
    /**
     * @return the ghost flag about which ghost we will be able to exchange
     */
    GHOST_FLAG Ghost_flag() const {return ghost_flag;}
    /**
     * @return the precision of the field in bytes
     */
    int Precision() const{
      if( typeid(Float) == typeid(float) )
	return 4;
      else if( typeid(Float) == typeid(double) )
	return 8;
      else
	return 0;
    }
    /**
     * @brief Prints some info about the field such as (precision, size in bytes, and allocation type)
     */
    void printInfo();
    /**
     * @brief Communicates only the side ghosts in chosen direction,orientation
     * @param dirOr: choose the dir,orien. If negative does all dir, orien. If >=0 then (0,1,2,3,4,5,6,7,8) -> (+x,+y,+z,+t,-x,-y,-z,-t)
     */
    void communicateSideGhost(short dir=-1, ORIENTATION sign=DIR_BOTH, ACTION action=DO_ALL);
    /**
     * @brief Communicates side+corner ghosts in chosen direction,orientation
     * @param dirOr: choose the dir,orien. If negative does all dir, orien. If >=0 then (0,1,2,3,4,5,6,7,8) -> (+x,+y,+z,+t,-x,-y,-z,-t)
     */
    void communicateCornerGhost(short dir=-1, ORIENTATION sign=DIR_BOTH, ACTION action=DO_ALL);
    void communicateVertexGhost(short dir=-1, ORIENTATION sign=DIR_BOTH, ACTION action=DO_ALL);
    /**
     * @brief Communicates side or side+corner ghosts in chosen direction,orientation give the ghost type
     * @param dirOr: choose the dir,orien. If negative does all dir, orien. If >=0 then (0,1,2,3,4,5,6,7,8) -> (+x,+y,+z,+t,-x,-y,-z,-t)
     * @param which_ghost: choose what kind of ghost to exchange
     */
    void communicateGhost(short dir=-1, ORIENTATION sign=DIR_BOTH, GHOST_FLAG which_ghost=ALL_GHOSTS, ACTION action=DO_ALL);

    std::vector<int> getSiteShape() const {return site_shape;}
    void setSiteShape(std::vector<int> new_shape) {
      int current_size=1, new_size = 1;
      std::for_each(site_shape.begin(), site_shape.end(), [&] (int n) {current_size *= n;});
      std::for_each(new_shape.begin(), new_shape.end(), [&] (int n) {new_size *= n;});
      assert(current_size==new_size);
      site_shape = new_shape;
    }
    std::string fill_H5_shapes(std::vector<hsize_t> &shape, std::vector<hsize_t> &lshape, std::vector<hsize_t> &start) const;
    /**
     * @brief On the host pack the data from d.o.fs running faster to d.o.fs running slower. Common practice before copy data to GPU
     * @param Array with input data to pack
     */
    void pack(Float *topack);
    /**
     * @brief On the host unpack the data from d.o.fs running slower to d.o.fs running faster. 
     * @param Array with output data after unpack
     */
    void unpack(Float *out);
    /**
     * @brief Load on the GPU data. Just memcpy without changing data layout
     */
    void load();
    /**
     * @brief Download from the GPU the data. Just memcpy without changing data layout
     */
    void unload() const;
    /**
     * @brief shift the field by one step in direction,orientation. (0,1,2,3) Push the field in (+x,+y,+z,+t) while (4,5,6,7) push the field in (-x,-y,-z,-t)
     */
    void shift(PLEGMA_Field &Fin, short dirOr);
    void shift(PLEGMA_Field &Fin, short dirOr1, short dirOr2);
    void shift(PLEGMA_Field &Fin, short dirOr1, short dirOr2, short dirOr3);
    /**
     * @brief Initialize the random number generator of the field. Includes memory allocation
     * @param The seed will be used for the random number generator
     */
    void randInit(int seed);
    /**
     * @brief Destroyes the random number generator if is not needed anymore to free memory
     */
    void destroy_randstate();
    /**
     * @brief creates stochastic sources from Z_n the roots of one
     * @param n: the number of roots of one
     */
    void stochastic_Z(int n=2);
    /**
     * @brief Fills the field with random numbers 
     * @param sampling: Draw random numbers either from uniform or gaussian distribution
     */
    void random(DIST sampling=Uniform);
    /**
     * @brief Puts the chosen d.o.fs to units e.g. For Su3 field indOne ={0,4,8};
     * @param indDiag: a std vector with the chosen d.o.f indices
     */
    void setUnit(std::vector<int> indDiag);
    void conjugate();
    /**
     * @brief copies the elements from one field to another on the desired location. If the two fields do not have matched field_length an error is thrown. If the precision is not match a casting is used.
     * @param where: the location where we copy from-to
     */
    template<typename FloatIn>
    void copy(PLEGMA_Field<FloatIn> &f, ALLOCATION_FLAG where=DEVICE);
    /**
     * @brief multiply the elements of the field with position dependent momentum phases. 
     * @param sign: the sign on the exponential
     */
    template<typename FloatMom>
    void mulMomentumPhases(std::vector<FloatMom> mom, int sign=-1);
    /**
     * @brief Adds two fields where the input field is scaled by complex number and the results is stored on "this" (F += a*Fin)
     * @param alpha: complex number scales input field
     */
    void add(PLEGMA_Field &Fin, std::complex<Float> alpha = 1.);
    /**
     * @brief The dot product of two complex fields 
     * @param FieldIn: Right side field
     * @return complex number cdot(this,FieldIn)
     */
    std::complex<Float> dot(PLEGMA_Field<Float> &FieldIn);    
    /**
     * @brief Computes the norm of the field ||^2 
     * @return real number which is the norm
     */
    Float norm();
    /**
     * @brief Scales the field by a real number
     * @param the scaling value
     */
    void scale(Float val);
    /**
     * @brief Scales the field by a complex number
     * @param the scaling value
     */
    void cscale(std::complex<Float> val);
    /**
     * @brief Given a hierarchical coloring this multiplies Hadamard vectors to a field
     * @param fin: Input field 
     * @param hprob: Hierarchical coloring
     * @param ih: index to move over the hadamard vectors
     * @param indDof: choose the d.o.fs where the application of the Hadamard vectors will take place
     */
    void applyHpropColoring4D(PLEGMA_Field<Float> &fin,PLEGMA_Hprobing &hprob, int ih, std::vector<int> indDof);
    void absorbTimeslice(PLEGMA_Field<Float> &srcfield, int global_it, bool forcetozero=true);
    /**
     * @brief Contracts two gluon field strength tensors (FST) connected SU3 matrices and then take a trace
     * @param Fl: FST on the left
     * @param munu_l: the mu,nu pair of lorentz indices for the left FST
     * @param Wl: the left SU3 line that connects Fl -> Fr
     * @param Fr: FST on the right
     * @param munu_r: the mu,nu pair of lorentz indices for the right FST
     * @param Wr: the right SU3 line that connects Fr -> Fl
     */
    void TrFmunuSu3FmunuSu3(PLEGMA_Fmunu<Float> &Fl, std::pair<int,int> munu_l, PLEGMA_Su3field<Float> &Wl,
			    PLEGMA_Fmunu<Float> &Fr, std::pair<int,int> munu_r, PLEGMA_Su3field<Float> &Wr);
    /**
     * @brief Plaquette definition of the gluon loops  
     * @param munu: the mu,nu pair of lorentz indices of the Plq
     */
    void trPmunu(PLEGMA_Gauge<Float> &gauge, std::pair<int,int> munu);
    
    void SU3Trace(PLEGMA_Su3field<Float> &su3field);

    /**
       @brief Absorbs all  elements from a 3D field and puts it at a specific global time of the 4D field
       @param PLEGMA_Field3D<Float> prop, The 3D field
       @param int global_it, The global time slice where data which will be inserted, the rest of the time-slices will become zero in the 4D field
       @return void
     **/    
    void absorb(const PLEGMA_Field3D<Float> &field, int global_it, bool forcetozero=true);
    /**
       @brief Multiplies a field with theta twists in temporal direction, namely e^{i \theta \pi t/T}
       @param double theta: the parameter \theta as used above
       @param bool dagger: If true flips the sign in the exponential
     **/
    void mulThetaPhase(Float theta, bool dagger=false);

    /**
     * @brief Read from lime a field
     */
    virtual void readLIME(std::string filename, bool loadToDev=true);
    /**
     * @brief Write to lime a field
     */
    virtual void writeLIME(std::string filename, bool unloadFromDev=true) const;
    /**
     * @brief Write to hdf5 a field
     */
    virtual void writeHDF5(std::string filename, bool unloadFromDev=true) const;
    void readFile(std::string filename, FILE_FORMAT format, bool loadToDev=true) {
      IO<void,bool>::readFile(filename, format, loadToDev);
    }
    void readFile(std::string filename, bool loadToDev=true) {
      IO<void,bool>::readFile(filename, loadToDev);
    }
    void writeFile(std::string filename, FILE_FORMAT format, bool unloadFromDev=true) const {
      IO<void,bool>::writeFile(filename, format, unloadFromDev);
    }
    void writeFile(std::string filename, bool unloadFromDev=true) const{
      IO<void,bool>::writeFile(filename, unloadFromDev);
    }

    virtual bool includesActiveTimeSlice() const{return true;}
    virtual bool is4D() const{assert(Total_length()==HGC_localVolume); return true;}
  };



  /**
     @brief A child class of Field specialized for 3D fields
   **/
  template<typename Float>
  class PLEGMA_Field3D : virtual public PLEGMA_Field<Float> {
  public:
    bool activeTimeSlice; /*!< Becomes true if the global time slice is active on the current process */
    
    PLEGMA_Field3D(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT, GHOST_FLAG ghost_flag=NO_GHOSTS, bool isPinnedHost = false, bool checkErr = true) :
      PLEGMA_Field<Float>(alloc_flag, classT, ghost_flag, isPinnedHost, checkErr), activeTimeSlice(false) { }
    PLEGMA_Field3D(ALLOCATION_FLAG alloc_flag, int site_size, GHOST_FLAG ghost_flag=NO_GHOSTS, bool isPinnedHost = false, bool checkErr = true) :
      PLEGMA_Field<Float>(alloc_flag, site_size, HGC_localVolume3D, ghost_flag, isPinnedHost, checkErr), activeTimeSlice(false) { }
    PLEGMA_Field3D() : PLEGMA_Field<Float>(NONE, 0, HGC_localVolume3D), activeTimeSlice(false) { }

    virtual bool includesActiveTimeSlice() const{return activeTimeSlice;}
    virtual bool is4D() const{assert(this->Total_length()==HGC_localVolume3D); return false;}

    template<typename FloatIn>
    void copy(PLEGMA_Field3D<FloatIn> &f, ALLOCATION_FLAG where=DEVICE) {
      this->activeTimeSlice = f.activeTimeSlice;
      return ((PLEGMA_Field<Float>*) this)->copy(f,where);
    }
    
    /**
       @brief Absorbs a time-slice from a 4D field to a 3D field
       @param PLEGMA_Field<Float> field, The 4D field
       @param int global_it, The global time slice from where data will be extracted from the the 4D field
       @return void
     **/    
    void absorb(const PLEGMA_Field<Float> &field, int global_it, bool broadcast=false);
    std::complex<Float> dot(PLEGMA_Field3D<Float> &fieldIn);
  };
}
#endif
