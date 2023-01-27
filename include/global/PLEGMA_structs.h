//======== Some custom data struct =========//
template<typename Float> struct texture;

struct site : std::array<int,N_DIMS> {
  site() = default;
  site(const std::array<int,N_DIMS>& val) : std::array<int,N_DIMS>(val) {}

  operator int4() {
    return make_int4((*this)[0],(*this)[1],(*this)[2],(*this)[3]);
  }
  // TODO: add functions line to_ID, from_ID, etc
};
inline std::ostream& operator << (std::ostream &o, site &x){
  o<<x[0];
  for(int i=1; i<N_DIMS; i++)   o<<"-"<<x[i];
  return o;
}
inline std::istream& operator >> (std::istream &i, site &x){
  for(int j=0; j<N_DIMS; j++) {
    bool check = static_cast<bool> (i >> x[j]);
    if(!check) {
      PLEGMA_warning("Not enough arguments to unpack site\n");
      break;
    }
  }
  return i;
}


// Global variable for mom list
struct tex_mom_list {
  size_t Nmoms;

#if __HIP__
  hipTextureObject_t tex;
#else
  cudaTextureObject_t tex;
#endif
  void* devPtr;

  tex_mom_list() : Nmoms(0), tex(), devPtr(nullptr) {}

#ifdef __NVCC__
  tex_mom_list(size_t Nmoms, cudaTextureObject_t tex, void* devPtr) :
    Nmoms(Nmoms), tex(tex), devPtr(devPtr) {}
#elif defined (__HIP__)
  tex_mom_list(size_t Nmoms, hipTextureObject_t tex, void* devPtr) :
    Nmoms(Nmoms), tex(tex), devPtr(devPtr) {}
#endif

  
  inline __device__ int4 get(const size_t &i) const {
#ifdef __NVCC__
    return tex1Dfetch<int4>(tex,i);
#else
    return make_int4(0,0,0,0);
#endif
  };
};

struct pointer_holder {
  const size_t size;
  const size_t bytes;
  void * hostPointer;
  void ** devPointer;
  const std::string var_name;
  const std::string type_name;
  const char type_char;
  
  template<typename hostT>
  pointer_holder(const std::string& name, const size_t& size, hostT *host, void ** device) :
    size(size), bytes(sizeof(hostT)), hostPointer((void*) host), devPointer(device),
    var_name(name), type_name(plegma::type_name<hostT>()), type_char(plegma::type_char<hostT>()) { }

  void copyToDeviceConstant() {
    if(devPointer != nullptr) {
      //cudaError_t err = qudaMemcpyToSymbol( *devPointer, hostPointer, bytes*size);
      //if (err != cudaSuccess) {
        errorQuda("Failed to copy constant host memory of size to device %zu \n", size);
      //}
    }
  }
  void copyFromDeviceConstant() {
    if(false and devPointer != nullptr) {
      //cudaError_t err = qudaMemcpyFromSymbol(hostPointer, *devPointer, bytes*size, 0, cudaMemcpyDeviceToHost);
      //if (err != cudaSuccess) {
        errorQuda("Failed to copy constant host memory of size to device %zu \n", size);
      //}
    }
  }
  bool checkDeviceConstant() {
    if(false and devPointer != nullptr) {
      char tmp[bytes*size];
      memcpy(tmp,hostPointer,bytes*size);
      copyFromDeviceConstant();
      bool check=true;
      for(size_t i=0; i<bytes*size; i++) {
	if(tmp[i]!=*((char*)hostPointer + i)) {
	  check=false;
	  break;
	}
      }
      memcpy(hostPointer,tmp,bytes*size);
      return check;
    }
    return true;
  }
  std::string get_value() const {
    std::string line = var_name + " = (type: " + type_name + ", size: " + std::to_string(size) + (size==1 ? ", value:" : ", values:");
    char* tmp = (char*) hostPointer;
    for(size_t i = 0; i<size; i++) {
      line += " " + type_print((void*)(tmp + bytes*i), type_char, bytes);
    }
    line += ");\n";
    return line;
  }
};
  
// here we collect the global variables for then running some default functions on them (print and copy to device)
struct global_vars {
  std::vector<pointer_holder> globals;

  template<typename hostT>
  void add(const std::string& name, const size_t& size, hostT *host, void ** device = nullptr) {
    globals.push_back(pointer_holder(name, size, host, device));
  }
  void copyToDevice() {
    for(size_t i = 0; i != globals.size(); i++) {
      globals[i].copyToDeviceConstant();
    }
  }
  bool check() {
    for(size_t i = 0; i < globals.size(); i++) {
      if(globals[i].checkDeviceConstant() == false) return false;
    }
    return true;
  }
  void print() {
    PLEGMA_printf("\nGlobal constants available only on host:\n");
    for(size_t i = 0; i < globals.size(); i++) {
      if(globals[i].devPointer != nullptr) continue;
      std::string line = "HGC_" + globals[i].get_value();
      PLEGMA_printf("%s",line.c_str());
    }
    PLEGMA_printf("\nGlobal constants available on both, host and device:\n");
    for(int i = 0; i < globals.size(); i++) {
      if(globals[i].devPointer == nullptr) continue;
      if(globals[i].checkDeviceConstant()) {
	std::string line = "H/DGC_" + globals[i].get_value();
	PLEGMA_printf("%s",line.c_str());
      } else {
	PLEGMA_printf("!!!!!! ERROR: HGC_ and DGC_ differ in the following !!!!!!!\n");
	std::string line = "HGC_" + globals[i].get_value();
	PLEGMA_printf("%s",line.c_str());
	globals[i].copyFromDeviceConstant();
	line = "DGC_" + globals[i].get_value();
	PLEGMA_printf("%s",line.c_str());
      }
    }
    PLEGMA_printf("\n\n");
  }
};
