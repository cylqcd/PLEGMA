//======== Some custom data struct =========//
template<typename Float> struct texture;

struct site {
  std::array<int,N_DIMS> x;
  site() = default;
  site(std::array<int,N_DIMS> y) : x(y) {}
  // TODO: add functions line to_ID, from_ID, etc
};
inline std::ostream& operator << (std::ostream &o, site &x){
  o<<x.x[0];
  for(int i=1; i<N_DIMS; i++)   o<<"-"<<x.x[i];
  return o;
}
inline std::istream& operator >> (std::istream &i, site &x){
  for(int j=0; j<N_DIMS; j++) {
    bool check = static_cast<bool> (i >> x.x[j]);
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
  cudaTextureObject_t tex;
  inline __device__ int4 get(size_t i){
#ifdef __NVCC__
    return tex1Dfetch<int4>(tex,i);
#else
    return make_int4(0,0,0,0);
#endif
  };
  void free(){
    cudaResourceDesc desc;
    cudaGetTextureObjectResourceDesc(&desc, tex);
    cudaFree(desc.res.linear.devPtr);
    cudaDestroyTextureObject(tex);
    checkCudaError();
    Nmoms=0;
  };
};

struct pointer_holder {
  void * hostPointer;
  void * devPointer;
  int size;
  int bytes;
  std::string var_name;
  std::string type_name;
  char type_char;
  
  template<typename hostT>
  pointer_holder(const char* name, hostT *host, int s=0) {
    hostPointer = (void*) host;
    devPointer = NULL;
    if(s<=0) size = sizeof(host)/sizeof(hostT);
    else size=s;
    bytes = sizeof(hostT);
    var_name = name;
    type_name = plegma::type_name<hostT>();
    type_char = plegma::type_char<hostT>();
  }
  template<typename hostT, typename devT>
  pointer_holder(const char* name, hostT *host, devT *dev, int s=0) : pointer_holder(name, host, s) {
    devPointer = (void*) dev;
  }
  void copyToDevice() {
    if(devPointer != NULL) {
      cudaMemcpy( devPointer, hostPointer, bytes*size, cudaMemcpyHostToDevice);
      checkCudaError();
    }
  }
  void copyFromDevice() {
    if(devPointer != NULL) {
      cudaMemcpy( hostPointer, devPointer, bytes*size, cudaMemcpyDeviceToHost);
      checkCudaError();
    }
  }
  void copyToDeviceConstant() {
    if(devPointer != NULL) {
      cudaMemcpyToSymbol( *((char**) devPointer), hostPointer, bytes*size);
      checkCudaError();
    }
  }
  void copyFromDeviceConstant() {
    if(devPointer != NULL) {
      cudaMemcpyFromSymbol(hostPointer, *((char**) devPointer), bytes*size);
      checkCudaError();
    }
  }
  bool checkDeviceConstant() {
    if(devPointer != NULL) {
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
  std::string get_value() {
    std::string line = var_name + " = (type: " + type_name + ", size: " + std::to_string(size) + (size==1 ? ", value:" : ", values:");
    char* tmp = (char*) hostPointer;
    for(int i = 0; i<size; i++) {
      line+= " " + type_print((void*)(tmp + bytes*i), type_char, bytes);
    }
    line += ");\n";
    return line;
  }
};
  
// here we collect the global variables for then running some default functions on them (print and copy to device)
struct global_vars {
  std::vector<pointer_holder> globals;

  template<typename hostT>
  void add(const char* name, hostT &host, int size) {
    globals.push_back(pointer_holder(name,&host,size));
  }
  template<typename hostT, typename deviceT>
  void add(const char* name, hostT &host, deviceT &device, int size) {
    globals.push_back(pointer_holder(name,&host,&device, size));
  }
  void copyToDevice() {
    for(int i = 0; i != globals.size(); i++) {
      globals[i].copyToDeviceConstant();
    }
  }
  bool check() {
    for(int i = 0; i < globals.size(); i++) {
      if(globals[i].checkDeviceConstant() == false) return false;
    }
    return true;
  }
  void print() {
    PLEGMA_printf("\nGlobal constants available only on host:\n");
    for(int i = 0; i < globals.size(); i++) {
#ifdef __NVCC__
      if(globals[i].devPointer != NULL) continue;
#endif
      std::string line = "HGC_" + globals[i].get_value();
      PLEGMA_printf(line.c_str());
    }
#ifdef __NVCC__
    PLEGMA_printf("\nGlobal constants available on both, host and device:\n");
    for(int i = 0; i < globals.size(); i++) {
      if(globals[i].devPointer == NULL) continue;
      if(globals[i].checkDeviceConstant()) {
	std::string line = "H/DGC_" + globals[i].get_value();
	PLEGMA_printf(line.c_str());
      } else {
	PLEGMA_printf("!!!!!! ERROR: HGC_ and DGC_ differ in the following !!!!!!!\n");
	std::string line = "HGC_" + globals[i].get_value();
	PLEGMA_printf(line.c_str());
	globals[i].copyFromDeviceConstant();
	line = "DGC_" + globals[i].get_value();
	PLEGMA_printf(line.c_str());
      }
    }
#endif
    PLEGMA_printf("\n\n");
  }
};
