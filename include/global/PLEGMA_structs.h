//======== Some custom data struct =========//
template<typename Float> struct texture;

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
  void copyToDeviceConstant() {
    if(devPointer != NULL) {
      cudaMemcpyToSymbol( *((char**) devPointer), hostPointer, bytes*size);
      checkCudaError();
    }
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
  void print() {
    PLEGMA_printf("\nGlobal constants available only on host:\n");
    for(int i = 0; i < globals.size(); i++) {
      if(globals[i].devPointer != NULL) continue;
      std::string line = "HGC_" + globals[i].get_value();
      PLEGMA_printf(line.c_str());
    }
    PLEGMA_printf("\nGlobal constants available on both, host and device:\n");
    for(int i = 0; i < globals.size(); i++) {
      if(globals[i].devPointer == NULL) continue;
      std::string line = "H/DGC_" + globals[i].get_value();
      PLEGMA_printf(line.c_str());
    }
    PLEGMA_printf("\n\n");
  }
};
