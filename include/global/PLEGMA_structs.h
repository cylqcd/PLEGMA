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

struct var_holder {
  void * hostPointer;
  void * devPointer;
  int size;
  int bytes;
  std::string var_name;
  std::string type_name;
  std::string type_print;
  
  template<typename hostT>
  var_holder(const char* name, hostT &host, int s=0) {
    hostPointer = (void*) &host;
    devPointer = NULL;
    if(s==0) size = sizeof(host)/sizeof(hostT);
    else size=s;
    bytes = sizeof(hostT);
    var_name = name;
    type_name = plegma::type_name<hostT>();
    type_print = type_char<hostT>();
  }
  template<typename hostT, typename devT>
  var_holder(const char* name, hostT &host, devT &dev, int s=0) : var_holder(name, host, s) {
    devPointer = (void*) &dev;
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
};
  
// here we collect the global variables for then running some default functions on them (print and copy to device)
struct global_vars {
  std::vector<var_holder> globals;

  template<typename hostT>
  void add(const char* name, hostT &host, int size=0) {
    globals.push_back(var_holder(name,host,size));
  }
  template<typename hostT, typename deviceT>
  void add(const char* name, hostT &host, deviceT &device, int size=1) {
    globals.push_back(var_holder(name,host,device, size));
  }
  void copyToDevice() {
    for(int i = 0; i != globals.size(); i++) {
      globals[i].copyToDeviceConstant();
    }
  }
  void print() {
    /*
    printfQuda("Global constants available only on host:\n");
    for(int i = 0; i != host_only_pointer.size(); i++) {
      std::string line = "HGC_" + host_only_name[i] + ", type: " + host_only_type_name[i] + ", size: " + std::to_string(host_only_size[i]) + ", values:";
      int bytes = host_only_bytes[i] / host_only_size[i]; 
      for(int j=0; j<host_only_size[i]; j++) {
	char * value, type[] = " %d";
	type[2] = host_only_type[i];
	asprintf(&value, type, *((char *) host_only_pointer[i] + j*bytes));
	line += value;
	free(value);
      }
      line+="\n";
      printfQuda(line.c_str());
    }
    */
  }
};
