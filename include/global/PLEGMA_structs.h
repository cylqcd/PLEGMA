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

// here we collect the global variables for then running some default functions on them (print and copy to device)
struct global_vars {
  // globals on host only
  std::vector<void*> host_only_pointer;
  std::vector<int> host_only_size;
  std::vector<size_t> host_only_bytes;
  std::vector<char> host_only_type;
  std::vector<std::string> host_only_type_name;
  std::vector<std::string> host_only_name;
    
  // globals on both, host and device
  std::vector<std::array<void*, 2>> both_pointer;
  std::vector<int> both_size;
  std::vector<size_t> both_bytes;
  std::vector<char> both_type;
  std::vector<std::string> both_type_name;
  std::vector<std::string> both_name;

  template<typename hostT>
  void add(const char* name, hostT &host, int size=1) {
    host_only_pointer.push_back((void*) &host);
    host_only_size.push_back(size);
    host_only_bytes.push_back(sizeof(hostT)*size);
    host_only_type.push_back(type_char<hostT>());	
    host_only_type_name.push_back(demangle(typeid(host).name()));	
    host_only_name.push_back(name);	
  }
  template<typename hostT, typename deviceT>
  void add(const char* name, hostT &host, deviceT &device, int size=1) {
    both_pointer.push_back({(void*) &host, (void*) &device});
    both_size.push_back(size);
    both_bytes.push_back(sizeof(hostT)*size);
    both_type.push_back(type_char<hostT>());	
    both_type_name.push_back(type_name<hostT>());	
    both_name.push_back(name);
  }
  void copyToDevice() {
    for(int i = 0; i != both_pointer.size(); i++) {
      cudaMemcpyToSymbol( *((char**) both_pointer[i][1]), both_pointer[i][0], both_bytes[i]);
    }
    checkCudaError();
  }
  void print() {

  }
};
