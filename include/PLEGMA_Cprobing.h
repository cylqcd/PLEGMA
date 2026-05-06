#pragma once

namespace plegma {

inline std::vector<int> getRankCoord(int ind, const std::vector<int>& nProc){
  if(nProc.empty()) PLEGMA_error("Size of the vector is zero");
  if(ind < 0) PLEGMA_error("Ind provided is negative");

  int V = 1;
  for(size_t i = 0; i < nProc.size(); i++){
    if(nProc[i] <= 0) PLEGMA_error("One or more directions are zero or negative");
    V *= (int)nProc[i];
  }
  if(ind >= V) PLEGMA_error("The ind exceeds the total volume");

  std::vector<int> x(nProc.size(), 0);
  int cur = ind;
  for(int i = (int)nProc.size()-1; i >= 0; i--){
    x[i] = cur % nProc[i];
    cur /= nProc[i];
  }
  return x;
}

  class PLEGMA_Cprobing{
    private:
      int Nc; // Number of colors
      short probing_dimension; // Number of dimension of Cprob (For now probing_dimension={3,4})
      short coloring_distance; // Distance of coloring
      int* h_localColors; // array to hold the local colors for each MPI task on host
      int* d_localColors; // array to hold the local colors for each MPI task on device
      std::vector<int> sigma; // vector to hold the sigma values

      void graph_coloring(){
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::vector<int> lL = {HGC_localL[0], HGC_localL[1], HGC_localL[2], HGC_localL[3]}; // local lattice
        std::vector<int> nProc = {HGC_nProc[0], HGC_nProc[1], HGC_nProc[2], HGC_nProc[3]}; // MPI grid
        // if(probing_dimension == 3)
        //   get_sigma_3D();
        // else if(probing_dimension == 4)
        //   get_sigma_4D();
        // else PLEGMA_error("Classical probing supports only 3D and 4D coloring");
        PLEGMA_printf("Sigma is (%d,%d,%d,%d)\n",sigma[0],sigma[1],sigma[2],sigma[3]);
        std::vector<int> rank_coord = getRankCoord(rank, nProc); // Compute rank coordinates in MPI grid
        std::vector<int> proc_offset(4);
        for(int d=0; d<4; ++d) proc_offset[d] = rank_coord[d] * lL[d];
        for(size_t i=0; i < HGC_localVolume; i++){
          std::vector<int> x_local = getIndToVec(i, lL); // Local coordinates
          // Global coordinates
          std::vector<int> x_global(4);
          for(int d = 0; d < 4; d++)
            x_global[d] = x_local[d] + proc_offset[d];

          int col = sigma[0]*x_global[3] + sigma[1]*x_global[2] + sigma[2]*x_global[1] + sigma[3]*x_global[0]; // ordering xyzt
          col = (col % Nc) + 1;
          h_localColors[i] = col;
        }
      }

      bool load_sigma_from_file(const std::string& file,
                                int dim,
                                int distance,
                                std::vector<int>& sigma,
                                int& Nc)
      {
          int rank;
          MPI_Comm_rank(MPI_COMM_WORLD, &rank);

          if(rank == 0)
              PLEGMA_printf("Loading sigma config file: %s\n", file.c_str());

          std::ifstream f(file);
          if(!f.is_open()) {
              if(rank == 0)
                  PLEGMA_printf("ERROR: Could not open sigma config file: %s\n", file.c_str());
              return false;
          }

          // compute global lattice size
          int globalL[4];
          for(int i = 0; i < 4; i++)
              globalL[i] = HGC_localL[i] * HGC_nProc[i];

          std::string lattice_key =
              std::to_string(globalL[0]) + "x" +
              std::to_string(globalL[1]) + "x" +
              std::to_string(globalL[2]) + "x" +
              std::to_string(globalL[3]);

          if(rank == 0)
              PLEGMA_printf("Looking for lattice key %s (dim=%d, dist=%d)\n",
                            lattice_key.c_str(), dim, distance);

          std::string line;
          while(std::getline(f, line)) {

              if(line.empty() || line[0] == '#') continue;

              std::istringstream iss(line);

              std::string lat;
              int d, dist, nc;
              int s0, s1, s2, s3;

              if(!(iss >> lat >> d >> dist >> s0 >> s1 >> s2 >> s3 >> nc))
                  continue;

              if(lat == lattice_key && d == dim && dist == distance){
                  sigma = {s0, s1, s2, s3};
                  Nc = nc;

                  if(rank == 0)
                      PLEGMA_printf("Loaded sigma = (%d,%d,%d,%d), Nc = %d\n",
                                    s0, s1, s2, s3, nc);

                  return true;
              }
          }

          if(rank == 0)
              PLEGMA_printf("ERROR: No matching sigma found for key %s (dim=%d, dist=%d)\n",
                            lattice_key.c_str(), dim, distance);

          return false;
      }


    public:
      PLEGMA_Cprobing(int coloring_distance, int probing_dimension=4, const std::string& config_file = ""):Nc(0),coloring_distance(coloring_distance),probing_dimension(probing_dimension),sigma(4, 0){
        if(!HGC_init_PLEGMA_flag){ fprintf(stderr, "Error PLEGMA should be initialized before using this class"); exit(-1);}
        if(probing_dimension != 4 && probing_dimension != 3) PLEGMA_error("Classical probing supports only 3D and 4D coloring");
        PLEGMA_printf("Distance of neigbors is %d\n",coloring_distance);
        for(int i = 0 ; i < probing_dimension ; i++){
          if(coloring_distance >= HGC_localL[i]) PLEGMA_error("The coloring distance is larger than the lattice extent in direction %d\n",i);
        }
        h_localColors = (int*)malloc(HGC_localVolume * sizeof(int));

        // --- load sigma from config, or throw if missing
        if(config_file.empty()) PLEGMA_error("Config file must be specified for sigma values");
        bool loaded = load_sigma_from_file(config_file, probing_dimension, coloring_distance, sigma, Nc);
        if(!loaded) PLEGMA_error("Could not load sigma for this lattice/coloring distance from file: %s", config_file.c_str());

        graph_coloring();
        PLEGMA_printf("Number of colors for classical probing is %d\n",Nc);
        cudaMalloc((void**)&d_localColors, HGC_localVolume*sizeof(int));
        checkCudaError();
        cudaMemcpy(d_localColors, h_localColors, HGC_localVolume*sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError();    
      }
      ~PLEGMA_Cprobing(){
        free(h_localColors);
        cudaFree(d_localColors);
        checkCudaError();
      }
      int* H_localColors() const{return h_localColors;}
      int* D_localColors() const{return d_localColors;}
      int get_Ncol() const{return Nc;}
      int get_dimension() const { return probing_dimension; }

  };
}