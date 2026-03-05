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
        if(probing_dimension == 3)
          get_sigma_3D();
        else if(probing_dimension == 4)
          get_sigma_4D();
        else PLEGMA_error("Classical probing supports only 3D and 4D coloring");
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

      void get_sigma_4D(){ // works only for 64x32^3
  
        if(coloring_distance == 0){
          sigma[0] = 0;
          sigma[1] = 0;
          sigma[2] = 0;
          sigma[3] = 0;
          
          Nc = 1;
        }

        if(coloring_distance == 1){ // fix index ordering to xyzt
          sigma[0] = 1;
          sigma[1] = 1;
          sigma[2] = 1;
          sigma[3] = 1;
          
          Nc = 2;
        }
        
        if(coloring_distance == 2){
          sigma[0] = 1;
          sigma[1] = 2;
          sigma[2] = 3;
          sigma[3] = 4;
          
          Nc = 10;
        }
        
        if(coloring_distance == 3){
          sigma[0] = 1;
          sigma[1] = 5;
          sigma[2] = 55;
          sigma[3] = 61;
          
          Nc = 16;
        }
        
        if(coloring_distance == 4){
          sigma[0] = 1;
          sigma[1] = 8;
          sigma[2] = 12;
          sigma[3] = 18;
          
          Nc = 64;
        }

        if(coloring_distance == 5){
          sigma[0] = 1;
          sigma[1] = 12;
          sigma[2] = 16;
          sigma[3] = 38;
          
          Nc = 128;
        }

        if(coloring_distance == 6){
          sigma[0] = 3;
          sigma[1] = 20;
          sigma[2] = 48;
          sigma[3] = 50;
          
          Nc = 320;
        }

        if(coloring_distance == 7){
          sigma[0] = 32;
          sigma[1] = 33;
          sigma[2] = 40;
          sigma[3] = 61;
          
          Nc = 512;
        }
      }

      void get_sigma_3D(){  // works only for 32^3
        
        if(coloring_distance == 0){
          sigma[0] = 0;
          sigma[1] = 0;
          sigma[2] = 0;
          sigma[3] = 0;
          
          Nc = 1;
        }

        if(coloring_distance == 1){
          sigma[0] = 0;
          sigma[1] = 1;
          sigma[2] = 1;
          sigma[3] = 1;
          
          Nc = 2;
        }
        
        if(coloring_distance == 2){
          sigma[0] = 0;
          sigma[1] = 1;
          sigma[2] = 2;
          sigma[3] = 3;
          
          Nc = 8;
        }
        
        if(coloring_distance == 3){
          sigma[0] = 0;
          sigma[1] = 1;
          sigma[2] = 3;
          sigma[3] = 5;
          
          Nc = 16;
        }
        
        if(coloring_distance == 4){
          sigma[0] = 0;
          sigma[1] = 1;
          sigma[2] = 6;
          sigma[3] = 9;
          
          Nc = 32;
        }
      
        if(coloring_distance == 5){ 
          sigma[0] = 0; 
          sigma[1] = 1; 
          sigma[2] = 11; 
          sigma[3] = 27; 

          Nc = 88; 
        } 

        if(coloring_distance == 6){ 
          sigma[0] = 0; 
          sigma[1] = 1; 
          sigma[2] = 8; 
          sigma[3] = 44; 

          Nc = 128; 
        } 

        if(coloring_distance == 7){ 
          sigma[0] = 0; 
          sigma[1] = 1; 
          sigma[2] = 9; 
          sigma[3] = 33; 

          Nc = 176; 
        } 

        if(coloring_distance == 8){ 
          sigma[0] = 0; 
          sigma[1] = 7; 
          sigma[2] = 48; 
          sigma[3] = 51; 

          Nc = 272; 
        }

        if(coloring_distance == 9){ 
          sigma[0] = 0; 
          sigma[1] = 1; 
          sigma[2] = 33; 
          sigma[3] = 45; 

          Nc = 352; 
        }
      }

    public:
      PLEGMA_Cprobing(int coloring_distance, int probing_dimension=4):Nc(0),coloring_distance(coloring_distance),probing_dimension(probing_dimension),sigma(4, 0){
        if(!HGC_init_PLEGMA_flag){ fprintf(stderr, "Error PLEGMA should be initialized before use this class"); exit(-1);}
        if(probing_dimension != 4 && probing_dimension != 3) PLEGMA_error("Classical probing supports only 3D and 4D coloring");
        PLEGMA_printf("Distance of neigbors is %d\n",coloring_distance);
        for(int i = 0 ; i < probing_dimension ; i++){
          if(coloring_distance >= HGC_localL[i]) PLEGMA_error("The coloring distance is larger than the lattice extent in direction %d\n",i);
        }
        h_localColors = (int*)malloc(HGC_localVolume * sizeof(int));
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