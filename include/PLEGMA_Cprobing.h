#pragma once

namespace plegma {
  // inline int getVecToInd(std::vector<int> x, std::vector<int> L){
  //   if(x.size() != L.size()) PLEGMA_error("Dimensions do not match");
  //   if(x.size() == 0)PLEGMA_error("Size of the vector is zero");
  //   int D=x.size();
  //   for(int i = 0 ; i < D; i++)
  //     if(x[i] >= L[i])
	//       PLEGMA_error("Error the position of the vector exceeds the extent of dimension %d", i);
  //   int acc=x[D-1];
  //   for(int i = D-2 ; i >= 0; i--) acc = acc*L[i] + x[i];
  //   return acc;
  // }

  // inline std::vector<int> getIndToVec(int ind, std::vector<int> L){
  //   int V=1;
  //   std::vector<int> x;
  //   if(L.size() == 0)PLEGMA_error("Size of the vector is zero");
  //   if(ind < 0 )PLEGMA_error("Ind provided is negative");
  //   int D = L.size();
  //   for(int i = 0 ; i < D-1; i++ ) V *= L[i];
  //   if(V<0) PLEGMA_error("The volume is negative which is not allowed");
  //   if(V==0) PLEGMA_error("One or more directions are zero");
  //   if(ind >= V*L[D-1]) PLEGMA_error("The ind exceeds the total volume");
  //   int sub=0;
  //   for(int i = D-1; i >= 0; i--){
  //     ind -= sub;
  //     x.insert(x.begin(), ind/V);
  //     if(V==1) break;
  //     sub=x[0]*V;
  //     V /= L[i-1];
  //   }
  //   return x;
  // }

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
      // short Lu; // extent of the elementaty coloring block (assume symmetric block)
      // int* h_arrVc; // array to hold the coloring of the lattice on HOST
      // int* d_arrVc; // array to hold the coloring of the lattice on Device
      // int* arrlc; // array to hold the elementary coloring block
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
          
          // PLEGMA_printf("Local coord (%d,%d,%d,%d) ",x_local[0],x_local[1],x_local[2],x_local[3]);
          // PLEGMA_printf("Rank coord (%d,%d,%d,%d) ",rank_coord[0],rank_coord[1],rank_coord[2],rank_coord[3]);
          // PLEGMA_printf("Local L (%d,%d,%d,%d) ",lL[0],lL[1],lL[2],lL[3]);
          // PLEGMA_printf("Global coord (%d,%d,%d,%d)\n",x_global[0],x_global[1],x_global[2],x_global[3]);

          int col = sigma[0]*x_global[3] + sigma[1]*x_global[2] + sigma[2]*x_global[1] + sigma[3]*x_global[0]; // ordering xyzt
          col = (col % Nc) + 1;
          h_localColors[i] = col;
        }
      }
      //  void checkColoring();

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
        //  if(check)checkColoring();
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


/* inline static int boundaryCheck(int x, int L){ */
/*   int y=x; */
/*   if (y >= L) */
/*     y=y%L; */
/*   if (y < 0) */
/*     y=y+L; */
/*   return y; */
/* } */

/* void Hprobing::checkColoring(){ */
/*   if(HGC_nProc[0]*HGC_nProc[1]*HGC_nProc[2]*HGC_nProc[3] != 1) PLEGMA_error("The coloring check works only with 1 MPI task"); */
/*   std::vector<int> lL = {HGC_localL[0], HGC_localL[1], HGC_localL[2], HGC_localL[3]}; */
/*   for(int t = 0 ; t < HGC_totalL[3] ; t++) */
/*     for(int z = 0 ; z < HGC_totalL[2] ; z++) */
/*       for(int y = 0 ; y < HGC_totalL[1] ; y++) */
/* 	for(int x = 0 ; x < HGC_totalL[0] ; x++){ */
/* 	  std::vector<int> xx = {x,y,z,t}; */
/* 	  int c1 = h_arrVc[getVecToInd(xx, lL)]; */
/* 	  for(int dx = -D+1 ; dx < D ; dx++) */
/* 	    for(int dy = -D+1 ; dy < D ; dy++) */
/* 	      for(int dz = -D+1 ; dz < D ; dz++) */
/* 		for(int dt = -D+1 ; dt < D ; dt++){ */
/* 		  int ds = abs(dx) + abs(dy) + abs(dz) + abs(dt); */
/* 		  if ((ds<D) && (ds != 0)){ */
/* 		    int xn = x + dx; */
/* 		    xn = boundaryCheck(xn,HGC_totalL[0]); */
/* 		    int yn = y + dy; */
/* 		    yn = boundaryCheck(yn,HGC_totalL[1]); */
/* 		    int zn = z + dz; */
/* 		    zn = boundaryCheck(zn,HGC_totalL[2]); */
/* 		    int tn = t + dt; */
/* 		    tn = boundaryCheck(tn,HGC_totalL[3]); */
/* 		    xx[0] = xn; xx[1] = yn; xx[2] = zn; xx[3] = tn; */
/* 		    int c2 = h_arrVc[getVecToInd(xx, lL)]; */
/* 		    if(c1 == c2){ */
/* 		      PLEGMA_printf("Colors (%d,%d)\n",c1,c2); */
/* 		      PLEGMA_error("Mistake found in the coloring with (%d,%d,%d,%d) and (%d,%d,%d,%d)",x,y,z,t,xn,yn,zn,tn); */
/* 		    } */
/* 		  } */
/* 		} */
/* 	} */
/*   PLEGMA_printf("Check in coloring passed successfully\n"); */
/* } */
