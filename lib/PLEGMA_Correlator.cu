#include <PLEGMA_Correlator.h>
#include <string>
#include <PLEGMA_mesons.cuh>
#include <PLEGMA_baryons.cuh>
#include <functional>
using namespace plegma;

//--------------------------------//
// class PLEGMA_Correlator //
//--------------------------------//

template<typename Float>
void PLEGMA_Correlator<Float>::
initialize() {
  if(isAlloc && site_size == getSiteSize())
    return;
  finalize();
  site_size = getSiteSize();
  if(corr_space == MOMENTUM_SPACE) {
    corr_mom_space = new PLEGMA_FT<Float>(Q2_max);
    corr_mom_space->checkAllocation(site_size);
    corr = corr_mom_space->H_elem();
    vol_size = corr_mom_space->Nmoms()*corr_mom_space->DimT();
    if(HGC_moms.Nmoms>0) {
      HGC_moms.free();
    }
    HGC_moms = corr_mom_space->getTexMomList();
    cudaMemcpyToSymbol(DGC_moms, (void*) &HGC_moms, sizeof(tex_mom_list));
    checkCudaError();
  }
  else if(corr_space == POSITION_SPACE) {
    corr_pos_space = new PLEGMA_Field<Float>(HOST, site_size, NO_GHOSTS);
    corr = corr_pos_space->H_elem();
    vol_size = corr_pos_space->Total_length();
  }
  else {
    PLEGMA_error("corr_space not supported by correlator");
  }
  isAlloc = true;
}

template<typename Float>
void PLEGMA_Correlator<Float>::
finalize() {
  if (isAlloc) {
    if(corr_space == POSITION_SPACE)
      delete corr_pos_space;
    else if(corr_space == MOMENTUM_SPACE)	 
      delete corr_mom_space;
    else
      PLEGMA_error("corr_space not supported by correlator");
  }
  isAlloc = false;
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractMesons(PLEGMA_Propagator<Float> &prop1,
	       PLEGMA_Propagator<Float> &prop2, 
	       int source[4]){

  setSource(source);
  n_flavors = 2;
  n_groups = N_MESONS;
  shape = {};
  flavors =  {"twop_meson_1", "twop_meson_2"};
  groups =  {"pseudoscalar", "scalar", "g5g1", "g5g2", "g5g3", "g5g4", "g1", "g2", "g3", "g4"};
  description = "";
  
  initialize();
  propTex<Float> prop1Tex, prop2Tex;
  prop1Tex.tex = prop1.createTexObject();
  prop2Tex.tex = prop2.createTexObject();

  for(int it = 0 ; it < HGC_localL[3] ; it++) {
    contract_mesons(prop1Tex,prop2Tex,*this,it);
  }

  prop1.destroyTexObject(prop1Tex.tex);
  prop2.destroyTexObject(prop2Tex.tex);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractBaryons(PLEGMA_Propagator<Float> &prop1,
		PLEGMA_Propagator<Float> &prop2, 
		int source[4]){

  setSource(source);
  n_flavors = 2;
  n_groups = N_BARYONS;
  shape = {16};
  flavors = {"twop_baryon_1", "twop_baryon_2"};
  groups =  {"nucl_nucl",
#ifdef ALL_BARYONS
	     "nucl_nucl2","nucl2_nucl","nucl2_nucl2","deltap_deltaz_11","deltap_deltaz_22","deltap_deltaz_33",
	     "deltapp_deltamm_11","deltapp_deltamm_22","deltapp_deltamm_33"
#endif
  };
  description = "1,g1,g2,g3,g4,g5,g5g1,g5g2,g5g3,g5g4,s12,s13,s23,s41,s42,s43";

  initialize();
  propTex<Float> prop1Tex, prop2Tex;
  prop1Tex.tex = prop1.createTexObject();
  prop2Tex.tex = prop2.createTexObject();

  for(int it = 0; it < HGC_localL[3]; it++) {
    contract_baryons(prop1Tex,prop2Tex,*this,it);
  }
  prop1.destroyTexObject(prop1Tex.tex);
  prop2.destroyTexObject(prop2Tex.tex);
}

template<typename FloatC,typename FloatA, typename FloatB>
void contractPropOpProp_local(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2,
			      int signProps, int it, std::vector<GAMMAS> gammas);
template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_local(PLEGMA_Propagator<Float> &bwdProp,
			  PLEGMA_Propagator<Float> &fwdProp,
			  int signProps, std::vector<GAMMAS> gammas,
			  int source[4]){
  n_flavors = 1;
  n_groups = 1;
  shape = {(int) gammas.size()};
  setSource(source);
  flavors = {"threep"};
  groups =  {"Local"};
  description = getGammasString(gammas)+" / re,im";
  initialize();

  propTex<Float> bwdPropTex, fwdPropTex;
  bwdPropTex.tex = bwdProp.createTexObject();
  fwdPropTex.tex = fwdProp.createTexObject();
  if(gammas.size() == 0) PLEGMA_error("List of gammas provided is empty");

  for(int it = 0; it < HGC_localL[3]; it++)
    contractPropOpProp_local(*this,bwdPropTex,fwdPropTex,signProps,it,gammas);
  bwdProp.destroyTexObject(bwdPropTex.tex);
  fwdProp.destroyTexObject(fwdPropTex.tex);
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS>
void contractPropOpProp_oneD(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2,
			     int signProps, su3Tex<FloatS> su3, int it, int dir,std::vector<GAMMAS> gammas);
template<typename FloatC,typename FloatA, typename FloatB, typename FloatS>
void contractPropOpProp_noe(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1, propTex<FloatB> prop2,
			    int signProps, su3Tex<FloatS> su3, int it, int dir,std::vector<GAMMAS> gammas);

template<typename Float>
static void contractNucleonThrp_derGen(PLEGMA_Correlator<Float> &corr, PLEGMA_Propagator<Float> &bwdProp,
				       PLEGMA_Propagator<Float> &fwdProp, PLEGMA_Gauge<Float> &gauge,
				       int signProps, std::vector<GAMMAS> gammas,
				       std::function<void(PLEGMA_Correlator<Float>&,propTex<Float>,
							  propTex<Float>,int,su3Tex<Float>,int,int,
							  std::vector<GAMMAS>)> funcContract){
  // gauge should have the sign for the antiperiodic boundary conditions

  PLEGMA_Su3field<Float> gsu3(DEVICE);
  propTex<Float> bwdPropTex, fwdPropTex;
  su3Tex<Float> gsu3Tex;
  gsu3Tex.tex = gsu3.createTexObject();
  bwdProp.communicateGhost();
  fwdProp.communicateGhost();
  bwdPropTex.tex = bwdProp.createTexObject();
  fwdPropTex.tex = fwdProp.createTexObject();
  PLEGMA_printf("contractNucleonThrp: Will perform in %s precision\n", typeid(Float) == typeid(float) ? "single" :  "double");
  for(int idir = 0; idir < N_DIMS; idir++){
    gsu3.absorbDir_device(gauge,idir);
    gsu3.communicateGhost(idir+N_DIMS);
    for(int it = 0; it < HGC_localL[3]; it++)
      funcContract(corr,bwdPropTex,fwdPropTex,signProps,gsu3Tex,it, idir,gammas);
  }
  bwdProp.destroyTexObject(bwdPropTex.tex);
  fwdProp.destroyTexObject(fwdPropTex.tex);
  gsu3.destroyTexObject(gsu3Tex.tex);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_oneD(PLEGMA_Propagator<Float> &bwdProp,
			 PLEGMA_Propagator<Float> &fwdProp,
			 PLEGMA_Gauge<Float> &gauge,
			 int signProps, std::vector<GAMMAS> gammas,
			 int source[4]){
  n_flavors = 1;
  n_groups = 1;
  shape = {N_DIMS, (int) gammas.size()};
  setSource(source);
  flavors = {"threep"};
  groups =  {"OneD"};
  description = "x,y,z,t / "+getGammasString(gammas)+" / re,im";
  initialize();

  if(gammas.size() == 0) PLEGMA_error("List of gammas provided is empty");
  contractNucleonThrp_derGen<Float>(*this,bwdProp,fwdProp,gauge,signProps,gammas,
				    contractPropOpProp_oneD<Float,Float,Float,Float>);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_noe(PLEGMA_Propagator<Float> &bwdProp,
			PLEGMA_Propagator<Float> &fwdProp,
			PLEGMA_Gauge<Float> &gauge,
			int signProps, int source[4]){
  n_flavors = 1;
  n_groups = 1;
  shape = {N_DIMS};
  setSource(source);
  flavors = {"threep"};
  groups =  {"Noether"};
  description = "x,y,z,t / re,im";
  initialize();

  std::vector<GAMMAS> gammas = {};
  contractNucleonThrp_derGen<Float>(*this,bwdProp,fwdProp,gauge,signProps,gammas,
				    contractPropOpProp_noe<Float,Float,Float,Float>);
}

template<typename FloatC,typename FloatA, typename FloatB, typename FloatS>
void contractPropOpProp_wilsonLine(PLEGMA_Correlator<FloatC> &corr, propTex<FloatA> prop1,
				   propTex<FloatB> prop2, int signProps,
				   su3Tex<FloatS> su3, int it, std::vector<GAMMAS> gammas);
template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_wilsonLine(PLEGMA_Propagator<Float> &bwdProp,
			       PLEGMA_Propagator<Float> &fwdProp,
			       PLEGMA_Su3field<Float> &su3,
			       int signProps, std::vector<GAMMAS> gammas,
			       int source[4]){
  n_flavors = 1;
  n_groups = 1;
  shape = {(int) gammas.size()};
  setSource(source);
  flavors = {"threep"};
  groups =  {"wilsonLine"};
  description = getGammasString(gammas)+" / re,im";
  initialize();
  
  propTex<Float> bwdPropTex, fwdPropTex;
  su3Tex<Float> sTex;
  bwdPropTex.tex = bwdProp.createTexObject();
  fwdPropTex.tex = fwdProp.createTexObject();
  sTex.tex = su3.createTexObject();
  if(gammas.size() == 0) PLEGMA_error("List of gammas provided is empty");

  for(int it = 0; it < HGC_localL[3]; it++)
    contractPropOpProp_wilsonLine(*this,bwdPropTex,fwdPropTex,signProps,sTex,it,gammas);

  bwdProp.destroyTexObject(bwdPropTex.tex);
  fwdProp.destroyTexObject(fwdPropTex.tex);
  su3.destroyTexObject(sTex.tex);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
writeFile(const char*filename, FILE_WRITE_FORMAT format) {
  if(format == ASCII_FORM) {
    if(HGC_verbosity > 1) PLEGMA_printf("Going to write file %s in ASCII format\n",filename);
    writeASCII(filename);
  }
  else if(format == HDF5_FORM) {
    if(HGC_verbosity > 1) PLEGMA_printf("Going to write file %s in HDF5 format\n",filename);
    writeHDF5(filename);
  }
  else {
    PLEGMA_error("FILE_WRITE_FORMAT not supported: %d\n", format);
  }
}

/*
template<typename Float>
void PLEGMA_Correlator<Float>::
writeFile(PLEGMA_params &params) {
  const char*filename, *Qsq, *name2;
  std::string ext="", name="";
  if(params.corr_space==MOMENTUM_SPACE) asprintf(&Qsq,"Qsq%d_",params.Q_sq);
  else asprintf(&Qsq,"");
  if(params.CorrFileFormat == ASCII_FORM) ext = ".dat";
  else if(params.CorrFileFormat == HDF5_FORM) ext = ".h5";
  switch(corr_type) {
  case MESONS:
    name = "twop.%04d_mesons";
    break;
  case BARYONS:
    name = "twop.%04d_baryons";
    break;
  case THRP_LOCAL:
    name = "thrp.%04d_local";
    break;
  case THRP_NOETHER:
    name = "thrp.%04d_noether";
    break;
  case THRP_ONED:
    name = "thrp.%04d_oneD";
    break;
  default:
    name = "unknown.%04d";
  }
  asprintf(&name2, name.c_str(), params.traj);
  asprintf(&filename,"%s/%s_%sSS.%02d.%02d.%02d.%02d%s" ,
	   params.corr_dir, name2, Qsq,
	   params.sourcePosition[isource][0],
	   params.sourcePosition[isource][1],
	   params.sourcePosition[isource][2],
	   params.sourcePosition[isource][3], ext.c_str());

  writeFile(filename, params);
  free(name2);
  free(Qsq);
  free(filename);
}
*/

template<typename Float>
void PLEGMA_Correlator<Float>::
writeASCII(const char *filename_out) {
  MPI_Comm comm;
  size_t g_vol_size;
  int rank;
  
  if(corr_space == MOMENTUM_SPACE && (HGC_timeRank > HGC_nProc[3] || HGC_timeRank <0 ))
    return;

  switch(corr_space) {
  case MOMENTUM_SPACE:
    g_vol_size = vol_size*HGC_nProc[3];
    comm = HGC_timeComm;
    rank = HGC_timeRank;
    break;
  case POSITION_SPACE:
    g_vol_size = vol_size*HGC_nProc[0]*HGC_nProc[1]*HGC_nProc[2]*HGC_nProc[3];
    comm = MPI_COMM_WORLD;
    rank = comm_rank();
    break;
  default:
    PLEGMA_error("Corralator: corrSpace not supported: %d\n", corr_space);
  }

  Float *corrGlobal;
  if(rank == 0) hostMalloc(corrGlobal, g_vol_size*site_size*2*sizeof(Float));

  // TODO: this works fine for timeComm (MOMENTUM_SPACE) but not for MPI_COMM_WORLD (POSITION SPACE)
  // in the second case requires reordering of the memory
  MPI_Gather(corr,site_size*vol_size*2,MPI_Type(corr),
	     corrGlobal,site_size*vol_size*2,MPI_Type(corr),
	     0,comm);

  FILE *ptr_out = NULL;
  if(rank == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) PLEGMA_error("Error opening file for writing\n");

    if(corr_space == MOMENTUM_SPACE) {
      int Nmoms = corr_mom_space->Nmoms();
      std::vector<std::vector<int>> momV = corr_mom_space->MomList();
      for(int it=0; it<HGC_totalL[3]; it++) {
	int it_shift = (it + source_position[3])%HGC_totalL[3];
	for(int imom=0; imom<Nmoms; imom++) {
	  int ipos = (it_shift*Nmoms + imom)*site_size;
	  fprintf(ptr_out, "%d  %+d  %+d  %+d ", it, momV[imom][0], momV[imom][1], momV[imom][2]);
	  for(int is = 0; is<site_size; is++)
	    fprintf(ptr_out, "%+e %+eI ", corrGlobal[ipos*2+is*2], corrGlobal[ipos*2+is*2+1]);
	  fprintf(ptr_out, "\n");
	}
      }
    }
    else if (corr_space == POSITION_SPACE) {
      //TODO
      PLEGMA_error("WriteASCII do not support writing in position space.\n");
    }
    fclose(ptr_out);
    hostFree(corrGlobal, g_vol_size*site_size*2*sizeof(Float));
  }  
}

template<typename Float>
int PLEGMA_Correlator<Float>::
getNDims() {
  int ndims = shape.size()+1; // shape + re-im
  switch(corr_space) {
  case MOMENTUM_SPACE:
    ndims += 2; // mom, t
    break;
  case POSITION_SPACE:
    ndims += 4; // t, z, y, x
    break;
  default:
    PLEGMA_error("Corralator: corr_space not supported: %d\n", corr_space);
  }
 
  return ndims;
}

template<typename Float>
void PLEGMA_Correlator<Float>::
fillDims(hsize_t* dims, hsize_t* ldims, hsize_t* start, bool shift_source) {
  int i=0;
  switch(corr_space) {
  case MOMENTUM_SPACE:
    start[0] = HGC_timeRank*HGC_localL[3]; //starting point
    if(shift_source) {
      start[0] = (start[0] + HGC_totalL[3] - source_position[3]) % HGC_totalL[3];
    }
    ldims[0] = HGC_localL[3]; //LT
    dims[0] = HGC_totalL[3]; //T
    start[1] = 0; ldims[1] = dims[1] = vol_size/HGC_totalL[3]; //Nmoms
    i=2;
    break;
  case POSITION_SPACE:
    for(i=0; i<N_DIMS; i++) {
      start[i] = comm_coords(HGC_default_topo)[i]*HGC_localL[i]; //starting
      if(shift_source) {
	start[i] = (start[i] + HGC_totalL[i] - source_position[i]) % HGC_totalL[i];
      }
      ldims[i] = HGC_localL[i]; //LT
      dims[i] = HGC_totalL[i]; //T
    }
    break;
  default:
    PLEGMA_error("Corralator: corr_space not supported: %d\n", corr_space);
  }
  std::for_each(shape.begin(), shape.end(), [&] (int n) {
					      start[i] = 0;
					      ldims[i] = dims[i] = n;
					      i++;}); //shape
  
  start[i] = 0; ldims[i] = dims[i] = 2; //re-im
}


template<typename Float>
void PLEGMA_Correlator<Float>::
writeHDF5(std::string filename, std::string top) {
  // only one per time writes in momentum space
  if(corr_space == MOMENTUM_SPACE && (HGC_timeRank > HGC_nProc[3] || HGC_timeRank <0 ))
    return;

  MPI_Comm comm;
  bool shift_source = false;
  switch(corr_space) {
  case MOMENTUM_SPACE:
    comm = HGC_timeComm;
    shift_source = true;
    break;
  case POSITION_SPACE:
    comm = MPI_COMM_WORLD;
    // shift source not yet supported in position space. the parallel writing needs to be fixed.
    // TODO: command_line_flag
    shift_source = false;
    break;
  default:
    PLEGMA_error("Corralator: corrSpace not supported: %d\n", corr_space);
  }

  int ndims = getNDims();
  hsize_t dims[ndims], ldims[ndims], start[ndims];
  fillDims(dims, ldims, start, shift_source);

  hid_t file = H5_open_file(filename, comm);

  char *source;
  asprintf(&source,"/sx%02dsy%02dsz%02dst%02d", source_position[0], source_position[1], source_position[2], source_position[3]);

  hid_t top_group = H5_open_group(file, top+source);
  free(source);
  
  size_t spaceSize = get_volume(ndims, ldims);
  for(int g=0; g<n_groups; g++){
    hid_t group = H5_open_group(top_group, groups[g]);    
    for(int d=0; d<n_flavors; d++) {
      Float *writeBuf = corr + (g*n_flavors+d)*spaceSize;
      H5_write_dataset(group, flavors[d], writeBuf,
		    ndims, dims, ldims, start); 
    }
    H5Gclose(group);
  }

  H5Gclose(top_group);
  H5Fclose(file);
}

template class PLEGMA_Correlator<float>;
template class PLEGMA_Correlator<double>;
