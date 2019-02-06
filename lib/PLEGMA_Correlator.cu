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
    cudaMemcpyToSymbol((void*) &DGC_moms, (void*) &HGC_moms, sizeof(tex_mom_list));
  }
  else if(corr_space == POSITION_SPACE) {
    corr_pos_space = new PLEGMA_Field<Float>(HOST, site_size, NO_GHOSTS);
    corr = corr_pos_space->H_elem();
    vol_size = corr_pos_space->Total_length();
  }
  else {
    errorQuda("corr_space not supported by correlator");
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
      errorQuda("corr_space not supported by correlator");
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
			  int source[4], const char* flavor_string){
  n_flavors = 1;
  n_groups = 1;
  shape = {gammas.size()};
  setSource(source);
  flavors = {flavor_string};
  groups =  {"Local"};
  description = getGammasString(gammas)+" / re,im";
  initialize();

  propTex<Float> bwdPropTex, fwdPropTex;
  bwdPropTex.tex = bwdProp.createTexObject();
  fwdPropTex.tex = fwdProp.createTexObject();
  if(gammas.size() == 0) errorQuda("List of gammas provided is empty");

  for(int it = 0; it < GK_localL[3]; it++)
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
  printfQuda("contractNucleonThrp: Will perform in %s precision\n", typeid(Float) == typeid(float) ? "single" :  "double");
  for(int idir = 0; idir < N_DIMS; idir++){
    gsu3.absorbDir_device(gauge,idir);
    gsu3.communicateGhost(idir+N_DIMS); // later do only the direction we are interested in
    for(int it = 0; it < GK_localL[3]; it++)
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
			 int source[4], const char* flavor_string){
  n_flavors = 1;
  n_groups = 1;
  shape = {N_DIMS, gammas.size()};
  setSource(source);
  flavors = {flavor_string};
  groups =  {"OneD"};
  description = getGammasString(gammas)+" / re,im";
  initialize();

  if(gammas.size() == 0) errorQuda("List of gammas provided is empty");
  contractNucleonThrp_derGen<Float>(*this,bwdProp,fwdProp,gauge,signProps,gammas,
				    contractPropOpProp_oneD<Float,Float,Float,Float>);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractNucleonThrp_noe(PLEGMA_Propagator<Float> &bwdProp,
			PLEGMA_Propagator<Float> &fwdProp,
			PLEGMA_Gauge<Float> &gauge,
			int signProps, int source[4], const char* flavor_string){
  n_flavors = 1;
  n_groups = 1;
  shape = {N_DIMS, gammas.size()};
  setSource(source);
  flavors = {flavor_string};
  groups =  {"Noether"};
  description = getGammasString(gammas)+" / re,im";
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
			       int source[4], const char* flavor_string){
  n_flavors = 1;
  n_groups = 1;
  shape = {gammas.size()};
  setSource(source);
  flavors = {flavor_string};
  groups =  {"wilsonLine"};
  description = getGammasString(gammas)+" / re,im";
  initialize();
  
  propTex<Float> bwdPropTex, fwdPropTex;
  su3Tex<Float> sTex;
  bwdPropTex.tex = bwdProp.createTexObject();
  fwdPropTex.tex = fwdProp.createTexObject();
  sTex.tex = su3.createTexObject();
  if(gammas.size() == 0) errorQuda("List of gammas provided is empty");

  for(int it = 0; it < GK_localL[3]; it++)
    contractPropOpProp_wilsonLine(*this,bwdPropTex,fwdPropTex,signProps,sTex,it,gammas);

  bwdProp.destroyTexObject(bwdPropTex.tex);
  fwdProp.destroyTexObject(fwdPropTex.tex);
  su3.destroyTexObject(sTex.tex);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
writeFile(const char*filename, FILE_WRITE_FORMAT format) {
  if(format == ASCII_FORM) {
    printfQuda("Going to write file %s in ASCII format\n",filename);
    writeASCII(filename);
  }
  else if(format == HDF5_FORM) {
    printfQuda("Going to write file %s in HDF5 format\n",filename);
    writeHDF5(filename, params);
  }
  else {
    errorQuda("FILE_WRITE_FORMAT not supported: %d\n", params.CorrFileFormat);
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
    errorQuda("Corralator: corrSpace not supported: %d\n", corr_space);
  }

  Float *corrGlobal;
  if(rank == 0){
    corrGlobal = (Float*) malloc(g_vol_size*site_size*2*sizeof(Float));
    if( corrGlobal == NULL ) errorQuda("writeASCII: Cannot allocate memory.");
  }

  MPI_Gather(corr,site_size*vol_size*2,MPI_Type(corr),
	     corrGlobal,site_size*vol_size*2,MPI_Type(corr),
	     0,comm);

  FILE *ptr_out = NULL;
  if(rank == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    if(n_flavors != 1) errorQuda("For now works with n_flavors = 1\n");
    for(size_t v=0; v<g_vol_size*site_size; v++) {
      int it = v/(GK_Nmoms*site_size);
      int imom = v/(site_size) - it*GK_Nmoms;
      int is = v%site_size;
      int it_shift = (it + source_position[3])%GK_totalL[3];
      int ipos = it_shift*GK_Nmoms*site_size + imom*site_size + is;
      if(corr_space == MOMENTUM_SPACE) {
	if(is == 0)fprintf(ptr_out, "%d  %+d  %+d  %+d ", v/(HGC_Nmoms*site_size), HGC_moms[imom][0], HGC_moms[imom][1], HGC_moms[imom][2]);
      }
      else if (corr_space == POSITION_SPACE) {
	//TODO
      }
      fprintf(ptr_out, "%+e %+eI ", corrGlobal[ipos*2], corrGlobal[ipos*2+1]);
      if(is == site_size-1)fprintf(ptr_out, "\n");
    }
    fclose(ptr_out);
    free(corrGlobal);
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
    errorQuda("Corralator: corr_space not supported: %d\n", corr_space);
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
    errorQuda("Corralator: corr_space not supported: %d\n", corr_space);
  }
  std::for_each(shape.begin(), shape.end(), [&] (int n) {
					      start[i] = 0;
					      ldims[i] = dims[i] = n;
					      i++;}); //shape
  
  start[i] = 0; ldims[i] = dims[i] = 2; //re-im
}

/* Attribute writing */
static void write_text_attribute(hid_t group_id,const char* attr_name, const char* attr_value) {
  hid_t attrdat_id = H5Screate(H5S_SCALAR);
  hid_t type_id = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id, strlen(attr_value));
  hid_t attr_id = H5Acreate2(group_id, attr_name, type_id, 
			     attrdat_id, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, type_id, attr_value);
  H5Aclose(attr_id);
  H5Tclose(type_id);
  H5Sclose(attrdat_id);
}

static inline size_t get_volume(int ndims, hsize_t* dims){
  size_t volume = 1;
  for(int i=0; i<ndims; i++){
    volume*=dims[i];
  }
  return volume;
}

static inline size_t get_id(int ndims,hsize_t* ids,hsize_t* dims){
  size_t id = ids[0];
  for(int i=1; i<ndims; i++){
    id = id*dims[i] + ids[i];
  }
  return id;
}

static inline void get_ids(int ndims, size_t id, hsize_t* ids, hsize_t* dims){
  for(int i=ndims-1; i>=0; i--){
    ids[i] = id % dims[i];
    id/=dims[i];
  }
}

static inline void add_ids(int ndims, hsize_t* ids, hsize_t* ids2){
  for(int i=ndims-1; i>=0; i--){
    ids[i] += ids2[i];
  }
}


// This writing function supports shifted chunk, i.e. when start[i]+ldims[i] > dims[i].
// The part exceeding is written starting from 0 as in case of periodic boundaries. 
template<typename Float>
static void write_dataset(hid_t group_id, const char* name, Float *buf, int ndims, hsize_t* dims, hsize_t* ldims, hsize_t* start) {
  hid_t DATATYPE_H5;
  if( typeid(Float) == typeid(float) ){
    DATATYPE_H5 = H5T_NATIVE_FLOAT;
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE_H5 = H5T_NATIVE_DOUBLE;
  }

  // checking for exceeding dims
  int exceeding = 0;
  int exceeding_dim[ndims];
  hsize_t exceeding_ammount[ndims];
  for(int i=0; i<ndims; i++) {
    if(start[i]+ldims[i] > dims[i]) {
      exceeding_dim[exceeding] = i;
      exceeding_ammount[i] = start[i]+ldims[i]-dims[i];
      //printf("rank %d: Dim %d exceeds of %d -> shifting\n",comm_rank(),i,exceeding_ammount[exceeding]);
      if(exceeding_dim[exceeding]>ldims[i]) {
	errorQuda("Exceeding is too high. The method may have problems.");
      }
      exceeding++;
    }
    else {
      exceeding_ammount[i] = 0;
    }
  }
  
  hid_t filespace  = H5Screate_simple(ndims, dims,  NULL);
  hid_t dataset_id = H5Dcreate(group_id, name, DATATYPE_H5, filespace,
			       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // We need 2^exceeding loops for writing everything 
  for(int i=0; i<(1<<exceeding); i++) {
    filespace = H5Dget_space(dataset_id);
    hsize_t tmp_ldims[ndims];
    hsize_t tmp_start[ndims];
    hsize_t tmp_shift[ndims];
    for(int j=0; j<ndims; j++) {
      tmp_ldims[j] = ldims[j] - exceeding_ammount[j];
      tmp_start[j] = start[j];
      tmp_shift[j] = 0;
    }
    // checking if we shift the exceeding direction
    int tmp=i, j=0;
    while(tmp>0) {
      if(tmp%2) {
	int dim = exceeding_dim[j];
	tmp_shift[dim] = tmp_ldims[dim];
	tmp_ldims[dim] = exceeding_ammount[dim];
	tmp_start[dim] = 0;
      }
      tmp/=2;
      j++;
    }
    //printf("rank %d: iter %d:\n",comm_rank(),i);
    //for(int j=0; j<ndims; j++) {
    //  printf("rank %d: i=%d, dims[i]=%d tmp_ldims[i]=%d, tmp_start[i]=%d, tmp_shift[i]=%d,\n",comm_rank(),j,dims[j], tmp_ldims[j], tmp_start[j], tmp_shift[j]);
    //}
    // copying the part of the buffer
    size_t vol = get_volume(ndims, tmp_ldims);
    Float tmp_buf[get_volume(ndims, tmp_ldims)];
    for(size_t j=0; j<vol; j++) {
      hsize_t ids[ndims];
      get_ids(ndims, j, ids, tmp_ldims);
      add_ids(ndims, ids, tmp_shift);
      tmp_buf[j] = buf[get_id(ndims,ids,ldims)];
    }
    
    hid_t subspace   = H5Screate_simple(ndims, tmp_ldims, NULL);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, tmp_start, NULL, tmp_ldims, NULL);
    
    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    // TODO: we need to fix the parallel writing in case of more than one processor has an exceeding direction.
    herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, 
			     i==0 ? plist_id : H5P_DEFAULT, tmp_buf);
    if(status<0) errorQuda("write_dataset: Unsuccessful writing of the dataset. Exiting\n");
    H5Sclose(subspace);
    H5Pclose(plist_id);    
    H5Sclose(filespace);
  }

  H5Dclose(dataset_id);
}


template<typename Float>
void PLEGMA_Correlator<Float>::
writeHDF5(const char*filename, const char* top) {
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
    errorQuda("Corralator: corrSpace not supported: %d\n", corr_space);
  }

  int ndims = getNDims();
  hsize_t dims[ndims], ldims[ndims], start[ndims];
  fillDims(dims, ldims, start, shift_source);

  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);

  char *group1_tag;
  hid_t group1_id = H5Gcreate(file_id, top, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  char *group2_tag;
  asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	   source_position[0],
	   source_position[1],
	   source_position[2],
	   source_position[3]);
  hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  //- Source position
  char *src_pos;
  asprintf(&src_pos," [x, y, z, t] = [%02d, %02d, %02d, %02d]\0",
	   source_position[0],
	   source_position[1],
	   source_position[2],
	   source_position[3]);
  write_text_attribute(group2_id, "source-position", src_pos);
  free(src_pos);

  //- Index identification-ordering, precision
  //char *corr_info;
  //asprintf(&corr_info,"Position-space meson 2pt-correlator\nIndex Order: [flav, t, z, y, x, real/imag]\nPrecision: %s\0",(typeid(Float) == typeid(float)) ? "single" : "double");
  //write_text_attribute(file_id, "Correlator-info", src_pos);
  //free(corr_info);

  size_t spaceSize = get_volume(ndims, ldims);

  for(int g=0; g<n_groups; g++){
    hid_t group3_id = H5Gcreate(group2_id, corr_groups_names[corr_type][g], H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);
    for(int d=0; d<n_flavors; d++){
      Float *writeBuf = corr + (g*n_flavors+d)*spaceSize;
      write_dataset(group3_id, corr_flavors_names[corr_type][d], writeBuf,
		    ndims, dims, ldims, start);
    }
    H5Gclose(group3_id);
  }

  H5Gclose(group2_id);
  H5Gclose(group1_id);
  H5Fclose(file_id);
}

template class PLEGMA_Correlator<float>;
template class PLEGMA_Correlator<double>;
