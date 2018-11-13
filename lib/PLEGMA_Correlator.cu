#include <PLEGMA_Correlator.h>
#include <hdf5.h>
#include <PLEGMA_mesons.cuh>
 
using namespace plegma;

//--------------------------------//
// class PLEGMA_Correlator //
//--------------------------------//

template<typename Float>
void PLEGMA_Correlator<Float>::
initialize(CORR_TYPE CorrType, CORR_SPACE CorrSpace) {
  if(isAlloc && corr_type==CorrType && corr_space==CorrSpace)
    return;

  finalize();
  corr_type=CorrType;
  corr_space=CorrSpace;

  switch(CorrSpace) {
  case MOMENTUM_SPACE:
    vol_size = GK_localL[3]*GK_Nmoms;
    break;
  case POSITION_SPACE:
    vol_size = GK_localVolume;
    break;
  default:
    errorQuda("Corralator: CorrSpace not supported: %d\n", CorrSpace);
  }

  n_groups = nGroups[CorrType];
  n_flavors = nFlavors[CorrType];
  n_comp = nComp[CorrType];
  site_size = n_groups*n_flavors*n_comp;

  bytes_total_length = vol_size*site_size*2*sizeof(Float);
  corr = (Float*) malloc(bytes_total_length);
  if(corr == NULL)
     errorQuda("Correlator: Cannot allocate memory of size %d.", bytes_total_length);
  else
    isAlloc = true;
}

template<typename Float>
void PLEGMA_Correlator<Float>::
contractMesons(PLEGMA_Propagator<Float> &prop1,
	       PLEGMA_Propagator<Float> &prop2, 
	       int isource, CORR_SPACE corrSpace){

  initialize(MESONS,corrSpace);
  this->isource = isource;
  
  propTex<Float> prop1Tex, prop2Tex;
  prop1Tex.tex = prop1.createTexObject();
  prop2Tex.tex = prop2.createTexObject();

  printfQuda("contractMesons: Will perform in %s precision\n", typeid(Float) == typeid(float) ? "single" :  "double");

  for(int it = 0 ; it < GK_localL[3] ; it++) {
    contract_mesons(prop1Tex,prop2Tex,*this,it);
  }

  prop1.destroyTexObject(prop1Tex.tex);
  prop2.destroyTexObject(prop2Tex.tex);
}

template<typename Float>
void PLEGMA_Correlator<Float>::
writeFile(char *filename, PLEGMA_params *params, FILE_WRITE_FORMAT CorrFileFormat) {
  char* filename_out;
  if(CorrFileFormat == ASCII_FORM) {
    asprintf(&filename_out,"%s.dat",filename);
    printfQuda("Going to write file %s in ASCII format\n",filename_out);
    writeASCII(filename_out);
  }
  else if(CorrFileFormat == HDF5_FORM) {
    asprintf(&filename_out,"%s.h5",filename);
    printfQuda("Going to write file %s in HDF5 format\n",filename_out);
    writeHDF5(filename_out, params);
  }
  else {
    errorQuda("FILE_WRITE_FORMAT not supported: %d\n", CorrFileFormat);
  }
  free(filename_out);
}


template<typename Float>
void PLEGMA_Correlator<Float>::
writeASCII(char *filename_out) {
  MPI_Comm comm;
  size_t g_vol_size;
  int rank;
  
  if(corr_space == MOMENTUM_SPACE && (GK_timeRank > GK_nProc[3] || GK_timeRank <0 ))
    return;

  switch(corr_space) {
  case MOMENTUM_SPACE:
    g_vol_size = vol_size*GK_nProc[3];
    comm = GK_timeComm;
    rank = GK_timeRank;
    break;
  case POSITION_SPACE:
    g_vol_size = vol_size*GK_nProc[0]*GK_nProc[1]*GK_nProc[2]*GK_nProc[3];
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
    for(size_t v=0; v<g_vol_size; v++) {
      size_t shift=v*site_size;
      if(corr_space == MOMENTUM_SPACE) {
	fprintf(ptr_out, "%30d  %+20d  %+20d  %+20d ", v/GK_totalL[3], GK_moms[v%GK_totalL[3]][0],
		GK_moms[v%GK_totalL[3]][1], GK_moms[v%GK_totalL[3]][2]);
	shift=((v/GK_totalL[3]+GK_sourcePosition[isource][3])%GK_totalL[3])*GK_Nmoms + v%GK_totalL[3];
      }
      else if (corr_space == POSITION_SPACE) {
	//TODO
      }
      for(size_t s=0; s<site_size; s+=2) {
	fprintf(ptr_out, "%+e %+eI ", corrGlobal[shift*2+s], corrGlobal[shift*2+s+1]);
      }
      fprintf(ptr_out, "\n");
    }
    fclose(ptr_out);
    free(corrGlobal);
  }  
}

static int getNDims(CORR_TYPE CorrType, CORR_SPACE CorrSpace) {
  int ndims = 1; //re-im
  switch(CorrSpace) {
  case MOMENTUM_SPACE:
    ndims += 2; // mom, t
    break;
  case POSITION_SPACE:
    ndims += 4; // t, z, y, x
    break;
  default:
    errorQuda("Corralator: CorrSpace not supported: %d\n", CorrSpace);
  }
  ndims+=nDims[CorrType];
 
  return ndims;
}

static void fillDims(CORR_TYPE CorrType, CORR_SPACE CorrSpace, int ndims,
		     hsize_t* dims, hsize_t* ldims, hsize_t* start, int *sourcePosition, bool shift_source) {
  start[ndims-1] = 0; ldims[ndims-1] = dims[ndims-1] = 2; //re-im
  switch(CorrSpace) {
  case MOMENTUM_SPACE:
    start[ndims-2] = 0; ldims[ndims-2] = dims[ndims-2] = GK_Nmoms; //Nmoms
    start[ndims-3] = GK_timeRank*GK_localL[3]; //starting point
    ldims[ndims-3] = GK_localL[3]; //LT
    dims[ndims-3] = GK_totalL[3]; //T
    if(shift_source) {
      start[ndims-3] = (start[ndims-3] + GK_totalL[3] - sourcePosition[3]) % GK_totalL[3];
    }
    break;
  case POSITION_SPACE:
    for(int i=0; i<N_DIMS; i++) {
      start[ndims-1-i] = comm_coords(default_topo)[i]*GK_localL[i]; //starting
      ldims[ndims-1-i] = GK_localL[i]; //LT
      dims[ndims-1-i] = GK_totalL[i]; //T
      if(shift_source) {
	start[ndims-1-i] = (start[ndims-1-i] + GK_totalL[i] - sourcePosition[i]) % GK_totalL[3];
      }
    }
    break;
  default:
    errorQuda("Corralator: CorrSpace not supported: %d\n", CorrSpace);
  }
  switch(CorrType) {
  case MESONS:
    break;
  case BARYONS:
    start[0] = 0; ldims[0] = dims[0] = 16; //n-gamma
    break;
  case THRP_LOCAL:
    start[0] = 0; ldims[0] = dims[0] = 16; //n-gamma
    break;
  case THRP_NOETHER:
    start[0] = 0; ldims[0] = dims[0] = 4; //n-gamma
    break;
  case THRP_ONED:
    start[1] = 0; ldims[1] = dims[1] = 16; //n-gamma
    start[0] = 0; ldims[0] = dims[0] = 4; //n-dirs
    break;
  default:
    errorQuda("Corralator: CorrType not supported: %d\n", CorrType);
  }
}

/* Attribute writing */
static void write_text_attribute(hid_t group_id,const char* attr_name, char* attr_value) {
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
    printfQuda("writeHDF5: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE_H5 = H5T_NATIVE_DOUBLE;
    printfQuda("writeHDF5: Will write in double precision\n");
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
writeHDF5(char *filename, PLEGMA_params *info) {
  // only one per time writes in momentum space
  if(corr_space == MOMENTUM_SPACE && (GK_timeRank > GK_nProc[3] || GK_timeRank <0 ))
    return;

  MPI_Comm comm;
  bool shift_source = false;
  switch(corr_space) {
  case MOMENTUM_SPACE:
    comm = GK_timeComm;
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

  Float* corrHDF5 = (Float*) malloc(bytes_total_length);
  // exchanging volume and site
  for(int v=0; v<vol_size; v++){
    for(int s=0; s<site_size; s++){
      corrHDF5[(s*vol_size + v)*2 + 0] = corr[(v*site_size + s)*2 + 0];
      corrHDF5[(s*vol_size + v)*2 + 1] = corr[(v*site_size + s)*2 + 1];
    }
  }

  int ndims = getNDims(corr_type, corr_space);
  hsize_t dims[ndims], ldims[ndims], start[ndims];
  fillDims(corr_type, corr_space, ndims, dims, ldims, start, GK_sourcePosition[isource], shift_source);

  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);

  char *group1_tag;
  asprintf(&group1_tag,"conf_%04d",info->traj);
  hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  char *group2_tag;
  asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  //- Source position
  char *src_pos;
  asprintf(&src_pos," [x, y, z, t] = [%02d, %02d, %02d, %02d]\0",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
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
      Float *writeBuf = corrHDF5 + (g*n_flavors+d)*spaceSize;
      write_dataset(group3_id, corr_flavors_names[corr_type][d], writeBuf,
		    ndims, dims, ldims, start);
    }
    H5Gclose(group3_id);
  }

  H5Gclose(group2_id);
  H5Gclose(group1_id);
  H5Fclose(file_id);
  free(corrHDF5);
}

template class PLEGMA_Correlator<float>;
template class PLEGMA_Correlator<double>;
