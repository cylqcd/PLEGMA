#pragma once
extern "C" {
#include <lime.h>
}

#include <unistd.h>
#include <stdio.h>

inline bool exists_file (const char* name) {
  return ( access( name, F_OK ) != -1 );
}

static void swap_8(double *Rd, int N)
{
   register char *i,*j,*k;
   char swap;
   char *max;
   char *R = (char*) Rd;

   max = R+(N<<3);
   for(i=R;i<max;i+=8)
   {
      j=i; k=j+7;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
   }
}

static void swap_4(float *Rd, int N)
{
  register char *i,*j,*k;
  char swap;
  char *max;
  char *R =(char*) Rd;

  max = R+(N<<2);
  for(i=R;i<max;i+=4)
  {
    j=i; k=j+3;
    swap = *j; *j = *k;  *k = swap;
    j++; k--;
    swap = *j; *j = *k;  *k = swap;
  }
}

static int isBigEndian()
{
   union{
     char C[4];
     int  R   ;
        }word;
   word.R=1;
   if(word.C[3]==1) return 1;
   if(word.C[0]==1) return 0;

   return -1;
}

static char* getParam(const char* token,char* params,int len) {
   int i,token_len=strlen(token);
   for(i=0;i<len-token_len;i++) {
     if(memcmp(token,params+i,token_len)==0) {
       i+=token_len;
       *(strchr(params+i,'<'))='\0';
       break;
     }
   }
   return params+i;
}

static char* getParamComma(const char * token, char* params, int len) {
  int i,token_len=strlen(token);
  for(i=0;i<len-token_len;i++) {
    if(memcmp(token,params+i,token_len)==0) {
      i+=token_len;
      *(strchr(params+i,','))='\0';
      break; 
    }
  }
  return params+i;
}

static void print_xlf_info(LimeReader *limereader) {
  n_uint64_t lime_data_size = limeReaderBytes(limereader);
  char * lime_data;
  hostMalloc(lime_data, lime_data_size);
  limeReaderReadData((void *)lime_data, &lime_data_size, limereader);
  
  double dDummy;
  sscanf(getParamComma("kappa =",lime_data, lime_data_size),"%lf",&dDummy);    
  if(HGC_verbosity) PLEGMA_printf("Kappa conf is : %.8f\n", dDummy);
  
  sscanf(getParamComma("mu =",lime_data, lime_data_size),"%lf",&dDummy);
  if(HGC_verbosity) PLEGMA_printf("Mu conf is : %f\n", dDummy);
  
  hostFree(lime_data, lime_data_size);
}

static int print_ildg_format(LimeReader *limereader) {
  n_uint64_t lime_data_size = limeReaderBytes(limereader);
  char * lime_data;
  hostMalloc(lime_data, lime_data_size);
  limeReaderReadData((void *)lime_data, &lime_data_size, limereader);
  
  int iDummy, ln[4],precision;
  sscanf(getParam("<precision>", lime_data, lime_data_size),"%i",&iDummy);
  precision = iDummy/8;
  if(HGC_verbosity) PLEGMA_printf("Precision:\t%i bit\n",iDummy);
  //  if(iDummy != 64) PLEGMA_warning("Only double precision supported (64). Continuing...\n");
	      
  sscanf(getParam("<lx>", lime_data, lime_data_size),"%i",&iDummy);
  if(iDummy != HGC_totalL[0]) PLEGMA_warning("Read lx different from HGC_totalL[0]. Continuing...\n");
  ln[0] = iDummy;

  sscanf(getParam("<ly>", lime_data, lime_data_size),"%i",&iDummy);
  if(iDummy != HGC_totalL[1]) PLEGMA_warning("Read ly different from HGC_totalL[1]. Continuing...\n");
  ln[1] = iDummy;

  sscanf(getParam("<lz>", lime_data, lime_data_size),"%i",&iDummy);
  if(iDummy != HGC_totalL[2]) PLEGMA_warning("Read lz different from HGC_totalL[2]. Continuing...\n");
  ln[2] = iDummy;

  sscanf(getParam("<lt>", lime_data, lime_data_size),"%i",&iDummy);
  if(iDummy != HGC_totalL[3]) PLEGMA_warning("Read lt different from HGC_totalL[3]. Continuing...\n");
  ln[3] = iDummy;
  
  if(HGC_verbosity) PLEGMA_printf("Volume:   \t%ix%ix%ix%i\n", ln[0], ln[1], ln[2], ln[3]);
	      
  hostFree(lime_data, lime_data_size);
  return precision;
}

static int gat_lime_header(LimeReader *limereader){
  int precision=0;
  while(limeReaderNextRecord(limereader) != LIME_EOF ) {
    char* lime_type = limeReaderType(limereader); 
    if(strcmp(lime_type,"ildg-binary-data") == 0)
      break;
    if(strcmp(lime_type,"xlf-info")==0)
      print_xlf_info(limereader);
    if(strcmp(lime_type,"ildg-format")==0)
      precision = print_ildg_format(limereader);
  }
  if((precision != 4) && (precision != 8)) PLEGMA_error("Precision %d is not supported",precision);
  return precision;
}

template<typename Float>
static void read_from_lime(FILE *fid, LimeReader *limereader, Float *data, int dof){
#ifdef	MULTI_GPU
  // Read 1 byte to set file-pointer to start of binary data
  n_uint64_t one=1;
  char dummy;
  limeReaderReadData(&dummy,&one,limereader);
  MPI_Offset offset = ftell(fid)-1;
#endif

  Float *ftmp;
  long int sizeVec=((long int) dof)*HGC_localVolume;
  hostMalloc(ftmp, sizeVec*2*sizeof(Float));

#ifdef	MULTI_GPU
  MPI_Datatype subblock;  //MPI-type, (N_DIMS+1)d subarray
  MPI_File mpifid;
  MPI_Status status;
  int sizes[N_DIMS+1], lsizes[N_DIMS+1], starts[N_DIMS+1];
  for(int i=0; i<N_DIMS; i++) {
    sizes[i] = HGC_totalL[N_DIMS-1-i];
    lsizes[i] = HGC_localL[N_DIMS-1-i];
    starts[i] = HGC_procPosition[N_DIMS-1-i]*HGC_localL[N_DIMS-1-i];
  }
  lsizes[N_DIMS] = sizes[N_DIMS] = dof*2;
  starts[N_DIMS] = 0;

  MPI_Type_create_subarray(N_DIMS+1,sizes,lsizes,starts,MPI_ORDER_C,MPI_Type(data),&subblock);
  MPI_Type_commit(&subblock);
	
  MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpifid);
  MPI_File_set_view(mpifid, offset, MPI_Type(data), subblock, "native", MPI_INFO_NULL);

  if(sizeVec*2 > 2147483648) PLEGMA_warning("Be careful for possible integer overflow in MPI_File_read_all function");
      
  if(MPI_File_read_all(mpifid, ftmp, sizeVec*2, MPI_Type(data), &status) == 1)
    PLEGMA_error("Error in MPI_File_read_all\n");
  MPI_File_close(&mpifid);  
#else
  if(fread(ftmp, sizeof(Float), sizeVec*2, fid) != sizeVec*2) {
    PLEGMA_error("Error, could not read proper amount of data");
  }
#endif
  if(!isBigEndian()){
    if(sizeof(Float) == 8)swap_8(ftmp,sizeVec*2);
    else if(sizeof(Float) == 4) swap_4(ftmp,sizeVec*2);
    else PLEGMA_error("Cannot byte swap with this precision");
  }

  for(size_t i = 0; i < HGC_localVolume; i++) {
    Float U[dof][2];
    memcpy(U[0], ftmp + ((long int) i)*dof*2, 2*dof*sizeof(Float));
    for(int s = 0; s < dof; s++) {
      data[(s*HGC_localVolume + i)*2 + 0] = U[s][0];
      data[(s*HGC_localVolume + i)*2 + 1] = U[s][1];
    }
  }
  hostFree(ftmp, sizeVec*2*sizeof(Float));
}
