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
  char * lime_data = (char * )malloc(lime_data_size);
  limeReaderReadData((void *)lime_data, &lime_data_size, limereader);
  
  double dDummy;
  sscanf(getParamComma("kappa =",lime_data, lime_data_size),"%lf",&dDummy);    
  printfQuda("Kappa conf is : %.8f\n", dDummy);
  
  sscanf(getParamComma("mu =",lime_data, lime_data_size),"%lf",&dDummy);
  printfQuda("Mu conf is : %f\n", dDummy);
  
  free(lime_data);
}

static void print_ildg_format(LimeReader *limereader) {
  n_uint64_t lime_data_size = limeReaderBytes(limereader);
  char * lime_data = (char * )malloc(lime_data_size);
  limeReaderReadData((void *)lime_data, &lime_data_size, limereader);
  
  int iDummy, ln[4];
  sscanf(getParam("<precision>", lime_data, lime_data_size),"%i",&iDummy);    
  printfQuda("Precision:\t%i bit\n",iDummy);
  if(iDummy != 64) warningQuda("Only double precision supported (64). Continuing...\n");
	      
  sscanf(getParam("<lx>", lime_data, lime_data_size),"%i",&iDummy);
  if(iDummy != HGC_totalL[0]) warningQuda("Read lx different from HGC_totalL[0]. Continuing...\n");
  ln[0] = iDummy;

  sscanf(getParam("<ly>", lime_data, lime_data_size),"%i",&iDummy);
  if(iDummy != HGC_totalL[1]) warningQuda("Read ly different from HGC_totalL[1]. Continuing...\n");
  ln[1] = iDummy;

  sscanf(getParam("<lz>", lime_data, lime_data_size),"%i",&iDummy);
  if(iDummy != HGC_totalL[2]) warningQuda("Read lz different from HGC_totalL[2]. Continuing...\n");
  ln[2] = iDummy;

  sscanf(getParam("<lt>", lime_data, lime_data_size),"%i",&iDummy);
  if(iDummy != HGC_totalL[3]) warningQuda("Read lt different from HGC_totalL[3]. Continuing...\n");
  ln[3] = iDummy;
  
  printfQuda("Volume:   \t%ix%ix%ix%i\n", ln[0], ln[1], ln[2], ln[3]);
	      
  free(lime_data);
}
