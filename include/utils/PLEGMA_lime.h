#pragma once
extern "C" {
#include <lime.h>
}

#include <unistd.h>
#include <stdio.h>

inline bool exists_file (const char* name) {
  return ( access( name, F_OK ) != -1 );
}

static void qcd_swap_8(double *Rd, int N)
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

static void qcd_swap_4(float *Rd, int N)
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

static int qcd_isBigEndian()
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

static char* qcd_getParam(char token[],char* params,int len)
{
   int i,token_len=strlen(token);

   for(i=0;i<len-token_len;i++)
   {
      if(memcmp(token,params+i,token_len)==0)
      {
         i+=token_len;
         *(strchr(params+i,'<'))='\0';
         break;
      }
   }
   return params+i;
}

static char* qcd_getParamComma(char token[],char* params,int len)
{
  int i,token_len=strlen(token);
  for(i=0;i<len-token_len;i++)
    {
      if(memcmp(token,params+i,token_len)==0)
        {
          i+=token_len;
          *(strchr(params+i,','))='\0';
          break;
        }
    }
  return params+i;
}
