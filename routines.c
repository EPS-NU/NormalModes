
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>

#include "routines.h"


int next_2(const int n){
  return (int) ceil(log10((double) n) / log10(2.0));
}

void read_stick_file(const char* f, int* n, double** w, complex** z){

  int i;
  double wtmp, atmp, btmp;
  char buf[BUFFER];
  char delim[] = "(,)";
  char* token;  
  char* err;
  FILE* stick = fopen(f,"r");

  if( (err = fgets(buf,BUFFER,stick)) == NULL) exit(EXIT_FAILURE);
  sscanf(buf,"%d %*s %*s", n);

  *w = malloc(sizeof(double) * (*n));
  *z = malloc(sizeof(complex) * (*n));

  for(i=0;i<(*n);i++){
    if( (err = fgets(buf,BUFFER,stick)) == NULL) exit(EXIT_FAILURE);
    sscanf(buf,"%lf",&wtmp);
    *(*w+i) = 2*M_PI/wtmp;
  }

  for(i=0;i<(*n);i++){
    if( (err = fgets(buf,BUFFER,stick)) == NULL) exit(EXIT_FAILURE);
    token = strtok(buf,delim);
    token = strtok(NULL,delim);
    sscanf(token,"%lf",&atmp);
    token = strtok(NULL,delim);
    sscanf(token,"%lf",&btmp);
    *(*z+i) = atmp + I*btmp;
  }


  fclose(stick);
}

void get_config_params(const char* f, struct config_params* cp){

  int i;
  char buf[BUFFER];
  char* err;
  FILE* config = fopen(f,"r");

  if((err=fgets(buf,BUFFER,config))==NULL) exit(EXIT_FAILURE);
  sscanf(buf,"%d", &(cp->nsta));

  cp->sta = (char**) malloc(sizeof(char*) * cp->nsta);
  for( i = 0; i < cp->nsta; i++){
    if((err=fgets(buf,BUFFER,config))==NULL) exit(EXIT_FAILURE);
    cp->sta[i] = malloc(strlen(buf)+1);
    sscanf(buf,"%s",cp->sta[i]);
  }

  fclose(config);
}

void get_moment_components(const char* m, struct moment_tensor* mt){

  int i;
  char buf[BUFFER];
  char* err;
  FILE* moment = fopen(m,"r");

  if((err=fgets(buf,BUFFER,moment))==NULL) exit(EXIT_FAILURE);
  sscanf(buf,"%le",&(mt->mrr));

  if((err=fgets(buf,BUFFER,moment))==NULL) exit(EXIT_FAILURE);
  sscanf(buf,"%le",&(mt->mtt));
  
  if((err=fgets(buf,BUFFER,moment))==NULL) exit(EXIT_FAILURE);
  sscanf(buf,"%le",&(mt->mpp));
  
  if((err=fgets(buf,BUFFER,moment))==NULL) exit(EXIT_FAILURE);
  sscanf(buf,"%le",&(mt->mrt));
  
  if((err=fgets(buf,BUFFER,moment))==NULL) exit(EXIT_FAILURE);
  sscanf(buf,"%le",&(mt->mrp));
  
  if((err=fgets(buf,BUFFER,moment))==NULL) exit(EXIT_FAILURE);
  sscanf(buf,"%le",&(mt->mtp));

  fclose(moment);
}

int exists(const char* fname){
  FILE* file;
  if((file = fopen(fname, "r"))){
    fclose(file);
    return 1;
  }
  return 0;
}

void import_plan(const char* s){
  int plan_imported = 0;
  if(exists(s)){
    plan_imported = fftw_import_wisdom_from_filename(s);
    if(!plan_imported){
      fprintf(stderr,"Plan was not able to be imported\n");
    }
  }
}

void save_plan(const char* s){
  int plan_saved = 0;
  if(!exists(s)){
    plan_saved = fftw_export_wisdom_to_filename(s);
    if(!plan_saved){
      fprintf(stderr,"Plan was not able to be saved\n");
    }
  }
}

double get_Q(const char* mode, const char* dir){ 

  double q;
  char buf[BUFFER];
  char path[BUFFER];
  char* err;

  strcpy(path,dir);
  strcat(path,mode);
  strcat(path,"/Q");

  FILE* fp = fopen(path,"r");

  if((err=fgets(buf,BUFFER,fp))==NULL) exit(EXIT_FAILURE);
  sscanf(buf,"%lf",&q);

  fclose(fp);
  return q;
}

complex synt_fft(complex z, double w0, double w, double Q, double T){
  return z * (cexp( I*(w0-w)*T  - (w0/(2.0*Q))*T)  - 1) / ( I*(w0-w) - w0/(2.0*Q) );
}
