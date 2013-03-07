/* combosynt_iter.c
 *
 * This program will read a SAC binary file and take its FFT.
 * Then, it will read in computed eigenfrequncies and associated
 * complex spectra for a specific mode, computed using the algorithms of 
 * Stein and Geller in splitpar.f. Then, from Buland and Gilbert (1978),
 * a synthetic FFT is generated 
 *
 * a synthetic seismogram, whose fourier spectrum will be compared
 * against real data. By comparing the fit of the synthetic to the
 * data, we can retrieve a more precise measure of the amount of 
 * energy released by the earthquake, as well as the earthquake's
 * source parameters.
 *
 * Author: Michael Witek
 *   Date: 2/1/2013
 *
 *
 *   gcc solveLS.c -lfftw3 -lm -llapacke -llapack -lrefblas -lgfortran -I$SACHOME/include -L$SACHOME/lib -lsacio -O2 -g3 -Wall -o solveLS.x
 *   gcc solveLS.c -lfftw3 -lm -llapacke -llapack -lcblas -lblas -lgfortran -I$SACHOME/include -L$SACHOME/lib -lsacio -O2 -g3 -o solveLS.x 
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <complex.h>
#include <lapacke.h>
#include <cblas.h>
#include <fftw3.h>
#include <math.h>
#include "sacio.h"

#define SAC_DATA_MAX 500000
#define BUFFER 256

struct config_params{
  int nsta; // number of stations
  char** sta;  // list of station names 
};

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


int main(int argc, char** argv){

  /* general vars */
  int h, i, j, k, l, m;
  double qmode;
  double nm2cm = 10e-7;  // conversion factor for nm to cm

  int Nxdt;

  double omega = 0.0, dw;
  double Tlo, Thi, wlo, whi;
  double diff;
  int start, end;

  double rms_new = 0, rms_last = 0;
  int keepLooping = 1;

  /* hardcode for 0s4 */
  Tlo = 1490;
  Thi = 1590;

  wlo = (2.0*M_PI)/Thi;
  whi = (2.0*M_PI)/Tlo;

  complex* x = NULL;


  /* moment tensor components */
  /*
   double mrr1 = 0.750;
   double mrr2 = 0.750;
   * 
   double mtt1 = -0.32547;
   double mtt2 = -0.32547;
   * 
   double mpp1 = -0.42453;
   double mpp2 = -0.42453;

   double mtp1 = -0.44796;
   double mtp2 = -0.44796;

   double mrt1 = -0.57544;
   double mrt2 = -0.57544;

   double mrp1 = -0.20944;
   double mrp2 = -0.20944;
   */

  double mrr1 = 1;
  double mrr2 = 1;

  double mtt1 = 1;
  double mtt2 = 1;

  double mpp1 = 1;
  double mpp2 = 1;

  double mtp1 = 1;
  double mtp2 = 1;

  double mrt1 = 1;
  double mrt2 = 1;

  double mrp1 = 1;
  double mrp2 = 1;


  /* sac lib vars */
  int len, newlen, error, max = SAC_DATA_MAX;
  float beg, delta;

  float tmp[SAC_DATA_MAX];

  /* stick file vars */
  int nsing1, nsing2, nsing3, nsing4, nsing5;
  int nsing6, nsing7, nsing8, nsing9, nsing10;
  double* w1= NULL, *w2= NULL, *w3= NULL, *w4= NULL, *w5 = NULL;
  double* w6= NULL, *w7= NULL, *w8= NULL, *w9= NULL, *w10 = NULL;
  complex* z1 = NULL, *z2 = NULL, *z3 = NULL, *z4 = NULL, *z5 = NULL;
  complex* z6 = NULL, *z7 = NULL, *z8 = NULL, *z9 = NULL, *z10 = NULL;

  /* directory paths */

  const char modedir[] = "/home/mwitek/research/sumatra/data/modes/";
  const char datadir[] = "/home/mwitek/research/sumatra/data/sac_bin/";
  char sf1[BUFFER], sf2[BUFFER], sf3[BUFFER], sf4[BUFFER], sf5[BUFFER], sac_fname[BUFFER];
  char sf6[BUFFER], sf7[BUFFER], sf8[BUFFER], sf9[BUFFER], sf10[BUFFER];
  strcpy(sf1,modedir);
  strcpy(sf2,modedir);
  strcpy(sf3,modedir);
  strcpy(sf4,modedir);
  strcpy(sf5,modedir);
  strcpy(sf6,modedir);
  strcpy(sf7,modedir);
  strcpy(sf8,modedir);
  strcpy(sf9,modedir);
  strcpy(sf10,modedir);
  strcpy(sac_fname,datadir);

  /* configuration file, input parameters */
  struct config_params params;
  char* conf_file = malloc(strlen(argv[1])+1);
  sscanf(argv[1],"%s",conf_file);
  get_config_params(conf_file, &params);

  /* fftw vars */
  fftw_complex* fft_data;
  fftw_complex* data;
  fftw_complex* fft_synt;
  fftw_plan data_plan;

  /* LAPACK vars */
  lapack_complex_double ***Gminor, **Gmajor, *G;
  lapack_complex_double **dminor, *dmajor, *tmpd;
  lapack_int Gm, Gn, Gld, info, Dld, nrhs;

  /* Gminor will contain the sub blocks of the G matrix, Gmajor 
   * it will have as many blocks as stations being processed
   * likewise for dminor
   */

  qmode = get_Q("0s4", modedir);

//  while( keepLooping == 1){

    Gminor = (lapack_complex_double***) calloc(params.nsta, sizeof(lapack_complex_double**));
    dminor = (lapack_complex_double**) malloc(sizeof(lapack_complex_double*) * params.nsta);

    for(i = 0; i < params.nsta; i++){ 
      // for(i = 0; i < 1; i++){


      /* test for one mode, one station before expanding */

      /* read the stick files for current station */
      /* all these strings will need to be reinitialized if there is more than one station
       * being processed. Otherwise the strings will just keep expanding 
       */
      strcat(sf1,"0s4/0s4.");
      strcat(sf1,params.sta[i]);
      strcat(sf1,".Mrt.1");

      strcat(sf2,"0s4/0s4.");
      strcat(sf2,params.sta[i]);
      strcat(sf2,".Mrp.1");

      strcat(sf3,"0s4/0s4.");
      strcat(sf3,params.sta[i]);
      strcat(sf3,".Mtp.1");

      strcat(sf4,"0s4/0s4.");
      strcat(sf4,params.sta[i]);
      strcat(sf4,".Mrrpp.1");

      strcat(sf5,"0s4/0s4.");
      strcat(sf5,params.sta[i]);
      strcat(sf5,".Mrrtt.1");

      strcat(sf6,"0s4/0s4.");
      strcat(sf6,params.sta[i]);
      strcat(sf6,".Mrt.2");

      strcat(sf7,"0s4/0s4.");
      strcat(sf7,params.sta[i]);
      strcat(sf7,".Mrp.2");

      strcat(sf8,"0s4/0s4.");
      strcat(sf8,params.sta[i]);
      strcat(sf8,".Mtp.2");

      strcat(sf9,"0s4/0s4.");
      strcat(sf9,params.sta[i]);
      strcat(sf9,".Mrrpp.2");

      strcat(sf10,"0s4/0s4.");
      strcat(sf10,params.sta[i]);
      strcat(sf10,".Mrrtt.2");


      printf("sf1: %s\n",sf1);

      /* !! seg fault if sf1/sf2 doesn't exist !! */

      read_stick_file(sf1,&nsing1,&w1,&z1);
      read_stick_file(sf2,&nsing2,&w2,&z2);
      read_stick_file(sf3,&nsing3,&w3,&z3);
      read_stick_file(sf4,&nsing4,&w4,&z4);
      read_stick_file(sf5,&nsing5,&w5,&z5);
      read_stick_file(sf6,&nsing6,&w6,&z6);
      read_stick_file(sf7,&nsing7,&w7,&z7);
      read_stick_file(sf8,&nsing8,&w8,&z8);
      read_stick_file(sf9,&nsing9,&w9,&z9);
      read_stick_file(sf10,&nsing10,&w10,&z10);

      printf("nsing1 = %d \n",nsing1);
      /* read the sac binary file for the current station */
      strcat(sac_fname,"12102.");
      strcat(sac_fname,params.sta[i]);

      printf("sac_fname: %s\n",sac_fname);

      /* initialize tmp to 0 to kill off crap from last read */
      for(j = 0; j < SAC_DATA_MAX; j++) tmp[j] = 0.0;

      rsac1(sac_fname, tmp, &len, &beg, &delta, &max, &error, strlen(sac_fname));
      if(error){
        fprintf(stderr,"Error reading in file(%d): %s\n",error, sac_fname);
        continue;
      }

      /* figure out next power of two and allocate memory for fftw arrays */
      newlen = (int) pow(2.0, (double) next_2(len));
 //     printf("next 2 = %d\n",newlen);

      data = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * newlen);
      fft_data = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * newlen);
      fft_synt = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * newlen);


      for(j = 0; j < len; j++) data[j] = tmp[j] + I*0.0;
      for(j = len; j < newlen; j++) data[j] = 0.0 + I*0.0;

      import_plan("combosynt_data.plan");
      data_plan = fftw_plan_dft_1d(newlen, data, fft_data, FFTW_FORWARD, FFTW_MEASURE);
      save_plan("combosynt_data.plan");

      // tmp has data in units of nm, convert to cm

      for(j = 0; j < len; j++) data[j] = (double) nm2cm*tmp[j] + I*0.0;
      for(j = len; j < newlen; j++) data[j] = 0.0 + I*0.0;

      if(data_plan != NULL){
        fftw_execute(data_plan);
      } else {
        fprintf(stderr,"Sorry, could not execute forward FFT\n");
        continue;
      }

 //     printf("FFT done\n");



      dw = (2.0*M_PI)/(newlen*delta);

      /* find j for which wlo is closest to a frequency in fft_data */

      printf("wlo = %le\ndw = %le\n", wlo/(2.0*M_PI), dw);
      start = 0;
      for(j = 1; j < newlen; j++){
        diff = fabs(wlo - j*dw);
        if(diff < fabs(wlo - (j-1)*dw)){
          start = j;
        }

      }
//      printf("start = %d\n",start);
      wlo = start*dw; /* redefine wlo so that the synthetic data starts at the same frequency */

      end = 0;
      for(j = 1; j < newlen; j++){
        diff = fabs(whi - j*dw);
        if(diff < fabs(whi - (j-1)*dw)){
          end = j;
        }
      }
      whi = end*dw; /* same as wlo above */

//      printf("end = %d\n whi = %le, wlo = %le\n", end, whi/(2.0*M_PI), wlo/(2.0*M_PI));

      printf("writing fft of data to disk\n");
      FILE* fp2 = fopen("data.dat","w");
      for(j = start; j <= end ; j++){
        fprintf(fp2,"%0.8lE %0.8lE\n",j*(1.0)/(delta*newlen),cabs(fft_data[j]));
      }
      fclose(fp2);

      /* Allocate space for sub-block in G */
      m = (whi - wlo)/dw +1;
  //    printf("m = %d\n", m);


      Gminor[i] = (lapack_complex_double**) calloc(m, sizeof(lapack_complex_double*)); 
      for(j = 0; j < m; j++){
        Gminor[i][j] = (lapack_complex_double*) calloc(10, sizeof(lapack_complex_double) );
      }

      /*
       printf("%d\n",nsing1);
       printf("%d\n",nsing2);
       printf("%d\n",nsing3);
       printf("%d\n",nsing4);
       printf("%d\n",nsing5);
       printf("%d\n",nsing6);
       printf("%d\n",nsing7);
       printf("%d\n",nsing8);
       printf("%d\n",nsing9);
       printf("%d\n",nsing10);
       */

      dminor[i] = (lapack_complex_double*) malloc(sizeof(lapack_complex_double) * m);
      for(j = 0; j < m; j++){
        dminor[i][j] = fft_data[start+j];
      }

//      printf("Filling block\n");
      /* begin filling block */

      Nxdt = newlen*delta;

      FILE* agmatrix = fopen("agmatrix.dat","w");
      omega = wlo;
      k = 0;
      while(omega <= whi && k < m){

        for( j = 0; j < nsing1; j++){

          Gminor[i][k][0] +=  (z1[j]  * (cexp( I*(w1[j]-omega)*Nxdt  - (w1[j]/(2.0*qmode))*Nxdt)  - 1) / ( I*(w1[j]-omega) - w1[j]/(2.0*qmode) ));
          Gminor[i][k][1] +=  (z2[j]  * (cexp( I*(w2[j]-omega)*Nxdt  - (w2[j]/(2.0*qmode))*Nxdt)  - 1) / ( I*(w2[j]-omega) - w2[j]/(2.0*qmode) ));
          Gminor[i][k][2] +=  (z3[j]  * (cexp( I*(w3[j]-omega)*Nxdt  - (w3[j]/(2.0*qmode))*Nxdt)  - 1) / ( I*(w3[j]-omega) - w3[j]/(2.0*qmode) ));
          Gminor[i][k][3] -=  (z4[j]  * (cexp( I*(w4[j]-omega)*Nxdt  - (w4[j]/(2.0*qmode))*Nxdt)  - 1) / ( I*(w4[j]-omega) - w4[j]/(2.0*qmode) ));
          Gminor[i][k][4] -= (z5[j]  * (cexp( I*(w5[j]-omega)*Nxdt  - (w5[j]/(2.0*qmode))*Nxdt)  - 1) / ( I*(w5[j]-omega) - w5[j]/(2.0*qmode) ));

          /* add second earthquake */
          Gminor[i][k][5] +=  (z6[j]  * (cexp( I*(w6[j]-omega)*Nxdt  - (w6[j]/(2.0*qmode))*Nxdt)  - 1) / ( I*(w6[j]-omega)  - w6[j]/(2.0*qmode) ));
          Gminor[i][k][6] +=  (z7[j]  * (cexp( I*(w7[j]-omega)*Nxdt  - (w7[j]/(2.0*qmode))*Nxdt)  - 1) / ( I*(w7[j]-omega)  - w7[j]/(2.0*qmode) ));
          Gminor[i][k][7] +=  (z8[j]  * (cexp( I*(w8[j]-omega)*Nxdt  - (w8[j]/(2.0*qmode))*Nxdt)  - 1) / ( I*(w8[j]-omega)  - w8[j]/(2.0*qmode) ));
          Gminor[i][k][8] -=  (z9[j]  * (cexp( I*(w9[j]-omega)*Nxdt  - (w9[j]/(2.0*qmode))*Nxdt)  - 1) / ( I*(w9[j]-omega)  - w9[j]/(2.0*qmode) ));
          Gminor[i][k][9] -=  (z10[j] * (cexp( I*(w10[j]-omega)*Nxdt - (w10[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w10[j]-omega) - w10[j]/(2.0*qmode) ));
        }


        // printf("k = %d, w = %le\n", k, omega);
        
         fprintf(agmatrix,"(%le %le), ", creal(Gminor[i][k][0]), cimag(Gminor[i][k][0]));
         fprintf(agmatrix,"(%le %le), ", creal(Gminor[i][k][1]), cimag(Gminor[i][k][1]));
         fprintf(agmatrix,"(%le %le), ", creal(Gminor[i][k][2]), cimag(Gminor[i][k][2]));
         fprintf(agmatrix,"(%le %le), ", creal(Gminor[i][k][3]), cimag(Gminor[i][k][3]));
         fprintf(agmatrix,"(%le %le), ", creal(Gminor[i][k][4]), cimag(Gminor[i][k][4]));
         fprintf(agmatrix,"(%le %le), ", creal(Gminor[i][k][5]), cimag(Gminor[i][k][5]));
         fprintf(agmatrix,"(%le %le), ", creal(Gminor[i][k][6]), cimag(Gminor[i][k][6]));
         fprintf(agmatrix,"(%le %le), ", creal(Gminor[i][k][7]), cimag(Gminor[i][k][7]));
         fprintf(agmatrix,"(%le %le), ", creal(Gminor[i][k][8]), cimag(Gminor[i][k][8]));
         fprintf(agmatrix,"(%le %le) ",  creal(Gminor[i][k][9]), cimag(Gminor[i][k][9]));
         fprintf(agmatrix,"\n");
         
        k++;
        omega = omega + dw;

      }
//      printf("outside of block\n");


      fclose(agmatrix);

//      printf("block filled\n");

      fftw_free(fft_data);
      fftw_free(fft_synt);
      fftw_free(data);

      free(w1);
      free(z1);

      free(w2);
      free(z2);

      free(w3);
      free(z3);

      free(w4);
      free(z4);

      free(w5);
      free(z5);

      free(w6);
      free(z6);

      free(w7);
      free(z7);

      free(w8);
      free(z8);

      free(w9);
      free(z9);

      free(w10);
      free(z10);

      /* clear the strings */
      memset(sf1,  0, BUFFER);
      memset(sf2,  0, BUFFER);
      memset(sf3,  0, BUFFER);
      memset(sf4,  0, BUFFER);
      memset(sf5,  0, BUFFER);
      memset(sf6,  0, BUFFER);
      memset(sf7,  0, BUFFER);
      memset(sf8,  0, BUFFER);
      memset(sf9,  0, BUFFER);
      memset(sf10, 0, BUFFER);

      strcpy(sf1,      modedir);
      strcpy(sf2,      modedir);
      strcpy(sf3,      modedir);
      strcpy(sf4,      modedir);
      strcpy(sf5,      modedir);
      strcpy(sf6,      modedir);
      strcpy(sf7,      modedir);
      strcpy(sf8,      modedir);
      strcpy(sf9,      modedir);
      strcpy(sf10,     modedir);
      strcpy(sac_fname,datadir);

//      printf("strings cleared\n");
    }
    printf("params.nsta = %d\nm = %d\n",params.nsta, m);

/*
    Gmajor = (lapack_complex_double**) malloc(sizeof(lapack_complex_double*) * 10);
    for(i = 0; i < 10; i++){
      Gmajor[i] = (lapack_complex_double*) malloc(sizeof(lapack_complex_double) * m * params.nsta);
    }
*/
    G = (lapack_complex_double*) malloc(sizeof(lapack_complex_double) * 10 * m * params.nsta);
    
     Gmajor = (lapack_complex_double**) malloc(sizeof(lapack_complex_double*) * m * params.nsta);
     for(i = 0; i < m * params.nsta; i++){
       Gmajor[i] = (lapack_complex_double*) malloc(sizeof(lapack_complex_double) * 10);
  }
  

    printf("Gmajor allocated\n");

    /* combine blocks while transposing from row-major to column-major */
    /*
     FILE* fp = fopen("checkvalues.dat","w");
     l = 0;
     h = 0;
     for(i = 0; i < params.nsta; i++){
       for(j = 0; j < 10; j++){
         for(k = 0; k < m; k++){
           Gmajor[h] = Gminor[l][j+10*k];
           h++;
           fprintf(fp,"G[%d][%d] = %le %le \n",l,j+10*k,creal(Gminor[l][j+10*k]), cimag(Gminor[l][j+10*k]));
  }
  }
  l++;
  }
  fclose(fp);
  */

    printf("forming Gmajor array\n");

    k = 0;
    for( k = 0; k < params.nsta; k++){
      for( i = 0; i < m; i++){
        for(j = 0; j < 10; j++){
          Gmajor[i+k*m][j] = Gminor[k][i][j];
//        Gmajor[j][i] = Gminor[k][j][i];
//        if( (j % m) == 0 && j != 0) k++;
        }
      }
    }

    printf("forming G array \n");

    h = 0;
    for(j = 0; j < 10; j++){
      for(i = 0; i < m * params.nsta; i++){
        G[i+j*m*params.nsta] = Gmajor[i][j];
      }
    }

    // printf("params.nsta * m * 10 = %d, h = %d\n",params.nsta*m*10, h);

    printf("printing G to file \n");
    
     FILE* gmatrix = fopen("gmatrix.dat","w");
     for(i = 0; i < params.nsta * m * 10; i++){
       //  for(j = 0; j < params.nsta *m; j++){
       fprintf(gmatrix,"(%le %le)\n", creal(G[i]),cimag(G[i]));
       //  }
      }
      fclose(gmatrix);
  
    printf("Gmajor done\n");

    dmajor = (lapack_complex_double*) calloc(params.nsta*m, sizeof(lapack_complex_double));
    tmpd   = (lapack_complex_double*) calloc(params.nsta*m, sizeof(lapack_complex_double));
    for( i = 0; i < params.nsta ; i++){
      for( j = 0; j < m; j++){
        dmajor[i*m+j] = dminor[i][j];
        tmpd[i*m+j] = dmajor[i*m+j];
      }
    }

    FILE* dmatrix = fopen("dmatrix.dat","w");
    for(i = 0; i < params.nsta * m; i++){
      fprintf(dmatrix,"(%le, %le)\n", creal(dmajor[i]), cimag(dmajor[i]));
    }
    fclose(dmatrix);


    printf("dmajor done\n");

    Gm = m*params.nsta;
//    Gm = 10;
    Gn = 10;
//    Gn = m*params.nsta;
    Gld = m*params.nsta;
    Dld = m*params.nsta;
//    Dld = 1;
    nrhs = 1;

    printf("doing least squares....\n");

//    info = LAPACKE_zgels(LAPACK_ROW_MAJOR, 'N', Gm, Gn, nrhs, *Gmajor, Gld, dmajor, Dld); 
    info = LAPACKE_zgels(LAPACK_COL_MAJOR, 'N', Gm, Gn, nrhs, G, Gld, dmajor, Dld); 
    if(info < 0){
      printf("%d'th argument had illegal value\n",info);
      exit(EXIT_FAILURE);
    }

    printf("                      .........least squares done\n");

    printf(" Print out answer\n");

    for( i = 0; i < Gn; i++){
      for( j = 0; j < 1; j++){
        fprintf(stdout, "%lE, %lE, %lE ", creal(dmajor[i+Gm*j]), cimag(dmajor[i+Gm*j]), cabs(dmajor[i+Gm*j]));
      }
      printf("\n");
    }

    FILE* danswers = fopen("solutions.dat","w");
    for(i = 0; i < params.nsta * m; i++){
      fprintf(danswers,"%le %le %le\n",creal(dmajor[i]), cimag(dmajor[i]), cabs(dmajor[i]));
    }

    mrr1 = -(dmajor[3]+dmajor[4]);
    mrr2 = -(dmajor[8]+dmajor[9]);
    mtt1 = dmajor[4];
    mtt2 = dmajor[9];
    mpp1 = dmajor[3];
    mpp2 = dmajor[8];
    mrt1 = dmajor[0];
    mrt2 = dmajor[5];
    mrp1 = dmajor[1];
    mrp2 = dmajor[6];
    mtp1 = dmajor[2];
    mtp2 = dmajor[7];

    printf("Event 1:\n");
    printf(" Mrr = %le, Mtt = %le, Mpp = %le\n Mrt = %le, Mrp = %le, Mtp = %le\n",
           -(creal(dmajor[3])+creal(dmajor[4])),
           creal(dmajor[4]),
           creal(dmajor[3]),
           creal(dmajor[0]),
           creal(dmajor[1]),
           creal(dmajor[2]));

    printf("Event 2:\n");
    printf(" Mrr = %le, Mtt = %le, Mpp = %le\n Mrt = %le, Mrp = %le, Mtp = %le\n",
           -(creal(dmajor[8])+creal(dmajor[9])),
           creal(dmajor[9]),
           creal(dmajor[8]),
           creal(dmajor[5]),
           creal(dmajor[6]),
           creal(dmajor[7]));


    /* re-read everything to be able to print the forward problem using the results found */

    strcat(sf1,"0s4/0s4.");
    strcat(sf1,params.sta[0]);
    strcat(sf1,".Mrt.1");

    strcat(sf2,"0s4/0s4.");
    strcat(sf2,params.sta[0]);
    strcat(sf2,".Mrp.1");

    strcat(sf3,"0s4/0s4.");
    strcat(sf3,params.sta[0]);
    strcat(sf3,".Mtp.1");

    strcat(sf4,"0s4/0s4.");
    strcat(sf4,params.sta[0]);
    strcat(sf4,".Mrrpp.1");

    strcat(sf5,"0s4/0s4.");
    strcat(sf5,params.sta[0]);
    strcat(sf5,".Mrrtt.1");

    strcat(sf6,"0s4/0s4.");
    strcat(sf6,params.sta[0]);
    strcat(sf6,".Mrt.2");

    strcat(sf7,"0s4/0s4.");
    strcat(sf7,params.sta[0]);
    strcat(sf7,".Mrp.2");

    strcat(sf8,"0s4/0s4.");
    strcat(sf8,params.sta[0]);
    strcat(sf8,".Mtp.2");

    strcat(sf9,"0s4/0s4.");
    strcat(sf9,params.sta[0]);
    strcat(sf9,".Mrrpp.2");

    strcat(sf10,"0s4/0s4.");
    strcat(sf10,params.sta[0]);
    strcat(sf10,".Mrrtt.2");

    printf("%s\n",sf3);

    /* !! seg fault if sf1/sf2 doesn't exist !! */

    read_stick_file(sf1,&nsing1,&w1,&z1);
    read_stick_file(sf2,&nsing2,&w2,&z2);
    read_stick_file(sf3,&nsing3,&w3,&z3);
    read_stick_file(sf4,&nsing4,&w4,&z4);
    read_stick_file(sf5,&nsing5,&w5,&z5);
    read_stick_file(sf6,&nsing6,&w6,&z6);
    read_stick_file(sf7,&nsing7,&w7,&z7);
    read_stick_file(sf8,&nsing8,&w8,&z8);
    read_stick_file(sf9,&nsing9,&w9,&z9);
    read_stick_file(sf10,&nsing10,&w10,&z10);

    printf("sticks read\n");

    if( (x = (complex*) calloc(m , sizeof(complex))) == NULL){
      printf("Unable to allocate memory for x\n");
      exit(EXIT_FAILURE);
    }
    FILE* syntfft = fopen("syntfft.dat","w");
    omega = wlo;
    k=0;
    while(omega < whi && k < m){
      for( j = 0; j < nsing1; j++){

        /*
         x[k] += creal(dmajor[0]) * (z1[j] * (cexp( I*(w1[j]-omega)*Nxdt - (w1[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w1[j]-omega) - w1[j]/(2.0*qmode) ));
         x[k] += creal(dmajor[1]) * (z2[j] * (cexp( I*(w2[j]-omega)*Nxdt - (w2[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w2[j]-omega) - w2[j]/(2.0*qmode) ));
         x[k] += creal(dmajor[2]) * (z3[j] * (cexp( I*(w3[j]-omega)*Nxdt - (w3[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w3[j]-omega) - w3[j]/(2.0*qmode) ));
         x[k] -= creal(dmajor[3]) * (z4[j] * (cexp( I*(w4[j]-omega)*Nxdt - (w4[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w4[j]-omega) - w4[j]/(2.0*qmode) ));
         x[k] -= creal(dmajor[4]) * (z5[j] * (cexp( I*(w5[j]-omega)*Nxdt - (w5[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w5[j]-omega) - w5[j]/(2.0*qmode) ));
         */
        /* add second earthquake */
        /*
         x[k] += creal(dmajor[5]) * (z6[j] * (cexp( I*(w6[j]-omega)*Nxdt - (w6[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w6[j]-omega) - w6[j]/(2.0*qmode) ));
         x[k] += creal(dmajor[6]) * (z7[j] * (cexp( I*(w7[j]-omega)*Nxdt - (w7[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w7[j]-omega) - w7[j]/(2.0*qmode) ));
         x[k] += creal(dmajor[7]) * (z8[j] * (cexp( I*(w8[j]-omega)*Nxdt - (w8[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w8[j]-omega) - w8[j]/(2.0*qmode) ));
         x[k] -= creal(dmajor[8]) * (z9[j] * (cexp( I*(w9[j]-omega)*Nxdt - (w9[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w9[j]-omega) - w9[j]/(2.0*qmode) ));
         x[k] -= creal(dmajor[9]) * (z10[j] * (cexp( I*(w10[j]-omega)*Nxdt - (w10[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w10[j]-omega) - w10[j]/(2.0*qmode) ));
         */

        x[k] += (mrt1) * (z1[j] * (cexp( I*(w1[j]-omega)*Nxdt - (w1[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w1[j]-omega) - w1[j]/(2.0*qmode) ));
        x[k] += (mrp1) * (z2[j] * (cexp( I*(w2[j]-omega)*Nxdt - (w2[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w2[j]-omega) - w2[j]/(2.0*qmode) ));
        x[k] += (mtp1) * (z3[j] * (cexp( I*(w3[j]-omega)*Nxdt - (w3[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w3[j]-omega) - w3[j]/(2.0*qmode) ));
        x[k] -= (mpp1) * (z4[j] * (cexp( I*(w4[j]-omega)*Nxdt - (w4[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w4[j]-omega) - w4[j]/(2.0*qmode) ));
        x[k] -= (mtt1) * (z5[j] * (cexp( I*(w5[j]-omega)*Nxdt - (w5[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w5[j]-omega) - w5[j]/(2.0*qmode) ));

        /* add second earthquake */

        x[k] += (mrt2) * (z6[j] * (cexp( I*(w6[j]-omega)*Nxdt - (w6[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w6[j]-omega) - w6[j]/(2.0*qmode) ));
        x[k] += (mrp2) * (z7[j] * (cexp( I*(w7[j]-omega)*Nxdt - (w7[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w7[j]-omega) - w7[j]/(2.0*qmode) ));
        x[k] += (mtp2) * (z8[j] * (cexp( I*(w8[j]-omega)*Nxdt - (w8[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w8[j]-omega) - w8[j]/(2.0*qmode) ));
        x[k] -= (mpp2) * (z9[j] * (cexp( I*(w9[j]-omega)*Nxdt - (w9[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w9[j]-omega) - w9[j]/(2.0*qmode) ));
        x[k] -= (mtt2) * (z10[j] * (cexp( I*(w10[j]-omega)*Nxdt - (w10[j]/(2.0*qmode))*Nxdt) - 1) / ( I*(w10[j]-omega) - w10[j]/(2.0*qmode) ));

      }

 //     printf("%d\n",k);

      k++;
      omega += dw;
    }

    rms_last = rms_new;
    omega = wlo;
    j = 0;
    rms_new = 0;
    while(j < m){
      rms_new += pow(cabs(tmpd[j]) - cabs(x[j]), 2.0);
      j++;

    }
    rms_new = sqrt(rms_new/m);
    printf("rms = %le\n", rms_new);

    if( fabs(rms_new-rms_last) < FLT_EPSILON) keepLooping = 0;


    omega = wlo;
    j = 0;
    while(omega <= whi){
      fprintf(syntfft,"%0.8le %0.8le\n",omega/(2.0*M_PI),cabs(x[j]));
      j++;
      omega += dw;
    }
    fclose(syntfft);

    printf("synt made\n");

    free(w1);
    free(z1);
    free(w2);
    free(z2);
    free(w3);
    free(z3);
    free(w4);
    free(z4);
    free(w5);
    free(z5);
    free(w6);
    free(z6);
    free(w7);
    free(z7);
    free(w8);
    free(z8);
    free(w9);
    free(z9);
    free(w10);
    free(z10);

    printf("freed sticks\n");


    /*  deallocate memory */

    fftw_destroy_plan(data_plan);


    printf("plan destroyed\n");
    //  fftw_destroy_plan(synt_plan);

    for(i = 0; i < params.nsta; i++){
      free(dminor[i]);
    }
    free(dminor);
    printf("step 1\n");


    free(dmajor);
    free(tmpd);

    printf("step 2....freed data\n");

    for(i = 0; i < params.nsta; i++){
      for(j = 0; j < m; j++){
        free(Gminor[i][j]);
      }
      free(Gminor[i]);
    }

    printf("step 1\n");


    free(Gminor);
    printf("step 2...Gminor freed\n");


    for(i = 0; i < 10; i++){
      free(Gmajor[i]);
    }
    free(Gmajor);

    free(G);

    printf("freed g\n");
    

    free(x);
    /*

    memset(sf1,  0, BUFFER);
    memset(sf2,  0, BUFFER);
    memset(sf3,  0, BUFFER);
    memset(sf4,  0, BUFFER);
    memset(sf5,  0, BUFFER);
    memset(sf6,  0, BUFFER);
    memset(sf7,  0, BUFFER);
    memset(sf8,  0, BUFFER);
    memset(sf9,  0, BUFFER);
    memset(sf10, 0, BUFFER);

    strcpy(sf1,      modedir);
    strcpy(sf2,      modedir);
    strcpy(sf3,      modedir);
    strcpy(sf4,      modedir);
    strcpy(sf5,      modedir);
    strcpy(sf6,      modedir);
    strcpy(sf7,      modedir);
    strcpy(sf8,      modedir);
    strcpy(sf9,      modedir);
    strcpy(sf10,     modedir);
    strcpy(sac_fname,datadir);

    printf("strings cleared\n\n");
    */
  //}

  for(i = 0; i < params.nsta; i++){
    free(params.sta[i]);
  }
  free(params.sta);

  free(conf_file);

  printf("Exit success\n");
  return(info);
}

