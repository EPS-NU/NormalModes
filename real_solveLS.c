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

#include "routines.h"


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
 // Tlo = 1530;
  Tlo = 1/0.000655;
  Thi = 1/0.000635;

  wlo = (2.0*M_PI)/Thi;
  whi = (2.0*M_PI)/Tlo;

  double complex* x = NULL;

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
  double ***Gminor, **Gmajor, *G;
  double **dminor, *dmajor;
  lapack_int Gm, Gn, Gld, info, Dld, nrhs;


  qmode = get_Q("0s4", modedir);


  Gminor = (double***) LAPACKE_malloc(params.nsta * sizeof(double**));
  dminor = (double**) LAPACKE_malloc(sizeof(double*) * params.nsta);

    for(i = 0; i < params.nsta; i++){ 

//      printf("%s \n", params.sta[i]);


      /* read the stick files for current station */
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

      /* read the sac binary file for the current station */
      strcat(sac_fname,"12102.");
      strcat(sac_fname,params.sta[i]);

  //    printf("%s\n",sac_fname);


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

//      printf("end = %d\nwlo = %le whi = %le\n", end, whi/(2.0*M_PI), wlo/(2.0*M_PI));

      /* Allocate space for sub-block in G */
      m = (whi - wlo)/dw +1;
  //    printf("m = %d\n", m);


      /* times 2 for real and imaginary matrices */
      Gminor[i] = (double**) calloc(2 * m, sizeof(double*)); 
      for(j = 0; j < m*2; j++){
        Gminor[i][j] = (double*) calloc(10, sizeof(double) );
      }


      dminor[i] = (double*) LAPACKE_malloc(sizeof(double) * m * 2);
      for(j = 0; j < m; j++){
        dminor[i][j] = creal(fft_data[start+j]);
      }

      for(j = m; j < m*2; j++){
        dminor[i][j] = cimag(fft_data[start+j-m]);
      }

//      printf("Filling block\n");
      /* begin filling block */

      Nxdt = len*delta;

      FILE* agmatrix = fopen("agmatrix.dat","w");
      omega = wlo;
      k = 0;
      while(omega < whi && k < m){

        for( j = 0; j < nsing1; j++){

          /* first earthquake */
          Gminor[i][k][0] +=  creal(synt_fft( z1[j], w1[j],omega,qmode,Nxdt));
          Gminor[i][k][1] +=  creal(synt_fft( z2[j], w2[j],omega,qmode,Nxdt));
          Gminor[i][k][2] +=  creal(synt_fft( z3[j], w3[j],omega,qmode,Nxdt));
          Gminor[i][k][3] -=  creal(synt_fft( z4[j], w4[j],omega,qmode,Nxdt));
          Gminor[i][k][4] -=  creal(synt_fft( z5[j], w5[j],omega,qmode,Nxdt));

          /* add second earthquake */
          Gminor[i][k][5] +=  creal(synt_fft( z6[j], w6[j],omega,qmode,Nxdt));
          Gminor[i][k][6] +=  creal(synt_fft( z7[j], w7[j],omega,qmode,Nxdt));
          Gminor[i][k][7] +=  creal(synt_fft( z8[j], w8[j],omega,qmode,Nxdt));
          Gminor[i][k][8] -=  creal(synt_fft( z9[j], w9[j],omega,qmode,Nxdt));
          Gminor[i][k][9] -=  creal(synt_fft(z10[j],w10[j],omega,qmode,Nxdt));

        }


        // printf("k = %d, w = %le\n", k, omega);
        
         fprintf(agmatrix,"%le ", Gminor[i][k][0]);
         fprintf(agmatrix,"%le ", Gminor[i][k][1]);
         fprintf(agmatrix,"%le ", Gminor[i][k][2]);
         fprintf(agmatrix,"%le ", Gminor[i][k][3]);
         fprintf(agmatrix,"%le ", Gminor[i][k][4]);
         fprintf(agmatrix,"%le ", Gminor[i][k][5]);
         fprintf(agmatrix,"%le ", Gminor[i][k][6]);
         fprintf(agmatrix,"%le ", Gminor[i][k][7]);
         fprintf(agmatrix,"%le ", Gminor[i][k][8]);
         fprintf(agmatrix,"%le ",  Gminor[i][k][9]);
         fprintf(agmatrix,"\n");
       

        k++;
        omega = omega + dw;

      }

      omega = wlo;
      k = m;
      while(omega < whi && k < m*2){

        for( j = 0; j < nsing1; j++){

          /* first earthquake */
          Gminor[i][k][0] +=  cimag(synt_fft( z1[j], w1[j],omega,qmode,Nxdt));
          Gminor[i][k][1] +=  cimag(synt_fft( z2[j], w2[j],omega,qmode,Nxdt));
          Gminor[i][k][2] +=  cimag(synt_fft( z3[j], w3[j],omega,qmode,Nxdt));
          Gminor[i][k][3] -=  cimag(synt_fft( z4[j], w4[j],omega,qmode,Nxdt));
          Gminor[i][k][4] -=  cimag(synt_fft( z5[j], w5[j],omega,qmode,Nxdt));

          /* add second earthquake */
          Gminor[i][k][5] +=  cimag(synt_fft( z6[j], w6[j],omega,qmode,Nxdt));
          Gminor[i][k][6] +=  cimag(synt_fft( z7[j], w7[j],omega,qmode,Nxdt));
          Gminor[i][k][7] +=  cimag(synt_fft( z8[j], w8[j],omega,qmode,Nxdt));
          Gminor[i][k][8] -=  cimag(synt_fft( z9[j], w9[j],omega,qmode,Nxdt));
          Gminor[i][k][9] -=  cimag(synt_fft(z10[j],w10[j],omega,qmode,Nxdt));
        }


        // printf("k = %d, w = %le\n", k, omega);
        
         fprintf(agmatrix,"%le ", Gminor[i][k][0]);
         fprintf(agmatrix,"%le ", Gminor[i][k][1]);
         fprintf(agmatrix,"%le ", Gminor[i][k][2]);
         fprintf(agmatrix,"%le ", Gminor[i][k][3]);
         fprintf(agmatrix,"%le ", Gminor[i][k][4]);
         fprintf(agmatrix,"%le ", Gminor[i][k][5]);
         fprintf(agmatrix,"%le ", Gminor[i][k][6]);
         fprintf(agmatrix,"%le ", Gminor[i][k][7]);
         fprintf(agmatrix,"%le ", Gminor[i][k][8]);
         fprintf(agmatrix,"%le ",  Gminor[i][k][9]);
         fprintf(agmatrix,"\n");

        k++;
        omega = omega + dw;

      }


//      printf("outside of block\n");


      fclose(agmatrix);

//      printf("block filled\n");

      fftw_free(fft_data);
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


    G = (double*) LAPACKE_malloc(sizeof(double) * 2 * 10 * m * params.nsta);
    
    Gmajor = (double**) LAPACKE_malloc(sizeof(double*) * 2 * m * params.nsta);
    for(i = 0; i < 2 * m * params.nsta; i++){
      Gmajor[i] = (double*) LAPACKE_malloc(sizeof(double) * 10);
    }
  

//    printf("Gmajor allocated\n");

//    printf("forming Gmajor array\n");

    i = 0;
    for( i = 0; i < params.nsta; i++){
      for( j = 0; j < m * 2; j++){
        for(k = 0; k < 10; k++){
          Gmajor[j+i*m*2][k] = Gminor[i][j][k];
        }
      }
    }

//    printf("forming G array \n");
/* column major */
    for(j = 0; j < 10; j++){
      for(i = 0; i < m * params.nsta * 2; i++){
        G[i+j*m*params.nsta*2] = Gmajor[i][j];
      }
    }

    /* row major 
    for(i = 0; i < m * params.nsta * 2; i++){
      for(j = 0; j < 10; j++){
        G[i*10+j] = Gmajor[i][j];
      }
    }
*/

    // printf("params.nsta * m * 10 = %d, h = %d\n",params.nsta*m*10, h);

//    printf("printing G to file \n");
    
     FILE* gmatrix = fopen("gmatrix.dat","w");
     for(i = 0; i < params.nsta * m * 10 * 2; i++){
       fprintf(gmatrix,"%le\n", G[i]);
     }
     fclose(gmatrix);
  
//    printf("Gmajor done\n");

    dmajor = (double*) malloc(params.nsta*m*2* sizeof(double));
    for( i = 0; i < params.nsta ; i++){
      for( j = 0; j < m*2; j++){
        dmajor[i*m*2+j] = dminor[i][j];
      }
    }

    FILE* dmatrix = fopen("dmatrix.dat","w");
    for(i = 0; i < params.nsta * m * 2; i++){
      fprintf(dmatrix,"%le\n", dmajor[i]);
    }
    fclose(dmatrix);

    

    /* Attempt to directly do m = inv(G'G)*G'd 

    double* C = calloc(10*10, sizeof(double));
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 10, 10, params.nsta*m*2, 1.0, 
                G, params.nsta*m*2, G, params.nsta*m*2, 0.0, C, 10);

    printf("1\n");


    for(i = 0; i < 10; i++){
      for( j = 0; j < 10; j++){
        printf("%le ",C[j + i*10]);
      }
      printf("\n");
    }


    int* ipiv = calloc(10, sizeof(int));

    info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, 10,10, C, 10, ipiv); 

    printf("2\n");

    info = LAPACKE_dgetri(LAPACK_COL_MAJOR, 10, C, 10, ipiv);

    for(i = 0; i < 10; i++){
      for(j = 0; j < 10; j++){
        printf("%le ", C[j+i*10]);
      }
      printf("\n");
    }

    printf("3\n");

    double* D = calloc(params.nsta*m*2*10, sizeof(double));


    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 10, params.nsta*m*2, 10, 1.0,
                C, 10, G, params.nsta*m*2, 0.0, D, 10);

    printf("4\n");

    double* model = calloc(10, sizeof(double));

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 10, 1, params.nsta*m*2, 1.0,
                D, 10, dmajor, params.nsta*m*2, 0.0, model, 10);

    printf("5\n");

    for(i = 0; i < 10; i++){
      printf("%le\n",model[i]);
    }
    

    free(C);
    free(ipiv);
    free(D);
    free(model);

    */





















//    printf("dmajor done\n");

    Gm = m*params.nsta*2;
    Gn = 10;
    Gld = Gm;
    Dld = m*params.nsta*2;
    nrhs = 1;

    printf("doing least squares....\n");

    /* linear least squares using QR or LQ, solves problem based on assumption
     * that rank(G) = min(Gm,Gn), that G is not rank deficient */
//    info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', Gm, Gn, nrhs, G, Gld, dmajor, Dld); 

    /* linear least squares using SVD of G, which allows for possibility that
     * G is rank deficient. This may be better to use. Need to declare arrays
     * holding singular values of G, and a work array */

    /* S: array holding singular values in decreasing order, dimension min(Gm,Gn) */
    double* S = calloc(Gn, sizeof(double));

    /* rcond: used to determine effective rank of G
     * if rcond < 0, machine precision is used */
    double rcond = 10e-3;
    
    /* rank: effective rank of G, number of singular values greater than rcond*s(1) */
    int rank;
 
    info = LAPACKE_dgelsd(LAPACK_COL_MAJOR, Gm, Gn, nrhs, G, Gld, dmajor, Dld, S, rcond, &rank); 
//    info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, Gm, Gn, nrhs, G, Gld, dmajor, Dld, S, rcond, &rank); 
    if(info < 0){
      printf("%d'th argument had illegal value\n",info);
      exit(EXIT_FAILURE);
    }
    else if(info > 0){
      printf("SVD algorithm failed to converge\n");
      printf("%d off-diagonal elements of an intermediate bidiagonal form did not converge to zero\n",info);
      exit(EXIT_FAILURE);
    }

    /* end linear least squares inversion */

//    printf("                      .........least squares done\n");

//    printf(" Print out answer\n");
/*
    for( i = 0; i < Gn; i++){
      for( j = 0; j < 1; j++){
        fprintf(stdout, "%lE ", dmajor[i+Gm*j]);
      }
      printf("\n");
    }
*/
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

    FILE* moment_tensor_1_file = fopen("moment_tensor_1.dat", "w");
    fprintf(moment_tensor_1_file,"%le\n",mrr1);
    fprintf(moment_tensor_1_file,"%le\n",mtt1);
    fprintf(moment_tensor_1_file,"%le\n",mpp1);
    fprintf(moment_tensor_1_file,"%le\n",mrt1);
    fprintf(moment_tensor_1_file,"%le\n",mrp1);
    fprintf(moment_tensor_1_file,"%le\n",mtp1);
    fclose(moment_tensor_1_file);

    FILE* moment_tensor_2_file = fopen("moment_tensor_2.dat", "w");
    fprintf(moment_tensor_2_file,"%le\n",mrr2);
    fprintf(moment_tensor_2_file,"%le\n",mtt2);
    fprintf(moment_tensor_2_file,"%le\n",mpp2);
    fprintf(moment_tensor_2_file,"%le\n",mrt2);
    fprintf(moment_tensor_2_file,"%le\n",mrp2);
    fprintf(moment_tensor_2_file,"%le\n",mtp2);
    fclose(moment_tensor_2_file);


    printf("\nEvent 1:\n");
    printf(" Mrr = %le, Mtt = %le, Mpp = %le\n Mrt = %le, Mrp = %le, Mtp = %le\n",
           mrr1, mtt1, mpp1, mrt1, mrp1, mtp1);

    printf("Event 2:\n");
    printf(" Mrr = %le, Mtt = %le, Mpp = %le\n Mrt = %le, Mrp = %le, Mtp = %le\n",
           mrr2, mtt2, mpp2, mrt2, mrp2, mtp2);

    printf("\nSingular values of G:\n");
    for(i = 0; i < rank; i++){
      printf("\tS[%d] = %le\n",i,S[i]);
    }
    printf("\n");

    /*  deallocate memory */

    fftw_destroy_plan(data_plan);


//    printf("plan destroyed\n");

    free(S);

    for(i = 0; i < params.nsta; i++){
      LAPACKE_free(dminor[i]);
    }
    LAPACKE_free(dminor);
//    printf("step 1\n");


    LAPACKE_free(dmajor);

//    printf("step 2....freed data\n");

    for(i = 0; i < params.nsta; i++){
      for(j = 0; j < m; j++){
        LAPACKE_free(Gminor[i][j]);
      }
      LAPACKE_free(Gminor[i]);
    }

//    printf("step 1\n");


    LAPACKE_free(Gminor);
//    printf("step 2...Gminor freed\n");


    for(i = 0; i < 10; i++){
      LAPACKE_free(Gmajor[i]);
    }
    LAPACKE_free(Gmajor);

    LAPACKE_free(G);

//    printf("freed g\n");
    

    free(x);

  for(i = 0; i < params.nsta; i++){
    free(params.sta[i]);
  }
  free(params.sta);

  free(conf_file);

//  printf("Exit success\n");
  return(info);
}

