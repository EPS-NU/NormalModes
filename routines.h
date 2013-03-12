#define SAC_DATA_MAX 500000
#define BUFFER 256

struct config_params{
  int nsta;
  char** sta;
};

struct moment_tensor{
  double mrr;
  double mtt;
  double mpp;
  double mrt;
  double mrp;
  double mtp;
};

int next_2(const int );
void read_stick_file(const char*, int*, double**, complex**);
void get_config_params(const char*, struct config_params*);
void get_moment_components(const char*, struct moment_tensor*);
int exists(const char*);
void import_plan(const char*);
void save_plan(const char*);
double get_Q(const char*, const char*);
complex synt_fft(complex, double, double, double, double);
