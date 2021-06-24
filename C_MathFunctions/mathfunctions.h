#include <vector>
double azim_integ(int m1);
double two_azim_integ(int m1,int m2);
double three_azim_integ(int m1,int m2,int m3);

void Jint_sort_indices(int* l1,int* l2,int* l3,int* m1,int* m2,int* m3);
double Jint_signflip_renormalize(int l1,int l2,int l3,int* m1,int* m2,int* m3);
double ALP_normalize(int l,int m);
double Jint_normalize(int l1,int l2,int l3,int m1,int m2,int m3);
bool Jint_special_cases(int l1,int l2,int l3,int m1,int m2,int m3,double* result);
double ALP_integral(int l,int m);
double two_ALP_integral(int l1,int l2,int m1,int m2);
double three_ALP_J_integral(int l1,int l2,int l3,int m1,int m2,int m3);
double three_Ylm_integ(int l1,int l2,int l3,int m1,int m2,int m3);


double J_int_m2(int l1,int l2,int l3,int m1,int m2,int m3);
double J_int_m1(int l1,int l2,int l3,int m1,int m2,int m3);
double J_int_p1(int l1,int l2,int l3,int m1,int m2,int m3);
double J_int_m1_D(int l1,int l2,int l3,int m1,int m2,int m3);
double J_int_p1_D(int l1,int l2,int l3,int m1,int m2,int m3);

double I_m1_integral(int m1,int m2,int m3);
double I_p1_integral(int m1,int m2,int m3);
double I_m1_D_integral(int m1,int m2,int m3);
double I_p1_D_integral(int m1,int m2,int m3);

double prim_radial_ovlp(unsigned int la,unsigned int lb,unsigned int l,double zet_a,double zet_b,double r);
double prim_ovlp(std::vector<double> ra,std::vector<double> rb,double zeta_a,double zeta_b,unsigned int la,unsigned int lb,int ma,int mb);
void ao_ovlp(std::vector<double> ra,std::vector<double> rb,std::vector<int> nuc_bas_fun_a,std::vector<int> nuc_bas_fun_b,std::vector<int> cont_num_a,std::vector<int> cont_num_b,std::vector<double> zet_a,std::vector<double> zet_b,std::vector<double> cont_coeff_a, std::vector<double> cont_coeff_b,std::vector<unsigned int> la,std::vector<unsigned int> lb,std::vector<int> ma,std::vector<int> mb,std::vector<double>* S);
void MO_ovlp(std::vector<double> S,std::vector<double> lcao_a,std::vector<double> lcao_b,std::vector<double>* MO_S);
void slater_ovlp(int n_elec,int n_csf_a,int n_csf_b,std::vector<int> csf_mo_a,std::vector<int> csf_mo_b,std::vector<int> csf_spin_a,std::vector<int> csf_spin_b,int n_occ_a,int n_occ_b,std::vector<double> MO_S,std::vector<double>* CSF_S);
void ES_ovlp(std::vector<double> CSF_S,int n_csf_a,int n_csf_b,std::vector<double> ci_vector_a,std::vector<double> ci_vector_b,int n_states_a,int n_states_b,std::vector<double>* ES_S);


void prim_trdip_centered(std::vector<double> ra,std::vector<double> rb,double zeta_a,double zeta_b,unsigned int la,unsigned int lb,int ma,int mb,std::vector<double>* trdip);
double CY_m_sum(double thet,double phi,unsigned int la,unsigned int lb,unsigned int l,int ma,int mb);


double determinant(double *A,int dim);
void transpose(double *A,double *B, int dim1, int dim2);
void matrix_product(double *C,double *A,double *B,const int dim1,const int dim2,const int dim3);
double prefactor_rYlm(int l,int m);
double rYlm (int l,int m,double thet,double phi);

void compute_bessel_pice_mo(double*** pice_ortho_mo,double*** pice_ddx_mo,double*** pice_ddy_mo,double*** pice_ddz_mo,int jl_max,int n_occ,int basis_size,int nk,double kmax,double *MO_coeff_neutral,double **contraction_zeta,double **contraction_coeff,int * contraction_number,double** nucl_spher_pos,int *nucl_basis_func,int** angular_mom_numbers);
double clebsch_gordan_coeff(unsigned int l1,unsigned int l2,unsigned int l3,int m1,int m2,int m3);
void B_coeff(int l,int m1,int m2,std::vector<double> *B_val);
double gen_I_integ(unsigned int l1,unsigned int l2,double zeta,double k);

void square_root(double  *Rez, double *Imz, int len);
void cart_to_spher(double* x,double* y,double* z,double * r,double* t,double *f,int length);
void matrix_product(double *result_matrix,double *first_matrix,double *second_matrix,int dim1,int dim2,int dim3);
void transpose(double *input,double *transposed, int n_rows_input, int n_col_input);
unsigned long long int factorial(int n);
bool kronecker_delta(int a, int b);
bool two_cubes_moment(double *cube1,double *cube2,double *moment,int nx,int ny,int nz,double xmin,double xmax,double ymin,double ymax,double zmin,double zmax);
double cube_dot_product(double *cube1,double *cube2,int nx,int ny, int nz,double dx,double dy,double dz,int angle_vec_size,double *output);
double vector_prod(double vector1[],double vector2[],int gsize);
long double intplushalf_gamma(int n); //(Gamma(n+1/2))
long double gamma_int_or_half(double z);
double j_l(int l,double z);//spherical bessel function of order l
double dj_ldz(int l,double z); //Derivative of the spherical bessel function of order l
int dfactorial(int n);
void fact_prime_decomposer(int N, int* N_prime);
void prime_decomposer(int N, int* N_prime);
int factorized_diff(int* x1,int* x2,int* out);
bool factorized_sum(int* x1,int* x2,int* out);