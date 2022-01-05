//
//  computation.cpp
//  Dyson_computation
//
//  Created by Stephan van den Wildenberg on 20/12/16.
//  Copyright © 2016 Stephan van den Wildenberg. All rights reserved.
//

#include "prime.hpp"
#include <cstdlib>
#include <iostream>
#include <complex>
#include <cmath>
/*
def square_root(z):
    "Square root with different branch cut defined by alpha parameter."
    theta = np.pi/2
    argument = np.angle(z) # between -pi and +pi
    modulus = np.abs(z)
    argument = np.mod(argument + theta, 2 * np.pi) - theta
    return np.sqrt(modulus) * np.exp(1j * argument / 2)
*/
double double_modulus(double a, double b)
{
    double c = a;
    while( c > b )
    {
       c -= c;
    }
    return c;
}
void square_root(double  *Rez,double *Imz,int len)
{ //Square root with different branch cut defined by alpha parameter.

    double pi = acos(-1);
    for(int i = 0; i < len ; i++)
    {
        std::complex<double> z = std::complex<double>(Rez[i],Imz[i]);
        double theta = pi/2;
        double argument = std::arg(z); // between -pi and +pi
        double modulus = std::abs(z);
        argument = double_modulus(argument + theta, 2 * pi) - theta;
        Rez[i] = sqrt(modulus) * cos(argument/2);
        Imz[i] = sqrt(modulus) * sin(argument/2);
    }
}

void cart_to_spher(double* x,double* y,double* z,double * r,double* t,double *f,int length)
{
   int i=0;
   for( i = 0 ; i != length ; i++)
   {
      r[i]=sqrt(x[i]*x[i]+y[i]*y[i]+z[i]*z[i]);

      if(r[i]==0)
      {
         t[i]=0;
         f[i]=0;
      }
      else
      {
         t[i]=acos(z[i]/r[i]);
         if(x[i] == 0 && y[i] > 0)
         {
            f[i]=acos(-1)/2.;
         }
         else if (x[i] == 0 && y[i] < 0 )
         {
            f[i]=3.*acos(-1)/2.;
         }
         else
         {
            f[i]=atan2(y[i],x[i]);
         }
      }
      if(f[i] < 0)
         f[i]+=2*acos(-1);
   }
}
void matrix_product(double *C,double *A,double *B,int dim1,int dim2,int dim3)
{
   //C=A*B
    double ntemp;
    for (int i=0; i!=dim1; i++)
    {
        for (int j=0; j!=dim3; j++)
        {
            ntemp=0;

            for (int k=0; k!=dim2; k++)
            {
                ntemp+=A[i*dim2+k]*B[k*dim3+j];
            }
            C[i*dim3+j]=ntemp;
        }
    }

}
void transpose(double *A,double *B, int dim1, int dim2)
{
   //B=trans(A)
    for (int i=0; i!=dim1; i++)
    {
        for (int j=0; j!=dim2; j++)
        {
            B[j*dim1+i]=A[i*dim2+j];
        }
    }
}
void fact_prime_decomposer(int N, int* N_prime)
{
   //This routine factorizes the factorial of an integer number into prime numbers. It uses a global and constant array with maximum integer MAX_N_FACTORIAL
   if(N>MAX_N_FACTORIAL)
   {
      std::cout<<"ERROR ! FACTORIAL ARGUMENT LARGER THAN MAX AUTHORIZED VALUE ! N = "<<N<<std::endl;
      exit(EXIT_SUCCESS);
   }
   else
   {
      for(int i=0;i!=MAX_FACTORIAL_PRIME;i++)
      {
         N_prime[i]=PRIME_DECOMPOSED_FAC[N][i];
      }
   }
   return;
}

void prime_decomposer(int N, int* N_prime)
{
   //This routine factorizes an integer number into prime numbers.
   for(int i=0;i!=MAX_FACTORIAL_PRIME;i++)
      N_prime[i]=0;

   if(N==0)
   {
      std::cout<<"WARNING ! TRYING TO DECOMPOSE ZERO IN PRIME NUMBERS"<<std::endl;
      return;
   }
   for(int i=0;i!=MAX_FACTORIAL_PRIME;i++)
   {
      while( N%PRIME[i] == 0 )
      {
         N/=PRIME[i];
         N_prime[i]++;
      }
   }
   if(N!=1)
   {
      std::cout<<" INCOMPLETE FACTORIZATION. Remaining factor : "<<N<<std::endl;
   }
   return;
}
bool factorized_sum(int* x1,int* x2,int* out)
{
   //!!!!APPLIES ONLY TO FACTORIALS OF NUMBERS < MAX_N_FACTORIAL
   //We want to compute the factorized representation of a sum of two integers
   //1. we factorize the sum by comparing the vectors
   //2. We explicitly compute the remaining sum
   //3. we factorize the remaining sum and put everythingout

   int remain1(1);
   int remain2(1);
   int remain[MAX_FACTORIAL_PRIME];

   for(int i=0;i!=MAX_FACTORIAL_PRIME;i++)
   {
      out[i]=0;
      while(x1[i] != 0 && x2[i]!= 0)
      {
         out[i]++;
         x1[i]--;
         x2[i]--;
      }
      remain1*=pow(PRIME[i],x1[i]);
      remain2*=pow(PRIME[i],x2[i]);
   }
   prime_decomposer(remain1+remain2,remain);
   for(int i=0;i!=MAX_FACTORIAL_PRIME;i++)
   {
      out[i]+=remain[i];
   }
   return 0;
}
int factorized_diff(int* x1,int* x2,int* out)
{
   //!!!!APPLIES ONLY TO FACTORIALS OF NUMBERS < MAX_N_FACTORIAL
   //We want to compute the factorized representation of a difference of two integers x1-x2
   //1. we factorize the sum by comparing the vectors
   //2. We explicitly compute the remaining difference
   //3. we factorize the remaining difference and keep track of the sign
   //4. We put the resulting integer in the out array and the sign as a return

   int sign(1);
   int remain1(1);
   int remain2(1);
   int remain[MAX_FACTORIAL_PRIME];

   for(int i=0;i!=MAX_FACTORIAL_PRIME;i++)
   {
      out[i]=0;
      while(x1[i] != 0 && x2[i]!= 0)
      {
         out[i]++;
         x1[i]--;
         x2[i]--;
      }
      remain1*=pow(PRIME[i],x1[i]);
      remain2*=pow(PRIME[i],x2[i]);
   }
   //check the sign of the difference and factorize the difference
   if(remain1 == remain2)
      return 0;
   else if(remain1>remain2)
      prime_decomposer(remain1-remain2,remain);
   else
   {
      prime_decomposer(remain2-remain1,remain);
      sign=-1;
   }
   for(int i=0;i!=MAX_FACTORIAL_PRIME;i++)
   {
      out[i]+=remain[i];
   }
   return sign;
}
unsigned long long int factorial(int n)
{
   if(n>MAX_N_FACTORIAL)
   {
      std::cout<<"WARNING LARGE FACTORIAL ARGUMENT : N ="<<n<<std::endl<<"EXIT"<<std::endl;
      exit(EXIT_SUCCESS);
   }
   else if(n<0)
   {
      std::cout<<"FATAL ERROR! NEGATIVE ARGUMENT IN FACTORIAL"<<std::endl<<"N = "<<n<<std::endl<<"EXIT"<<std::endl;
      exit(EXIT_SUCCESS);
   }
   else
   {
      if (n==1 || n==0)
         return 1;
      else
         return n*factorial(n-1);
   }
}
int dfactorial(int n)
{
   if(n<=1)
      return 1;
   else
      return n*dfactorial(n-2);
}
long double intplushalf_gamma(int n) //(Gamma(n+1/2))
{
   int fac1[MAX_FACTORIAL_PRIME];
   int fac2[MAX_FACTORIAL_PRIME];

   fact_prime_decomposer(2*n,fac1);
   fact_prime_decomposer(n,fac2);
   fac1[2]-=2*n;

   double temp=1;
   for(int i=0;i!=MAX_FACTORIAL_PRIME;i++)
   {
      temp*=pow(PRIME[i],fac1[i]-fac2[i]);
   }
   return sqrt(acos(-1))*temp;
}

long double gamma_int_or_half(double z)
{
   int fac1[MAX_FACTORIAL_PRIME];

   if(ceil(z)==floor(z) && z > 0) // if the argument is integer and positive
   {
      fact_prime_decomposer(int(z)-1,fac1); // Gamma(n+1) = n!
      long double temp=1;
      for(int i=0;i!=MAX_FACTORIAL_PRIME;i++)
      {
         temp*=pow(PRIME[i],fac1[i]);
      }
      return temp; // return (n-1)!
   }

   else if(ceil(2*z) == floor(2*z) && z > 0) // if the argument is half integer and positive
      return intplushalf_gamma(int(z-0.5)); // Gamma( N / 2 ) = Gamma( (N / 2) - 1/2 + 1/2 ) = Gamma( N' + 1/2)

   else
   {
      std::cout<<"ERROR ! NON HALF INTEGER OR NON INTEGER OR NON POSITIVE ARGUMENT IN GAMMA_INT_OR_HALF FUCNTION"<<std::endl;
      std::cout<<" Z = "<<z<<std::endl;
      exit(EXIT_SUCCESS);
   }
}

double vector_prod(double vector1[],double vector2[],int gsize)
{
    double sum(0.);
#pragma omp parallel for
        for (int j=0; j<gsize; j++)
        {
            sum+=vector1[j]*vector2[j];
        }

    return sum;
}

bool kronecker_delta(int a, int b)
{
    if (a==b)
    {
        return 1;
    }
    else
        return 0;
}
double cube_dot_product(double *cube1,double *cube2,int nx,int ny, int nz,double dx,double dy,double dz,int angle_vec_size,double *output)
{
   double sum(0);
   int num(nx*ny*nz);
   int inc(1);

   #pragma omp parallel for
   for(int i=0;i<angle_vec_size;i++)
   {
      output[i]=0;
      sum=0;
      for(int j=0;j<nx*ny*nz;j++)
      {
         sum+=cube1[i*nx*ny*nz+j]*cube2[j];
      }
      output[i]=sum*dx*dy*dz;
   }
   return 0;
}
double j_l(int l,double z) //spherical bessel function of order l
{
   double test(1);
   double val(0);
   double valm1(0);
   double factor(1);
   int i(0);
   int fac1[MAX_FACTORIAL_PRIME];
   double temp(1);


   if(z==0)
   {
      if(l==0)
         return 1;
      else
         return 0;
   }
   else
   {
        if(l<-1)
            return 0;
        else if(l == -1)
           return cos(z)/z;
        else
        {
            val=1;
            i=1;
            while(test>=1e-15)
            {
               fact_prime_decomposer(i,fac1);
               temp=1;
               for(int ii=0;ii!=MAX_FACTORIAL_PRIME;ii++)
               {
                  temp*=pow(PRIME[ii],fac1[ii]);
               }
                valm1=val;
                factor=1;
                for(int k=1;k!=i+1;k++) factor*=(2*k+2*l+1);
                val+=pow(-1,i)*pow(z*z/2.,i)/(factor*temp);
                i++;
                test=fabs((val-valm1));
            }

            if(isnan(val))
               std::cout<<" ERROR ! BESSEL FUNCTION IS NAN"<<std::endl;

            return val*pow(z,l)/dfactorial(2*l+1);
        }
   }
}

double dj_ldz(int l,double z) //Derivative of the spherical bessel function of order l
{
   if(z==0 || l == 0)
      return 0;
   else
   {
      if(isnan(l*j_l(l-1,z)-(l+1)*j_l(l+1,z))/(2*l+1))
          std::cout<<" ERROR ! BESSEL DERIVATIVE FUNCTION IS NAN"<<std::endl;
      return (l*j_l(l-1,z)-(l+1)*j_l(l+1,z))/(2*l+1);
   }
}
