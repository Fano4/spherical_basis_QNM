#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <gsl/gsl_sf.h>
#include "prime.hpp"
#include "mathfunctions.h"

double azim_integ(int m)
{
   if( m != 0 )
      return 0;
   else
      return 2*acos(-1);
}
double two_azim_integ(int m1,int m2)
{
   //This function always returns 0 if (m1+m2)%2!=0
   //
   if( m1 == 0 && m2 == 0 ) // m1+m2 even
      return 2*acos(-1);

   else if ( ( m1 == 0 && m2 != 0 ) || ( m1 != 0 && m2 == 0 ) ) // integ !=0 iff m1+m2 is even
      return azim_integ(m1+m2);

   else if( m1 == m2 ) // m1+m2 even
      return acos(-1);
   
   else if(m1 == -m2 ) //m1+m2 even .. 
      return 0;

   else if( m1 > 0 && m2 > 0 )
      return 0.5*(azim_integ(m1+m2)+azim_integ(abs(m1-m2))); //// integ !=0 iff m1+m2 is even

   else if( m1 > 0 && m2 < 0 )
      return 0.5*( azim_integ(-abs(m1+m2)) +pow(-1,bool(m1+m2<0))*azim_integ(-abs(m1-m2))); //// integ !=0 iff m1+m2 is even

   else if( m1 < 0 && m2 > 0 )
      return 0.5*( azim_integ(-abs(m1+m2)) + pow(-1,bool(m1+m2<0))*azim_integ(-abs(m1-m2))); // integ !=0 iff m1+m2 is even
   else 
      return 0.5*( azim_integ(abs(m1-m2)) - azim_integ(abs(m1+m2))) ; // integ !=0 iff m1+m2 is even
}
double three_azim_integ(int m1,int m2,int m3)
{
   //This function always returns 0 if (m1+m2+m3)%2!=0
   if( m1 == 0 && m2 == 0 )
      return azim_integ(m3); // returns 0 if m3 !=0

   else if (m3 == 0)
      return two_azim_integ(m1,m2); //returns 0 if (m1+m2)%2!=0 

   else if ( ( m1 == 0 && m2 != 0 ) || ( m1 != 0 && m2 == 0 ) )
      return two_azim_integ(m1+m2,m3); //integ !=0 if (m1+m2+m3) is even

   else if( m1 > 0 && m2 > 0 )
      return 0.5*(two_azim_integ(m1+m2,m3)+two_azim_integ(abs(m1-m2),m3)); ///integ !=0 if (m1+m2+m3) is even

   else if( m1 > 0 && m2 < 0 )
      return -0.5*( pow(-1,bool(m1+m2<0))*two_azim_integ(-abs(m1+m2),m3) - pow(-1,bool(m1-m2<0))*two_azim_integ(-abs(m1-m2),m3)); //integ !=0 if (m1+m2+m3) is even

   else if( m1 < 0 && m2 > 0 )
      return -0.5*( pow(-1,bool(m1+m2<0))*two_azim_integ(-abs(m1+m2),m3) + pow(-1,bool(m1-m2<0))*two_azim_integ(-abs(m1-m2),m3));//integ !=0 if (m1+m2+m3) is even

   else 
      return 0.5*( two_azim_integ(abs(m1-m2),m3) - two_azim_integ(abs(m1+m2),m3)) ;//integ !=0 if (m1+m2+m3) is even
}
void Jint_sort_indices(int* l1,int* l2,int* l3,int* m1,int* m2,int* m3)
{
   //rearrange the array for getting l1<l2<l3
   int tmpl;
   int tmpm;
   if(*l1 > *l2)
   {
      tmpl=*l1;
      tmpm=*m1;
      *l1=*l2;
      *m1=*m2;
      *l2=tmpl;
      *m2=tmpm;
   }
   if(*l1 > *l3)
   {
      tmpl=*l1;
      tmpm=*m1;
      *l1=*l3;
      *m1=*m3;
      *l3=tmpl;
      *m3=tmpm;
   }
   if(*l2 > *l3)
   {
      tmpl=*l2;
      tmpm=*m2;
      *l2=*l3;
      *m2=*m3;
      *l3=tmpl;
      *m3=tmpm;
   }

   return;
}
double Jint_signflip_renormalize(int l1,int l2,int l3,int* m1,int* m2,int* m3)
{

   bool sgnm1( bool ( *m1 < 0 ) );
   bool sgnm2( bool ( *m2 < 0 ) );
   bool sgnm3( bool ( *m3 < 0 ) );
   double temp;
   double val(1);

   //Flip the sign of m's if they are negative
   if(sgnm1)
      *m1=-*m1;

   if(sgnm2)
      *m2=-*m2;

   if(sgnm3)
      *m3=-*m3;

   if(sgnm1)
   {
      temp=1;
      for (int tt=l1-*m1+1;tt!=l1+*m1+1;tt++)
         temp*=double(tt);
      val/=temp;
   }

   if(sgnm2)
   {
      temp=1;
      for (int tt=l2-*m2+1;tt!=l2+*m2+1;tt++)
         temp*=double(tt);
      val/=temp;
   }

   if(sgnm3)
   {
      temp=1;
      for (int tt=l3-*m3+1;tt!=l3+*m3+1;tt++)
          temp*=double(tt);
      val/=temp;
   }

   return pow(-1,sgnm1**m1+sgnm2**m2+sgnm3**m3)*val;
//     return val;

}
double ALP_normalize(int l,int m)
{
   double temp;
   double val(1);
      
   if(l==0)
      return 1;

   temp=1;
   for (int tt=l-m+1;tt!=l+m+1;tt++)
   {
      temp*=double(tt);
   }
   val=sqrt(temp);

   return val;
}
double Jint_normalize(int l1,int l2,int l3,int m1,int m2,int m3)
{
   double temp;
   double val(1);

      temp=1;
      for (int tt=l1-m1+1;tt!=l1+m1+1;tt++)
      {
         temp*=double(tt);
      }
      val*=sqrt(temp);

      temp=1;
      for (int tt=l2-m2+1;tt!=l2+m2+1;tt++)
      {
         temp*=double(tt);
      }
      val*=sqrt(temp);

      temp=1;
      for (int tt=l3-m3+1;tt!=l3+m3+1;tt++)
      {
         temp*=double(tt);
      }
      val*=sqrt(temp);

      temp=val;
      return temp;
}
bool Jint_special_cases(int l1,int l2,int l3,int m1,int m2,int m3,double* result)
{
   int delta(m2+(m1-m2)*bool(m2>=m1));
   double prefactor(0);

   // Checking if any of the polynomials is zero
   
   if(l1 <0 || l2<0 || l3<0) //Negative degree should not occur
   {
      *result=0;
      return 1;
   }
   else if(m1>l1 || m2>l2 || m3>l3) // These ensure the degree of the polynomial is at least zero
   {
      *result=0;
      return 1;
   }

   else if( (l1+l2+l3) % 2 != (m1+m2+m3) % 2 ) // The integrand should be even,meaning that the sums of orders and degree have same parity
   {
      *result=0;
      return 1;
   }

   //If none of the polynomials is zero and the integrand is even,
   //
   // Sorting, sign flips and normalization
   
   Jint_sort_indices(&l1,&l2,&l3,&m1,&m2,&m3); // rearrange to get l1 < l2 < l3
   prefactor=Jint_signflip_renormalize(l1,l2,l3,&m1,&m2,&m3); // Flip the sign of negative m's and renormalize accordingly
   prefactor*=Jint_normalize(l1,l2,l3,m1,m2,m3); // Normalize the product of ALPs
   
   //Then check for special cases

   if(m1+m2==m3)
   {
      *result = 2. * pow(-1,m3) * prefactor * gsl_sf_coupling_3j(2*l1,2*l2,2*l3,0,0,0)* gsl_sf_coupling_3j(2*l1,2*l2,2*l3,2*m1,2*m2,-2*m3);
      return 1;
   }

   else if(abs(m1+m2)==m3)
   {
      *result = 2.*pow(-1,delta-m1+m2)*prefactor* gsl_sf_coupling_3j(2*l1,2*l2,2*l3,0,0,0) * gsl_sf_coupling_3j(2*l1,2*l2,2*l3,-2*m1,2*m2,2*m1-2*m2);
      return 1;
   }
   else
      return 0;

}
double ALP_integral(int l,int m)
{

   if( (l+m) % 2 != 0)//Check for parity
      return 0;

   else if( l == 0 ) //Check for zero degree
      return 2;
   else
   {
      return (pow(double(-1),m)+pow(double(-1),l))
         *pow(2.,double(m-2))*double(m)
         *std::tgamma(double(l)/2.)
         *std::tgamma(double(l+m+1)/2.)
         /(std::tgamma(double(l-m)/2.+1)*std::tgamma(double(l+3)/2.));
   }
}

double two_ALP_integral(int l1,int l2,int m1,int m2) //integral over two ALP
{
   int m12(m1+m2);
   double G12(0);
   double sum(0);

   double prefactor(0);

   prefactor=ALP_normalize(l1,m1)*ALP_normalize(l2,m2);
   sum=0;

   for(int l12=abs(l1-l2);l12!=l1+l2+1;l12++)
   {
      if( (l1+l2+l12) % 2 != 0 || m12 > l12) //Here, the selection rules for Wigner 3J symbols apply on each term
         continue;
      else
      {
         G12=pow(-1.,m12)*(2.*l12+1.)*gsl_sf_coupling_3j(2*l1,2*l2,2*l12,0,0,0)*gsl_sf_coupling_3j(2*l1,2*l2,2*l12,2*m1,2*m2,-2*m12)/ALP_normalize(l12,m12);
         sum+=G12*ALP_integral(l12,m12);
      }
   }
   return sum*prefactor;
}
double three_ALP_J_integral(int l1,int l2,int l3,int m1,int m2,int m3)
{
   double temp;
   double sum(0);
   double G12(0);
   double prefactor(1);

   // Check for zero integrand and for special cases
   if( Jint_special_cases(l1,l2,l3,m1,m2,m3,&temp) ) 
      return temp;

   // If the integrand is not special, compute the expansion of the Legendre polynomials product

   // Sorting, sign flips and normalization
   Jint_sort_indices(&l1,&l2,&l3,&m1,&m2,&m3); // rearrange to get l1 < l2 < l3
   prefactor=Jint_signflip_renormalize(l1,l2,l3,&m1,&m2,&m3); // Flip the sign of negative m's and renormalize accordingly
   prefactor*=ALP_normalize(l1,m1)*ALP_normalize(l2,m2);

   //Declare intermediary m's   
   int m12(m1+m2);

   sum=0;
   for(int l12=abs(l1-l2);l12!=l1+l2+1;l12++)
   {
      if( (l1+l2+l12) % 2 !=0 || m12 > l12) //Here, the selection rules for Wigner 3J symbols apply on each term
         continue;
      else
      {

         G12=pow(-1.,m12)*(2.*l12+1.)*gsl_sf_coupling_3j(2*l1,2*l2,2*l12,0,0,0)*gsl_sf_coupling_3j(2*l1,2*l2,2*l12,2*m1,2*m2,-2*m12)/ALP_normalize(l12,m12);

         sum+=G12*two_ALP_integral(l12,l3,m12,m3);
      }
   }
   return prefactor*sum;
}
double I_m1_integral(int m1,int m2,int m3)
{

//   return 0.5*(three_azim_integ(-m1-1,m2,m3)-three_azim_integ(-m1+1,m2,m3));

   if(m1<0)
      return 0.5*(three_azim_integ(fabs(1+m1),m2,m3)-three_azim_integ(fabs(1-m1),m2,m3));
   else
   {
      if(m1 == 1)
         return 0.5*(three_azim_integ(-2,m2,m3));
      else
         return 0.5*(three_azim_integ(-fabs(m1+1),m2,m3)+pow(-1,bool(1-m1<0))*three_azim_integ(-fabs(1-m1),m2,m3));
   }

}
double I_p1_integral(int m1,int m2,int m3)
{
   if(m1<0)
   {
      if(fabs(m1) == 1)
          return 0.5*(three_azim_integ(-2,m2,m3));
      else
          return 0.5*(three_azim_integ(-fabs(-m1+1),m2,m3)-pow(-1,bool(m1+1<0))*three_azim_integ(-fabs(m1+1),m2,m3));
   }
   else
      return 0.5*(three_azim_integ(fabs(m1+1),m2,m3)+three_azim_integ(fabs(-m1+1),m2,m3));

}
double I_m1_D_integral(int m1,int m2,int m3)
{
   return -m2*I_m1_integral(m1,-m2,m3)-m3*I_m1_integral(m1,m2,-m3);
}
double I_p1_D_integral(int m1,int m2,int m3)
{
   return -m2*I_m1_integral(m1,-m2,m3)-m3*I_m1_integral(m1,m2,-m3);
}
double J_int_m2(int l1,int l2,int l3,int m1,int m2,int m3)
{
   // CHECKED ON MARCH 2 2020
   double sum(0);


   if(l1==0 && l2 == 0 && l3 == 0 && m1 == 0 && m2 == 0 && m3 == 0 )
      return (acos(-1));

   else if(l1<0 || l2<0 || l3<0 || m1>l1 || m2>l2 || m3>l3)
      return 0;

   else if(l1>0)
   {
      if(l1 == m1)
         return -((2*l1-1)*three_ALP_J_integral(l1-1,l2,l3,m1-1,m2,m3));

      else if(l1 == m1+1 && l1>=2)
         return -((2*m1+1)*three_ALP_J_integral(m1,l2,l3,m1-1,m2,m3));

      else if(l1==1 && m1==0)
      {
         return ((1./(2.*l2+1.))*((l2-m2+1)*J_int_m2(0,l2+1,l3,0,m2,m3)+(l2+m2)*J_int_m2(0,l2-1,l3,0,m2,m3)));
      }
      if(l1>=m1+2)
         return ((1./(double(l1-m1)*double(l1-m1-1)))*(double(2*l1-1)*three_ALP_J_integral(l1-1,l2,l3,m1+1,m2,m3)
               +double(l1+m1)*(l1+m1-1)*J_int_m2(l1-2,l2,l3,m1,m2,m3)));
   }
   else if(l2>0)
   {
      if(l2 == m2)
         return -(2*l2-1)*three_ALP_J_integral(l1,l2-1,l3,m1,m2-1,m3);
      else if(l2 == m2+1 && l2>=2)
         return -(2*m2+1)*three_ALP_J_integral(l1,m2,l3,m1,m2-1,m3);
      else if(l2==1 && m2==0)
      {
         return (1./(2.*l3+1.))*((l3-m3+1)*J_int_m2(0,0,l3+1,0,0,m3)+(l3+m3)*J_int_m2(0,0,l3-1,0,0,m3));
      }

      if(l2>=m2+2)
         return ((1./(double(l2-m2)*double(l2-m2-1)))*(double(2*l2-1)*three_ALP_J_integral(l1,l2-1,l3,m1,m2+1,m3)
               +double(l2+m2)*(l2+m2-1)*J_int_m2(l1,l2-2,l3,m1,m2,m3)));
   }
   else
   {
      if(l3 == m3)
         return - (2*l3-1)*three_ALP_J_integral(l1,l2,l3-1,m1,m2,m3-1);
      else if(l3 == m3+1 && l3>=2)
         return - (2*m3+1)*three_ALP_J_integral(l1,l2,m3,m1,m2,m3-1);
      else if(l3==1 && m3==0)
         return 0;

      if(l3>=m3+2)
         return  ((1./(double(l3-m3)*double(l3-m3-1)))*(double(2*l3-1)*three_ALP_J_integral(l1,l2,l3-1,m1,m2,m3+1)
               +double(l3+m3)*(l3+m3-1)*J_int_m2(l1,l2,l3-2,m1,m2,m3)));
   }
   std::cout<<"REACHED THE DEAD END OF J_INT_M2 FUNCTION. CASE IS"<<l1<<","<<m1<<";"<<l2<<","<<m2<<";"<<l3<<","<<m3<<std::endl;
   exit(EXIT_SUCCESS);
   return sum;
}
double J_int_m1(int l1,int l2,int l3,int m1,int m2,int m3)
{
   //CHECKED ON MARCH 2 2020
   if(l1<0 || l2<0 || l3<0 || m1>l1 || m2>l2 || m3>l3)
      return 0;
   if(l1==0 || l1-1 < m1+1)
   {
      return -((1./(2.*l1+1))*three_ALP_J_integral(l1+1,l2,l3,m1+1,m2,m3));
   }
   else
   {
      return ((1./(2.*l1+1))*(three_ALP_J_integral(l1-1,l2,l3,m1+1,m2,m3)-three_ALP_J_integral(l1+1,l2,l3,m1+1,m2,m3)));
   }
}
double J_int_p1(int l1,int l2,int l3,int m1,int m2,int m3)
{
   //CHECKED ON MARCH 2 2020
   if(l1<0 || l2<0 || l3<0 || m1>l1 || m2>l2 || m3>l3)
      return 0;
   if(l1==0 || l1-1 < m1)
   {
      return ((1./(2.*l1+1.))*(double(l1-m1+1.)*three_ALP_J_integral(l1+1,l2,l3,m1,m2,m3)));
   }
   else
   {
      return ((1./(2.*l1+1.))*(double(l1-m1+1.)*three_ALP_J_integral(l1+1,l2,l3,m1,m2,m3)+double(l1+m1)*three_ALP_J_integral(l1-1,l2,l3,m1,m2,m3)));
   }
}
double J_int_m1_D(int l1,int l2,int l3,int m1,int m2,int m3)
{
   //CHECKED ON MARCH 10 2020
   //This computes D_thet P_l^m(cos(thet))=-(1-x**2)**0.5 * D_xP_l^m(x)
   double prefactor;
   double DP2,DP3;
   if((l1+m1+l2+m2+l3+m3)%2==0) //If the integrand is odd
      return 0;
   else
   {

      prefactor=Jint_signflip_renormalize(l1,l2,l3,&m1,&m2,&m3); // Flip the sign of negative m's and renormalize accordingly

      if(l1<0 || l2<0 || l3<0 || m1>l1 || m2>l2 || m3>l3)
         return 0;

      else if( l2 == 0 && l3 == 0  )
         return 0;
      else
      {
             if(l2==0)
                DP2=0;
             else if(l2>0 && m2==0)
                DP2=prefactor*three_ALP_J_integral(l1,l2,l3,m1,1,m3);
             else if(l2==m2)
                DP2=-0.5*((l2+m2)*(l2-m2+1)*prefactor*three_ALP_J_integral(l1,l2,l3,m1,m2-1,m3));
             else
                DP2=-0.5*((l2+m2)*(l2-m2+1)*prefactor*three_ALP_J_integral(l1,l2,l3,m1,m2-1,m3)-prefactor*three_ALP_J_integral(l1,l2,l3,m1,m2+1,m3));
             if(l3==0)
                DP3=0;
             else if(l3>0 && m3==0)
                DP3=prefactor*three_ALP_J_integral(l1,l2,l3,m1,m2,1);
             else if(l3==m3)
                DP3=0.5*((l3+m3)*(l3-m3+1)*prefactor*three_ALP_J_integral(l1,l2,l3,m1,m2,m3-1));
             else
                DP3=-0.5*((l3+m3)*(l3-m3+1)*prefactor*three_ALP_J_integral(l1,l2,l3,m1,m2,m3-1)-prefactor*three_ALP_J_integral(l1,l2,l3,m1,m2,m3+1));

             return (DP2+DP3);
      }
   }
}
double J_int_p1_D(int l1,int l2,int l3,int m1,int m2,int m3)
{
   //CHECKED ON MARCH 10 2020
   if((l1+m1+l2+m2+l3+m3+1)%2 == 0)
      return 0;
   else if((l2 == 0 && l3 == 0))
      return 0;
   else
   {
      if(l1==0 || l1-1 < m1)
      {
         return ((1./(2.*l1+1.))*(double(l1-m1+1.)*J_int_m1_D(l1+1,l2,l3,m1,m2,m3)));
      }
      else
      {
         return ((1./(2.*l1+1.))*(double(l1-m1+1.)*J_int_m1_D(l1+1,l2,l3,m1,m2,m3)+double(l1+m1)*J_int_m1_D(l1-1,l2,l3,m1,m2,m3)));
      }

   }
}
double three_Ylm_integ(int l1,int l2,int l3,int m1,int m2,int m3)
{
   return sqrt( (2*l1+1) * ( 2*l2+1) * ( 2*l3+1 ) / ( 4 * acos(-1) ) ) * gsl_sf_coupling_3j(l1,l2,l3,0,0,0)* gsl_sf_coupling_3j(l1,l2,l3,m1,m2,m3);; 
}
double clebsch_gordan_coeff(unsigned int l1,unsigned int l2,unsigned int l3,int m1,int m2,int m3)
{
   std::cout<<"Entering Clebsch-Gordan with "<<l1<<","<<l2<<","<<l3<<" --- "<<m1<<","<<m2<<","<<m3<<std::endl;
   return pow(-1,l1-l2+m3)*sqrt(2*l3+1)*gsl_sf_coupling_3j(2*l1,2*l2,2*l3,2*m1,2*m2,-2*m3);
}
void B_coeff(int l,int m1,int m2,std::vector<double> *B_val)
{
   //This function computes the expansion coefficients of a product of a spherical harmonics with a Y_{1}^{m} spherical harmonics.
   //The expansion coefficients describe the coefficients for the expansion in a sum of spherical hamronics
   //

   B_val->clear();
   if(l < 0)
   {
      B_val->push_back(0);
      B_val->push_back(0);
      return ;
   }
   else if( (l-1) >= abs(m1+m2) )
   {
      B_val->push_back(pow(-1,m1+m2)*sqrt(3.*(2.*double(l)+1.)*(2.*double(l)-1)/(4*acos(-1)))*gsl_sf_coupling_3j(2*l,2,2*(l-1),2*m1,2*m2,-2*(m1+m2))*gsl_sf_coupling_3j(2*l,2,2*(l-1),0,0,0));

   }
   else
      B_val->push_back(0);

   B_val->push_back(pow(-1,m1+m2)*sqrt(3.*(2.*double(l)+1.)*(2.*double(l)+3.)/(4*acos(-1)))*gsl_sf_coupling_3j(2*l,2,2*(l+1),2*m1,2*m2,-2*(m1+m2))*gsl_sf_coupling_3j(2*l,2,2*(l+1),0,0,0));

   return;
}
//////////////////////////////////////////////////
//
// This computes the partial sums that are involved in the computation of the terms in the transition dipole integrals.
// The partial sum can be written \sum_{l=|l_a-l_b|}^{la+lb} ( Y_{l,0}C_{m_a,m_b,0}+\sum_{m=1}^{l}[Y_{l,|m|}C_{m_a,m_b,|m|}+Y_{l,-|m|}C_{m_a,m_b,-|m|}] ) 
//
//////////////////////////////////////////////////
double CY_m_sum(double thet,double phi,unsigned int la,unsigned int lb,unsigned int l,int ma,int mb)
{

   //in all the terms of this sum, the value is zero if m1+m2+m is odd.
   //Because three_ALP_J_integral is zero whenever (l1+l2+l + m1+m2+m) is odd, l1+l2+l must be even for the integrals not be zero
   double result(0);

   if( abs(ma) > la || abs(mb) > lb )
      return 0;

   //Add the m=0 term
   result+=rYlm(l,0,thet,phi)*prefactor_rYlm(la,ma)*prefactor_rYlm(lb,mb)*prefactor_rYlm(l,0)
      *three_azim_integ(ma,mb,0)*three_ALP_J_integral(la,lb,l,abs(ma),abs(mb),0) * bool((ma+mb) % 2 == 0);

   //add the other terms of the sum
   for(int m=1;m<=int(l);m++)
   {
      if( (ma+mb+m)%2!=0 )
         continue;
      else
         result+=(
         rYlm(l,m,thet,phi)*prefactor_rYlm(la,ma)*prefactor_rYlm(lb,mb)*prefactor_rYlm(l,m)
         *three_azim_integ(ma,mb,m)*three_ALP_J_integral(la,lb,l,abs(ma),abs(mb),m)
         +rYlm(l,-m,thet,phi)*prefactor_rYlm(la,ma)*prefactor_rYlm(lb,mb)*prefactor_rYlm(l,-m)
         *three_azim_integ(ma,mb,-m)*three_ALP_J_integral(la,lb,l,abs(ma),abs(mb),m));//*three_Ylm_integ(la,lb,l,ma,mb,m);
   }
   return result;
}
