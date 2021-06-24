#include <cstdlib>
#include <iostream>
#include <cmath>
#include <gsl/gsl_sf_legendre.h>


double prefactor_rYlm(int l,int m)
{
   /*
    * THIS FUNCTION COMPUTES THE SPHERICAL HARMONICS NORMALIZATION PREFACTOR 
    * WE EXPLICITELY INCLUDE THE CONDON SHORTLEY PHASE IN THE EXPRESSION OF THE SPHERICAL HARMONICS PREFACTOR
    * SO THAT WE DO NOT TAKE IT  TWICE INTO ACCOUNT WHENEVALUATING THE ASSOCIATED LEGENDRE POLYNOMIALS.
    */
   double temp(1);
   double val(0);

   if(fabs(m) > fabs(l))
   {
      std::cout<<"FATAL ERROR IN SPHERICAL HARMONICS PREFACTOR. M>L:"<<m<<">"<<l<<std::endl;
      return 0;
   }

   if(m == 0)
   {
      return sqrt((2*l+1)/(4*acos(-1)));
   }
   else if(m > 0)
   {
      val= pow(-1,m) * sqrt(2.) * sqrt((2*l+1)/ (4*acos(-1)));
      temp=1.;
      for (int tt=l-m+1;tt!=l+m+1;tt++)
      {
         temp/=double(tt);
      }
      val*=sqrt(temp);

      return val;
   }
   else 
   {
      return prefactor_rYlm(l,abs(m));
   }
}
double rYlm (int l,int m,double thet,double phi)
{
//   std::cout<<"Entering rYlm with parameters "<<l<<","<<m<<","<<thet<<","<<phi<<std::endl;
      if(m<0)
      {
         return prefactor_rYlm(l,m)*gsl_sf_legendre_Plm(l,-m,cos(thet))*sin(abs(m)*phi);
         //*std::assoc_legendre(l,-m,cos(thet))*sin(abs(m)*phi);
      }
      else if(m>0)
      {
         return prefactor_rYlm(l,m)*gsl_sf_legendre_Plm(l,m,cos(thet))*cos(m*phi);
         //*std::assoc_legendre(l,m,cos(thet))*cos(m*phi);
      }
      else
      {
         return prefactor_rYlm(l,m)*gsl_sf_legendre_Plm(l,m,cos(thet));//*std::assoc_legendre(l,m,cos(thet));
      }
}
