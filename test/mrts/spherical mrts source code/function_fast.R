library("fields") ; library("maps") ; library("GpGp") 
# library("rcosmo") ; library("pracma") ; library("CompRandFld") ;library("GeoDist")
# library("sp") ; library("fields") ; library("Matrix")
library("RSpectra")
library("geoR")
library("ggplot2")
# library("rgdal")
# library("maptools")
library("raster")
library("MASS")
# library("mvtnorm")
# library("rcosmo")
# library("gridExtra")
# library("LatticeKrig")
# library("mapproj")
#-----function
angle = function(t1, p1, t2, p2){
  t1 = t1*pi/180 ; p1 = p1*pi/180 ; t2 = t2*pi/180 ; p2 = p2*pi/180;
  a = acos( pmin(pmax(sin(t1)*sin(t2) + cos(t1)*cos(t2)*cos(p1-p2),-1.0),1.0) ) ; return(a)}

#Kf = function(t1, p1, t2, p2) {a = angle(t1, p1, t2, p2) ; integrand <- function(x) 
#{log(x)/(1-x)} ;
#result = 1-(pi^2)/6+integrate(integrand, lower = 1, upper = (1-cos(a))/2)$value ; return(result)}

Kf = function(t1, p1, t2, p2) {a = angle(t1, p1, t2, p2) ; if(a==0) {result=1} 
if(cos(a)==-1) {result=1-(pi^2)/6} else{integrand <- function(x)
{log(1-x)/x} ; result = 1-(pi^2)/6-integrate(integrand, lower = 0,
                                             upper = (1/2+cos(a)/2))$value} ; return(result)}

# Kf = function(t1, p1, t2, p2) {integrand <- function(x) {a = angle(t1, p1, t2, p2) ; log(x)*(1-1/x)*(1/((1-2*x*a+x^2)^(1/2)-1)-1)} ;
# result = (1/(4*pi))*integrate(integrand, lower = 0, upper = 1)$value ; return(result)}


#phi(s)
fk = function(L,l,KK,X)
{
  fktem = c(numeric(KK)) ; fktem2 = numeric(n); fktem[1] = (1/n)^(1/2) ; 
  for(i in 1:length(X[,1]))
  {
    fktem2[i] = Kf(L,l,X[i,1],X[i,2])
  } ;
  for(q in 2:KK)
  {
    fktem[q] = t(fktem2-K%*%onev) %*% eiK$vectors[,q-1] / eiK$values[q-1]
  } ;
  return(fktem)
}
#phi(s)_noconstant
fk1 = function(L,l,KK,X)
{
  fktem = c(numeric(KK)) ; fktem2 = numeric(n) ; 
  for(i in 1:length(X[,1]))
  {
    fktem2[i] = Kf(L,l,X[i,1],X[i,2])
  } ;
  for(q in 1:KK)
  {
    fktem[q] = t(fktem2) %*% eiK1$vectors[,q] / eiK1$values[q]
  } ;
  return(fktem)
}
#frk
frk = function(n,KK)
{
  if(KK==1) {f1tem = seq((1/n)^(1/2),(1/n)^(1/2),length.out = n)} else
  {
    q = KK
    f1tem = numeric(n) 
    for(i in 1:n)
    {
      f1tem[i] = t(K[i,]-K%*%onev) %*% eiK$vectors[,q-1] / eiK$values[q-1]
    }  
  }
  return(f1tem)
}
#frk_noconstant
frk1 = function(n,KK)
{
  q = KK
  f1tem = numeric(n) 
  for(i in 1:n)
  {
    f1tem[i] = t(K[i,]) %*% eiK1$vectors[,q] / eiK1$values[q]
  }  
  return(f1tem)
}
#--exp
expcov = function(L,l,para,vy_hat)
{
  covtem = numeric(n) ;
  for(i in 1:length(X[,1]))
  {
    
    a = angle(L, l, X[i,1], X[i,2])
    para_a = para ; covtem[i] = vy_hat*exp(-(a/para_a)) 
    
    
  } ;
  return(covtem)
}
poicov = function(L,l,X)
{
  covtem = numeric(n) ;
  for(i in 1:length(X[,1]))
  {
    a = angle(L, l, X[i,1], X[i,2])
    r = 0.5 ; covtem[i] = (1-r^2) / (1-2*r*cos(a)+r^2)^(3/2)
  } ;
  return(covtem)
}
sphcov = function(L,l,X)
{
  covtem = numeric(n) ;
  for(i in 1:length(X[,1]))
  {
    a = angle(L, l, X[i,1], X[i,2])
    aa = 1 ; if(a>aa){covtem[i]=0} else {covtem[i]=(1-(3*a/(2*aa))+(a^3/(2*aa^3)))}
  } ;
  return(covtem)
}
matcov = function(L,l,vy_hat_m)
{
  covtem = numeric(n) ;
  for(i in 1:length(X[,1]))
  {
    a = angle(L, l, X[i,1], X[i,2])
    if(a==0){covtem[i]==vy_hat_m} else{ cc = 1; covtem[i] = vy_hat_m*(2^(1/2)/gamma(1/2))*
      (a/cc)^(1/2) *besselK(a/cc,1/2 )}
  } ;
  return(covtem)
}

covin = function(x)
{
  co = {}
  KK=x 
  F = matrix(0,n,KK)
  F[,1] = seq((1/n)^(1/2),(1/n)^(1/2),length.out = n)
  for(q in 2:KK)
  {
    for(i in 1:n)
    {
      F[i,q] = t(K[i,]-K%*%onev) %*% eiK$vectors[,q-1] / eiK$values[q-1]
    }
  }
  co$F = F
  # FFeig = eigen(t(F)%*%F)
  # Fhalf = FFeig$vectors%*%diag(FFeig$values^(-1/2))%*%t(FFeig$vectors)
  Fhalf = getHalf(F,F)
  S = z%*%t(z)/T 
  P = eigen(Fhalf%*%t(F)%*%S%*%F%*%Fhalf)
  # for(i in 1:KK)
  # {
  #   if(P$values[i] - sigma2 < 0) P$values[i] = 0
  # }
  if(is.complex(P$values) == T | is.complex(P$vectors) == T) fu = 1 else {fu = 0}
  if(is.complex(P$values) == T) P$values = Re(P$values)
  P$values = pmax(P$values - sigma2, 0)
  if(is.complex(P$vectors) == T) P$vectors = Re(P$vectors)
  co$Mhat = Fhalf%*%P$vectors%*%diag(P$values)%*%t(P$vectors)%*%Fhalf
  for(i in 1:KK)
  {
    if(P$values[i] < pmax( (1/(n-i))*(sum(diag(S))-sum(P$values[1:i])),sigma2 ) ) L_ast=i-1
  }
  co$sigma2_xi_hat = pmax( (1/(n-L_ast))*(sum(diag(S))-sum(P$values[1:i]))-sigma2,0 )
  if(KK<=T) {df = KK*(KK+1)/2} else {df = KK*T+1-T*(T+1)/2}
  L = F%*%Fhalf
  co$covin = (1/(sigma2+sigma2_xi_hat))*(diag(1,n,n)-L%*%P$vectors%*%diag(P$values/(P$values+sigma2+sigma2_xi_hat))%*%t(P$vectors)%*%t(L))
  co$fu = fu
  return(co)
}

AIC = function(K,eiK,bigK)
{
  order = bigK ; candi = numeric(order)
  for(KK in 2:order)
  {
    F = matrix(0,n,KK)
    F[,1] = seq((1/n)^(1/2),(1/n)^(1/2),length.out = n)
    for(q in 2:KK)
    {
      for(i in 1:n)
      {
        F[i,q] = t(K[i,]-K%*%onev) %*% eiK$vectors[,q-1] / eiK$values[q-1]
      }
    }
    
    Fhalf = getHalf(F,F)
    S = z%*%t(z)/T 
    P = eigen(Fhalf%*%t(F)%*%S%*%F%*%Fhalf)
    if(is.complex(P$values) == T) P$values = Re(P$values)
    P$values = pmax(P$values - sigma2, 0)
    if(is.complex(P$vectors) == T) P$vectors = Re(P$vectors)
    for(i in 1:KK)
    {
      if(P$values[i] - sigma2 < 0) P$values[i] = 0
    }
    Mhat = Fhalf%*%P$vectors%*%diag(P$values)%*%t(P$vectors)%*%Fhalf
    if(KK<=T) {df = KK*(KK+1)/2} else {df = KK*T+1-T*(T+1)/2}
    AIC = T*sum(diag(S))/sigma2 + T*sum(log(P$values+sigma2)-P$values*P$values/(sigma2*(P$values+sigma2))) + 
      T*(n-KK)*log(sigma2)+2*df
    candi[KK] = AIC
  }
  candi[1] = max(candi)
  for(i in 2:order)
  {
    if(candi[i] == min(candi)) KK = i
  }
  return(KK)
}
rankcheck = function(KK)
{
  F = matrix(0,n,KK)
  F[,1] = seq((1/n)^(1/2),(1/n)^(1/2),length.out = n)
  for(q in 2:KK)
  {
    for(i in 1:n)
    {
      F[i,q] = t(K[i,]-K%*%onev) %*% eiK$vectors[,q-1] / eiK$values[q-1]
    }
  }
  FFeig = eigen(t(F)%*%F)
  Fhalf = FFeig$vectors%*%diag(FFeig$values^(-1/2))%*%t(FFeig$vectors)
  S = z%*%t(z)/T 
  P = eigen(Fhalf%*%t(F)%*%S%*%F%*%Fhalf)
  for(i in 1:length(P$values)) {if(is.na((P$values^(-1/2))[i]) == T) {a = 1} else {a = 0}}
  if(is.complex(P$values) == T){a=1}
  return(a)
}

predict.frk = function(i,KK,z)
{
  (fk(grids[i,1],grids[i,2],KK,X)%*%covin(KK)$Mhat%*%t(covin(KK)$F))%*%covin(KK)$covin%*%z
}

getHalf <-
  function (Fk, iDFk) 
  {
    dec = eigen(t(Fk) %*% iDFk)
    sroot = sqrt(pmax(dec$value, 0))
    sroot[sroot == 0] = Inf
    sroot = 1/sroot
    dec$vector %*% (sroot * t(dec$vector))
  }

getHalf2 <-
  function (Fk) 
  {
    dec = eigen(Fk)
    sroot = sqrt(pmax(dec$value, 0))
    sroot[sroot == 0] = Inf
    sroot = 1/sroot
    dec$vector %*% (sroot * t(dec$vector))
  }

Sig = function(z,c,vy_hat,X) 
{
  N = length(z)
  cov.mat = matrix(0, N, N)
  for(i in 1:N)
  {
    for(j in 1:N)
    {
      a = angle(X[i,1], X[i,2], X[j,1], X[j,2])
      para_a = c ; cov.mat[i,j] = vy_hat*exp(-(a/para_a))
    }
  }
  return(cov.mat)
}

L = function(cccc)
{
  z = residual_hat
  c = cccc[1] ; sigma2_hat = cccc[2] ; vy_hat = cccc[3]
  Sig = function(z,c,sigma2_hat) 
  {
    N = length(z)
    cov.mat = matrix(0, N, N)
    for(i in 1:N)
    {
      for(j in 1:N)
      {
        a = angle(X[i,1], X[i,2], X[j,1], X[j,2])
        if(i==j) {para_a = c ; cov.mat[i,j] = vy_hat*exp(-(a/para_a))+sigma2_hat}
        else {para_a = c ; cov.mat[i,j] = vy_hat*exp(-(a/para_a))}
      }
    }
    return(cov.mat)
  }
  if(T==1) 
  {
    detCOV = T*log(det(Sig(X[,1],c,sigma2_hat)))
    COV = t(z) %*% solve(Sig(X[,1],c,sigma2_hat)) %*% z
  } else 
  {
    detCOV = T*log(det(Sig(X[,1],c,sigma2_hat)))
    COV = 0 ; SS = ginv(Sig(X[,1],c,sigma2_hat))
    for(i in 1:T)
    {
      COV = COV + t(z[i,]) %*% SS %*% z[i,]
    }
  }
  -(1/2)*detCOV-(1/2)*COV
}


# L = function(cccc,residual_hat,SS)
# {
#   z = residual_hat
#   c = cccc[1] ; sigma2_hat = cccc[2] ; vy_hat = cccc[3]
#   detCOV = T*log(det(SS))
#   COV = t(z) %*% solve(SS) %*% z
#   return(-(1/2)*detCOV-(1/2)*COV)
# }

#cpp


library("Rcpp")
src <-
  "// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]

#define _USE_MATH_DEFINES
#include <RcppEigen.h>
#include <cmath>
#include <Rcpp.h>
#include <iostream>
#include <RcppNumerical.h>
using namespace Numer;
using namespace Rcpp;
using Eigen::Map;                       // 'maps' rather than copies
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::VectorXd;                  // variable size vector, double precision
using Eigen::SelfAdjointEigenSolver;    // one of the eigenvalue solvers

// [[Rcpp::export]]
List getEigen(Map<MatrixXd> M) {
SelfAdjointEigenSolver<MatrixXd> es(M);
MatrixXd x = es.eigenvectors();
VectorXd y = es.eigenvalues();
return List::create(y, x );
}

//inner product
double inprod(NumericVector A, NumericVector B, int m){
double xx=0;
for(int i=0; i<m; i++) {
xx+=A(i)*B(i); /*?????????C */
}
return xx;
}

//matrix product
NumericMatrix mprod(NumericMatrix A, NumericMatrix B, int m, int n, int p){
int C[m][p];
int i, j, k;
NumericMatrix xx(m,p);
for (i=0; i<m; i++) {
for (j=0; j<p; j++) {
C[i][j]=0; /*???????????C */
for(k = 0; k < n; k++) {
C[i][j] = C[i][j] +A(i,k) * B(k,j); /*????A????????B,????????C */
}
xx(i,j)=C[i][j]; /*?????????C */
}
}
return xx;
}

class func3: public Func
{
  public:
  
  double operator()(const double& x) const
  {
  return (double)log(1-x)/x;
  }
  };
  
  //Kf
  // [[Rcpp::export]]
  double cpp_Kf(double L1, double l1, double L2, double l2){
  double mia= (double)M_PI/180;
  double a = sin(L1*mia)*sin(L2*mia) + cos(L1*mia)*cos(L2*mia)*cos(l1*mia-l2*mia) ;
  NumericVector v1(2);
  v1[0]=a;
  v1[1]=-1;
  float x = max(v1) ;
  NumericVector v2(2);
  v2[0]=x;
  v2[1]=1;
  float b = min(v2) ;
  double aaa = acos(b);
  double result;
  if(cos(aaa)==-1) {
  result = 1-(double)pow(M_PI,2)/6 ;
  } else {
  double aa = (double)1/2+(double)cos( aaa )/2;
  double lower = 0;
  func3 f;
  double err_est;
  int err_code;
  const double res = integrate(f, lower, aa, err_est, err_code);
  result = 1-(double)pow(M_PI,2)/6-res;
  }
  
  return result;
  }

// [[Rcpp::export]]
NumericMatrix cpp_Kmatrix(int KK, NumericMatrix X, NumericMatrix ggrids, NumericVector Konev, 
                           NumericMatrix eiKvecmval, int n, int N) {

  NumericMatrix xx(N, KK);
  double mia = M_PI / 180.0;

  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    double L1 = ggrids(i, 0);
    double l1 = ggrids(i, 1);

    // Precompute f2 = cpp_Kf for this row
    std::vector<double> f2(n);
    for (int j = 0; j < n; j++) {
      double L2 = X(j, 0);
      double l2 = X(j, 1);

      double a = sin(L1 * mia) * sin(L2 * mia) +
                 cos(L1 * mia) * cos(L2 * mia) * cos(l1 * mia - l2 * mia);

      double b = std::max(-1.0, std::min(a, 1.0));
      double aaa = acos(b);
      double result;
      if (cos(aaa) == -1) {
        result = 1.0 - pow(M_PI, 2) / 6.0;
      } else {
        double aa = 0.5 + cos(aaa) / 2.0;
        double lower = 0;
        func3 f;
        double err_est;
        int err_code;
        double res = integrate(f, lower, aa, err_est, err_code);
        result = 1.0 - pow(M_PI, 2) / 6.0 - res;
      }
      f2[j] = result;
    }

    // t = f2 - Konev
    std::vector<double> t(n);
    for (int j = 0; j < n; j++) {
      t[j] = f2[j] - Konev[j];
    }

    // First basis function
    xx(i, 0) = sqrt(1.0 / n);

    // Project onto each basis function
    for (int k = 1; k < KK; k++) {
      double s = 0.0;
      for (int j = 0; j < n; j++) {
        s += t[j] * eiKvecmval(j, k - 1);
      }
      xx(i, k) = s;
    }
  }

  return xx;
}

//fk
// [[Rcpp::export]]
NumericVector cpp_fk(double L1, double l1, double KK, NumericMatrix X, NumericVector Konev, 
NumericMatrix eiKvecmval, double n){
NumericVector f1(KK);
NumericVector f2(n);
f1[0] = sqrt(1/n);
for (int i = 0; i < n; i++) {
f2[i] = cpp_Kf(L1,l1,X(i,0),X(i,1));
}
NumericMatrix t(1,n);
for (int i = 0; i < n; i++) {
t[i] = f2[i]-Konev[i];
}
for (int i = 1; i < KK; i++) {
f1[i] = inprod(t,eiKvecmval(_,(i-1)),n);
}
return f1;
}

// [[Rcpp::export]]
NumericMatrix cpp_Kmatrix3(double KK, NumericMatrix X, NumericMatrix ggrids, NumericVector Konev, 
NumericMatrix eiKvecmval, double n, double N) {
NumericMatrix xx(N,KK);
for (int i = 0; i < N; i++) {
xx(i,_) = cpp_fk(ggrids(i,0),ggrids(i,1),KK,X,Konev,eiKvecmval,n);
}
return xx;
}

// [[Rcpp::export]]
NumericMatrix cpp_K(NumericVector X,NumericVector Y, int n) {
NumericMatrix xx(n);
for (int i = 0; i < n; i++) {
for (int j = 0; j < n; j++) {
xx(i,j) = cpp_Kf(X[i],Y[i],X[j],Y[j]);
}
}
return xx;
}

// [[Rcpp::export]]
NumericMatrix cpp_exp(NumericMatrix X, NumericMatrix Y, int n, int N,double c, double vy) {
NumericMatrix xx(N,n);
for (int i = 0; i < N; i++) {
for (int j = 0; j < n; j++) {
double L1 = Y(i,0);
double l1 = Y(i,1);
double L2 = X(j,0);
double l2 = X(j,1);
double mia= (double)M_PI/180;
double a = sin(L1*mia)*sin(L2*mia) + cos(L1*mia)*cos(L2*mia)*cos(l1*mia-l2*mia) ;
NumericVector v1(2);
v1[0]=a;
v1[1]=-1;
float x = max(v1) ;
NumericVector v2(2);
v2[0]=x;
v2[1]=1;
float b = min(v2) ;
double aaa = acos(b);
xx(i,j) = vy*exp(-(double)aaa/c);
}
}
return xx;
}
"

sourceCpp(code = src)
