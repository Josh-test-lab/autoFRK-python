library("RSpectra") ; library("fields") ; library("GpGp") ; library("MASS")
library("RcppNumerical")
#cpp


library("Rcpp")
src <-
  "// [[Rcpp::depends(RcppArmadillo,RcppEigen)]]
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
// [[Rcpp::export]]
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

// [[Rcpp::export]]
SEXP eigenMapMatMult(const Eigen::Map<Eigen::MatrixXd> A, Eigen::Map<Eigen::MatrixXd> B){
    Eigen::MatrixXd C = A * B;

    return Rcpp::wrap(C);
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
NumericMatrix cpp_Kmatrix(double KK, NumericMatrix X, NumericMatrix ggrids, NumericVector Konev, 
NumericMatrix eiKvecmval, double n, double N) {
NumericMatrix xx(N,KK);
for (int i = 0; i < N; i++) {
xx(i,_) = cpp_fk(ggrids(i,0),ggrids(i,1),KK,X,Konev,eiKvecmval,n);
}
return xx;
}
"

sourceCpp(code = src)

mrts_sphere = function(knot, k, X)
{
    bigK = k
    n = nrow(knot) ; N = nrow(X)
    onev = seq(1/n,1/n,length.out = n)
    K = cpp_K(knot[,1], knot[,2],n)
    Q = diag(1,n,n) - (1/n)
    eiK = eigs_sym(Q%*%K%*%Q,bigK)
    eiKvecmval = matrix(0,n,bigK)
    for(i in 1:bigK)
    {
        eiKvecmval[,i] = eiK$vector[,i] /eiK$values[i]
    }
    dm_train = cpp_Kmatrix(bigK, knot, X, K%*%onev, eiKvecmval, n, N) 
    res = {}
    res$mrts = dm_train
    return(res)
}
