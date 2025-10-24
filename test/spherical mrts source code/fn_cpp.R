library("RSpectra") ; source("mKrig.R") ; library("fields") ; library("GpGp") ; library("MASS")
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

//fk
// [[Rcpp::export]]
NumericVector cpp_fk2(double L1, double l1, double KK, NumericMatrix X, NumericVector Konev, 
                      NumericMatrix eiKvecmval, double n){
    NumericVector f1(KK);
    NumericVector f2(n);
    f1[0] = sqrt(1/n);
    NumericMatrix t(1,n);
    for (int i = 0; i < n; i++) {
        t[i] = cpp_Kf(L1,l1,X(i,0),X(i,1))-Konev[i];
    }
    return t;
}


// [[Rcpp::export]]
NumericMatrix cpp_Kmatrix2(double KK, NumericMatrix X, NumericMatrix ggrids, NumericVector Konev, 
                           NumericMatrix eiKvecmval, double n, double N) {
    NumericMatrix xx(N,n);
    for (int i = 0; i < N; i++) {
        xx(i,_) = cpp_fk2(ggrids(i,0),ggrids(i,1),KK,X,Konev,eiKvecmval,n);
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

# knot = X ; bigK = 10 ; obsdata = z ; obsloc = X ; newloc = grids
mrlm = function(knot,bigK,obsdata,obsloc,newloc)
{
  n = nrow(obsloc) ; N = nrow(newloc)
  # X for control points ; bigK for the number of the basis functions ; grids for the data points
  dm_train = mrts_sphere(knot, bigK, obsloc)$mrts
  dm = mrts_sphere(knot, bigK, newloc)$mrts
  beta = numeric(bigK) ; 
  for(sk in 1:bigK)
  {
    s = dm_train[,sk] 
    beta[sk] = ((t(s)%*%s)^-1)*(t(s)%*%z )
  }
  s1 = mean((z - dm_train[,1:bigK]%*%beta[1:bigK])^2)
  cAICl = numeric(bigK)
  for(bb in 1:bigK)
  {
    if(bb == 1) 
    {
      cAICl[bb] = ( sum((z - dm_train[,1]*beta[1])^2)+2*bb*
                      s1 )/n
    } else 
    {
      cAICl[bb] = ( sum((z - dm_train[,1:bb]%*%beta[1:bb])^2)+2*bb*s1 )/n
    }
  }
  for(i in 1:length(cAICl))
  {
    if(cAICl[i]==min(cAICl)) {mincal = i}
  }
  yhat = matrix(0,N,bigK)
  bb = bigK
  if(bb == 1) {yhat=dm[,1:bb]*beta[1:bb];
  } else 
  {
    yhat = dm[,1:bb]%*%beta[1:bb]
  }
  bb = mincal
  if(bb == 1) {yhat_cal=dm[,1:bb]*beta[1:bb];
  } else 
  {
    yhat_cal = dm[,1:bb]%*%beta[1:bb]
  }
  res = {}
  res$pred = yhat ; res$pred_cal = yhat_cal ; res$caic_k = cAICl[bigK] ; res$mcaic = cAICl[mincal] ; res$sigma2 = s1
  res$mincaic = mincal
  return(res)
}

mrmm = function(knot,bigK,obsdata,obsloc,newloc,a,param)
{
  n = nrow(obsloc) ; N = nrow(newloc)
  # X for control points ; bigK for the number of the basis functions ; grids for the data points
  dm_train = mrts_sphere(knot, bigK, obsloc)$mrts
  dm = mrts_sphere(knot, bigK, newloc)$mrts
  fun <- function(param)
  {
    res = mKrig( obsloc[,2:1], obsdata, Z = dm_train[,1:bigK], m=0,cov.function = "stationary.taper.cov",#
                 cov.args = list(Dist.args = list(method = "greatcircle", R=6378.388),aRange= param[1]*6378.388,
                                 Taper.args = list(k=2, dimension=2, aRange = (6378.388*a*pi/180))),
                 sigma2= param[2], tau=param[3])$lnLike.FULL
    return(res)
  }
  L <- optim(c(0.2,0.5, 0.5), fun, method = "L-BFGS-B", # this method lets set lower bounds "BFGS"
             # lower = c(0.01,0.001,0.001,0.001), # lower limit for parameters
             # upper = 10,
             control = list(fnscale = -1, maxit = 10) # maximize the function
             # hessian = T # calculate Hessian matricce because we will need for confidence intervals
  )
  s1 = L$par[3]
  cAIC = numeric(bigK)
  for(bb in 1:bigK)
  {
    L <- optim(c(0.2,0.5, 0.5), fun, method = "L-BFGS-B", # this method lets set lower bounds "BFGS"
               # lower = c(0.01,0.001,0.001,0.001), # lower limit for parameters
               # upper = 10,
               control = list(fnscale = -1, maxit = 10) # maximize the function
               # hessian = T # calculate Hessian matricce because we will need for confidence intervals
    )
    vy_hat = L$par[2] ; c_hat = L$par[1] ; sigma2_hat = L$par[3]
    rema1 = cov_exp_hat = cov = vy_hat*Wendland(rdist.earth(obsloc[,2:1], R=1),theta=pi/a,k=2,dimension = 2)*
      stationary.cov(obsloc[,2:1], theta = c_hat,Dist.args=list(R=1), Distance = "rdist.earth" )
    
    cov = cov + diag(sigma2_hat,n,n)
    incov_exp_hat = solve(cov)
    fin = t(dm_train[,1:bb])%*%incov_exp_hat
    fincov = fin%*%dm_train[,1:bb]
    infincov = solve(fincov)
    beta_hat = infincov %*% fin %*% obsdata
    rema2 = rema1%*%incov_exp_hat;
    h1 = cbind(cov_exp_hat,dm_train[,1:bb])  ; h2 = cbind(t(dm_train[,1:bb]),matrix(0,bb,bb));
    h3 = rbind(h1,h2);
    H = t(solve( h3, t( cbind(rema1,dm_train[,1:bb]) ) ) );
    if(bb == 1)
    {
      yhat_temp = dm_train[,1:bb]*beta_hat[1:bb]+rema2%*%(z-dm_train[,1:bb]*beta_hat[1:bb])
    } else{
      yhat_temp = dm_train[,1:bb]%*%beta_hat[1:bb]+rema2%*%(z-dm_train[,1:bb]%*%beta_hat[1:bb])
    }
    cAIC[bb] = (sum((z - yhat_temp)^2) + 2*sum(diag(H) )*s1)/n
  }
  rema = vy_hat*Wendland(rdist.earth(newloc[,2:1],obsloc[,2:1], R=1),theta=pi/a,k=2,dimension = 2)*
    stationary.cov(newloc[,2:1],obsloc[,2:1], theta = c_hat,Dist.args=list(R=1), Distance = "rdist.earth" )%*%
    incov_exp_hat;
  for(i in 1:length(cAIC))
  {
    if(cAIC[i]==min(cAIC)) {minca = i}
  }
  bb = bigK
  if(bb == 1) {yhat = dm[,1:bb]%*%beta_hat[1]+rema%*%(z-dm_train[,1:bb]%*%beta_hat[1]);
  } else 
  {
    yhat = dm[,1:bb]%*%beta_hat[1:bb]+rema%*%(z-dm_train[,1:bb]%*%beta_hat[1:bb])
  }
  bb = minca
  if(bb == 1) {yhat_cal=dm[,1:bb]%*%beta_hat[1]+rema%*%(z-dm_train[,1:bb]%*%beta_hat[1]);
  } else 
  {
    yhat_cal = dm[,1:bb]%*%beta_hat[1:bb]+rema%*%(z-dm_train[,1:bb]%*%beta_hat[1:bb])
  }
  res = {}
  res$pred = yhat ; res$pred_cal = yhat_cal ; res$caic_k = cAICl[bigK] ; res$mcaic = cAICl[mincal] ; res$sigma2 = s1
  res$mincaic = minca
  return(res)
}

