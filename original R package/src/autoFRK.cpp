#include <Rcpp.h>
#include <RcppEigen.h>
#include <SymEigs.h>
#include <iostream>
#include <math.h>
// [[Rcpp::depends(RcppEigen)]]

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::Lower;
typedef Map<MatrixXd> MapMatd;
using Eigen::VectorXd;                  
using Eigen::SelfAdjointEigenSolver;
using namespace Rcpp;
using namespace std;
using namespace Eigen;
using namespace Spectra;

//' Internal function: Eigen-decompose a matrix
//' @keywords internal
//' @param matrix A matrix
//' @return A list of objects
//' \item{value}{A vector of eigenvalues}
//' \item{vector}{A matrix of eigenvectors}
//' @author Wen-Ting Wang
// [[Rcpp::export]]
Rcpp::List eigenDecompose(const Eigen::Map<Eigen::MatrixXd> matrix) {
  SelfAdjointEigenSolver<Eigen::MatrixXd> es(matrix);
	Eigen::MatrixXd V;
	Eigen::VectorXd lambda;
	lambda = es.eigenvalues();
	V = es.eigenvectors();
  return Rcpp::List::create(
    Rcpp::Named("value") = lambda,
    Rcpp::Named("vector") = V
  );
}

//' Internal function: A wrapper function of 'MatrixBase::sqrt()'
//' @keywords internal
//' @param matrix A matrix
//' @return A matrix
//' @author Wen-Ting Wang
// [[Rcpp::export]]
Eigen::MatrixXd getSquareRootMatrix(Eigen::MatrixXd matrix) {
  return(matrix.sqrt());
}

//' Internal function: Inverse square root matrix of  A^t * B
//' @keywords internal
//' @param left_matrix A matrix
//' @param right_matrix A matrix
//' @return A matrix
//' @author Wen-Ting Wang
// [[Rcpp::export]]
Eigen::MatrixXd getInverseSquareRootMatrix(const Eigen::Map<Eigen::MatrixXd> left_matrix,
                                           const Eigen::Map<Eigen::MatrixXd> right_matrix) {
  SelfAdjointEigenSolver<Eigen::MatrixXd> es(left_matrix.transpose() * right_matrix);
  return(es.operatorSqrt().inverse());
}

void decomposeSymmetricMatrix(const Eigen::MatrixXd &M,
                              const int ncv,
                              const int k,
                              Eigen::VectorXd &lambda,
                              Eigen::MatrixXd &gamma) {
  /*
   * Parameters:
   *  - M: decomposed symmetric matrix
   *  - k:	Number of eigenvalues requested. 
   *     - Should satisfy 1≤nev≤n−1, where n is the size of matrix.
   *  - ncv: controls the convergence speed of the algorithm 
   * 
   */
  DenseSymMatProd<double> op(M);
  
  // Construct eigen solver object, requesting the largest k eigenvalues
  SymEigsSolver< double, LARGEST_ALGE, DenseSymMatProd<double>> eigs(&op, k, ncv);
  
  eigs.init();
  eigs.compute(1000, 1e-10);
  
  lambda = eigs.eigenvalues();
  gamma.noalias() = eigs.eigenvectors();
}

double thinPlateSplines(const double dist, const int d) {
  /*
   * Parameters:
   *  - dist: distance
   *  - d: dimension of positions (d <= 3)
   * Returns:
   *  - thin plate splines function of `dist`
   */
  double ret;
  
  if(d == 1)
    ret = pow(dist, 3) / 12;
  else if(d == 2) {
    if(dist != 0)
      ret = dist * dist * log(dist) / (8.0 * M_PI);
    else
      ret = 0;
  }
  else if(d == 3)
    ret = - dist / 8;
  else
    Rcpp::stop("Invalid dimension\n");
  
  return ret;
}

void createThinPlateMatrix(const MatrixXd s, MatrixXd &L) {
  /*
   * Parameters:
   *  - s: position matrix (n x d)
   *  - L: (return) resultant matrix (n x n)
   */
  int n(s.rows()), d(s.cols());
  double dist;
  // Update elements in the upper triangle 
  for(unsigned int i = 0; i < n; ++i) {
    for(unsigned int j = i + 1; j < n; ++j) {
      dist = (s.row(i) - s.row(j)).norm();
      L(i, j) = thinPlateSplines(dist, d);
    }
  }
  
  L += L.transpose().eval();
}

void predictThinPlateMatrix(const MatrixXd s_new,
                            const MatrixXd s,
                            MatrixXd &L) {
  /*
   * Parameters:
   *  - s_new: new position matrix (n1 x d)
   *  - s: reference position matrix (n2 x d)
   *  - L: (return) resultant matrix (n1 x n2)
   */
  int n1(s_new.rows()), n2(s.rows()), d(s.cols());
  double dist;
  
  for(unsigned int i = 0; i < n1; ++i) {
    for(unsigned int j = 0; j < n2; ++j) {
      dist = (s_new.row(i) - s.row(j)).norm();
      L(i, j) = thinPlateSplines(dist, d);
    }
  }
}

void updateMrtsBasisComponents(const Eigen::MatrixXd s,
               const int k,
               Eigen::MatrixXd &Phi,
               Eigen::MatrixXd &B,
               Eigen::MatrixXd &BBB,
               Eigen::VectorXd &lambda,
               Eigen::MatrixXd &gamma) {
  /* MRTS: eigendecompose (I-B(B'B)^{-1}B')\Phi(I-B(B'B)^{-1}B')
   *    - Phi: thin plate splines
   *    - B := (1, s)
   *    
   * Parameters:
   *  - s: position matrix (n x d)
   *  - k:	Number of eigenvalues requested. 
   *     - Should satisfy 1≤nev≤n−1, where n is the size of matrix.
   * 
   */
  int n(s.rows()), d(s.cols());
  B = MatrixXd::Ones(n, d + 1);
  Phi = MatrixXd::Zero(n, n);
  
  // Create thin plate splines Phi
  createThinPlateMatrix(s, Phi);
  
  B.rightCols(d) = s;
  const Eigen::MatrixXd Bt = B.transpose();
  Eigen::MatrixXd BtB(MatrixXd(d + 1, d + 1).setZero().
                        selfadjointView<Lower>().rankUpdate(Bt));
  
  const Eigen::LLT<MatrixXd> llt(BtB);
  // BBB := B(B'B)^{-1}B'
  BBB = llt.solve(Bt);
  // Phi_proj := \Phi((I-B(B'B)^{-1}B')
  const Eigen::MatrixXd Phi_proj = Phi - (Phi * B) * BBB;
  // quadratic := ((I-B(B'B)^{-1}B')\Phi((I-B(B'B)^{-1}B')
  const Eigen::MatrixXd quadratic = Phi_proj - BBB.transpose() * (Bt * Phi_proj);
  
  // Set a convergence threshold for eigen-decomposition
  int ncv = min(n, max(2 * k + 1, 20));
  // Update lambda and gamma
  decomposeSymmetricMatrix(quadratic, ncv, k, lambda, gamma);
}

void updateMrtsCoreComponentX(const Eigen::MatrixXd s,
               const Eigen::MatrixXd Phi,
               const Eigen::MatrixXd B,
               const Eigen::MatrixXd BBB,
               const Eigen::VectorXd lambda,
               const Eigen::MatrixXd gamma,
               int k,
               Eigen::MatrixXd &X,
               Eigen::VectorXd &nconst) {
  int n(s.rows()), d(s.cols());
  double root = sqrt(n);
  X = MatrixXd::Ones(n, k + d + 1);
  
  Eigen::MatrixXd X_center = s.rowwise() - s.colwise().mean();
  nconst = X_center.colwise().norm();
  
  X.block(0, 1, n, d) = X_center.array().rowwise() * (root / nconst.transpose().array());
  X.block(0, d + 1, n, k) = gamma * root;
  
  nconst /= root;
}

void updateMrtsCoreComponentUZ(const Eigen::MatrixXd s,
                const Eigen::MatrixXd xobs_diag,
                const Eigen::MatrixXd Phi,
                const Eigen::MatrixXd B,
                const Eigen::MatrixXd BBB,
                const Eigen::VectorXd lambda,
                const Eigen::MatrixXd gamma,
                int k,
                Eigen::MatrixXd &UZ) {
  
  int n(s.rows()), d(s.cols());
  double root = sqrt(n);
  Eigen::MatrixXd gammas;
  
  gammas = (gamma - B * (BBB * gamma)).array().rowwise() / lambda.transpose().array() * root;
  
  UZ = MatrixXd::Zero(n + d + 1, k + d + 1);
  UZ.block(0, 0, n, k) = gammas;
  UZ(n , k) = 1;
  UZ.block(n + 1, k + 1, d, d) = xobs_diag;
}

//' Internal function: Compute MRTS method
//' @keywords internal
//' @param s A location matrix
//' @param xobs_diag A matrix of observations
//' @param k A rank
//' @return A list of objects
//' \item{X}{A matrix}
//' \item{UZ}{A matrix}
//' \item{BBBH}{A matrix}
//' \item{nconst}{A vector of column means}
//' @author Wen-Ting Wang
// [[Rcpp::export]]
Rcpp::List computeMrtsRcpp(const Eigen::Map<Eigen::MatrixXd> s,
                           const Eigen::Map<Eigen::MatrixXd> xobs_diag,
                           const int k) {
  Eigen::MatrixXd Phi, X, UZ, B, BBB, gamma;
  Eigen::VectorXd lambda, nconst;
  
  // Update B, BBB, lambda, gamma
  updateMrtsBasisComponents(s, k, Phi, B, BBB, lambda, gamma);
  // Update X, UZ, nconst
  updateMrtsCoreComponentX(s, Phi, B, BBB, lambda, gamma, k, X, nconst);
  updateMrtsCoreComponentUZ(s, xobs_diag, Phi, B, BBB, lambda, gamma, k, UZ);
  
  return Rcpp::List::create(Rcpp::Named("X") = X,
                            Rcpp::Named("UZ") = UZ,
                            Rcpp::Named("BBBH") = BBB * Phi,
                            Rcpp::Named("nconst") = nconst);
}

//' Internal function: Predict on new locatoins by MRTS method
//' @keywords internal
//' @param s A location matrix
//' @param xobs_diag A matrix of observations
//' @param s_new A new location matrix
//' @param k A rank
//' @return A list of objects
//' \item{X}{A matrix}
//' \item{UZ}{A matrix}
//' \item{BBBH}{A matrix}
//' \item{nconst}{A vector of column means}
//' \item{X1}{A matrix}
//' @author Wen-Ting Wang
// [[Rcpp::export]]
Rcpp::List predictMrtsRcpp(const Eigen::Map<Eigen::MatrixXd> s,
                           const Eigen::Map<Eigen::MatrixXd> xobs_diag,
                           const Eigen::Map<Eigen::MatrixXd> s_new,
                           const int k) {
  int n(s.rows()), d(s.cols()), n2(s_new.rows());
  Eigen::MatrixXd Phi, X, UZ, B, BBB, gamma, Phi_new;
  Eigen::VectorXd lambda, nconst;
  
  // Update B, BBB, lambda, gamma
  updateMrtsBasisComponents(s, k, Phi, B, BBB, lambda, gamma);
  // Update X, UZ, nconst
  updateMrtsCoreComponentX(s, Phi, B, BBB, lambda, gamma, k, X, nconst);
  updateMrtsCoreComponentUZ(s, xobs_diag, Phi, B, BBB, lambda, gamma, k, UZ);
  
  Phi_new = MatrixXd::Zero(n2, n);
  //Create thin plate splines, Phi_new by new positions `s_new`
  predictThinPlateMatrix(s_new, s, Phi_new);
  
  Eigen::MatrixXd X1 = Phi_new * UZ.block(0, 0, n, k);
  Eigen::MatrixXd B_new = MatrixXd::Ones(n2, d + 1);
  B_new.rightCols(d) = s_new;
  
  return Rcpp::List::create(Rcpp::Named("X") = X,
                            Rcpp::Named("UZ") = UZ,
                            Rcpp::Named("BBBH") = BBB * Phi,
                            Rcpp::Named("nconst") = nconst,
                            Rcpp::Named("X1") = X1 - B_new * ((BBB * Phi) * UZ.block(0, 0, n, k)));
  
}

//' Internal function: Predict on new locatoins by MRTS method
//' @keywords internal
//' @param s A location matrix
//' @param xobs_diag A matrix of observations
//' @param s_new A new location matrix
//' @param BBBH A matrix for internal computing use
//' @param UZ A matrix for internal computing use
//' @param nconst A A vector of column means
//' @param k A rank
//' @return A list of objects
//' \item{X}{A matrix}
//' \item{UZ}{A matrix}
//' \item{BBBH}{A matrix}
//' \item{nconst}{A vector of column means}
//' \item{X1}{A matrix}
//' @author Wen-Ting Wang
// [[Rcpp::export]]
Rcpp::List predictMrtsRcppWithBasis(const Eigen::Map<Eigen::MatrixXd> s,
                                    const Eigen::Map<Eigen::MatrixXd> xobs_diag,
                                    const Eigen::Map<Eigen::MatrixXd> s_new,
                                    const Eigen::Map<Eigen::MatrixXd> BBBH,
                                    const Eigen::Map<Eigen::MatrixXd> UZ,
                                    const Eigen::Map<Eigen::VectorXd> nconst,
                                    const int k) {
  int n(s.rows()), d(s.cols()), n2(s_new.rows());
  Eigen::MatrixXd Phi_new = MatrixXd::Zero(n2, n);
  //Create thin plate splines, Phi_new by new positions `s_new`
  predictThinPlateMatrix(s_new, s, Phi_new);
  
  Eigen::MatrixXd X1 = Phi_new * UZ.block(0, 0, n, k);
  Eigen::MatrixXd B = MatrixXd::Ones(n2, d + 1);
  B.rightCols(d) = s_new;
  
  return Rcpp::List::create(Rcpp::Named("X") = s,
                            Rcpp::Named("UZ") = UZ,
                            Rcpp::Named("BBBH") = BBBH,
                            Rcpp::Named("nconst") = nconst,
                            Rcpp::Named("X1") = X1 - B * ((BBBH) * UZ.block(0, 0, n, k)));
  
}
