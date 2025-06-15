#include <torch/extension.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace Eigen;
using namespace Spectra;

std::tuple<torch::Tensor, torch::Tensor> decomposeSymmetricMatrix(torch::Tensor M, int ncv, int k) {
  TORCH_CHECK(M.dim() == 2, "Input must be a 2D matrix");
  TORCH_CHECK(M.size(0) == M.size(1), "Matrix must be square");
  TORCH_CHECK(M.dtype() == torch::kFloat64, "Matrix must be of type double");

  const int n = M.size(0);

  // Always move to CPU for Eigen/Spectra computation
  torch::Tensor M_cpu = M.to(torch::kCPU);

  // Convert torch tensor to Eigen matrix
  Eigen::Map<const Eigen::MatrixXd> mat(M_cpu.data_ptr<double>(), n, n);

  // Construct Spectra solver
  Spectra::DenseSymMatProd<double> op(mat);
  Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double>> eigs(&op, k, ncv);

  eigs.init();
  eigs.compute(1000, 1e-10);
  TORCH_CHECK(eigs.info() == Spectra::CompInfo::Successful, "Eigen decomposition did not converge");

  Eigen::VectorXd lambda = eigs.eigenvalues();
  Eigen::MatrixXd gamma = eigs.eigenvectors();

  // Convert Eigen results to torch tensors (on CPU)
  auto lambda_tensor = torch::from_blob(lambda.data(), {k}, torch::TensorOptions().dtype(torch::kFloat64)).clone();
  auto gamma_tensor = torch::from_blob(gamma.data(), {n, k}, torch::TensorOptions().dtype(torch::kFloat64)).clone();

  // Move results back to the original device (GPU if input was on GPU)
  if (M.device().is_cuda()) {
    lambda_tensor = lambda_tensor.to(M.device());
    gamma_tensor = gamma_tensor.to(M.device());
  }

  return std::make_tuple(lambda_tensor, gamma_tensor);
}
