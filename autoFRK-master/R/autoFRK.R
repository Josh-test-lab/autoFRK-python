#'
#' @title Automatic Fixed Rank Kriging
#'
#' @description
#' This function performs resolution adaptive fixed rank kriging based on
#' spatial data observed at one or multiple time points via the following
#' spatial random-effects model:
#' \deqn{z[t]=\mu + G \cdot w[t]+\eta[t]+e[t], w[t] \sim N(0,M), e[t] \sim N(0, s \cdot D); t=1,...,T,}{
#' z[t]=\mu +G*w[t]+\eta[t]+e[t], w[t]~N(0,M); e[t] ~ N(0, s*D); t=1,...,T, }
#' where \eqn{z[t]} is an \emph{n}-vector of (partially) observed data at \emph{n} locations,
#' \eqn{\mu} is an \emph{n}-vector of deterministic mean values,
#' \eqn{D} is a given n by n matrix,
#' \eqn{G} is a given \emph{n} by \emph{K} matrix,
#' \eqn{\eta[t]} is an n-vector of random variables corresponding to a spatial stationary process,
#' and \eqn{w[t]} is a K-vector of unobservable random weights.
#' Parameters are estimated by maximum likelihood in a closed-form expression. The matrix \eqn{G} corresponding to basis functions is
#' given by an ordered class of thin-plate spline functions, with the number of basis functions
#' selected by  Akaike's information criterion.
#' @param data  \emph{n} by \emph{T} data matrix (NA allowed) with
#' \eqn{z[t]} as the \emph{t}-th column.
#' @param loc \emph{n} by \emph{d} matrix of coordinates corresponding to \emph{n} locations.
#' @param mu \emph{n}-vector or scalar for \eqn{\mu}; Default is 0.
#' @param D \emph{n} by \emph{n} matrix (preferably sparse) for the covariance matrix of the measurement errors up to a constant scale.
#' Default is an identity matrix.
#' @param G \emph{n} by \emph{K} matrix of basis function values with each column being a basis function taken values at \code{loc}.
#' Default is NULL, which is automatic determined.
#' @param finescale logical; if \code{TRUE} then a (approximate) stationary finer scale process \eqn{\eta[t]} will be included
#'  based on \code{LatticeKrig} pacakge.
#'  In such a case, only the diagonals of \eqn{D} would be taken into account. Default is \code{FALSE}.
#' @param maxit maximum number of iterations. Default is 50.
#' @param tolerance precision tolerance for convergence check. Default is 0.1^6.
#' @param maxK maximum number of basis functions considered. Default is
#' \eqn{10 \cdot \sqrt{n}} for \emph{n>100} or \emph{n} for \emph{n<=100}.
#' @param Kseq user-specified vector of numbers of basis functions considered. Default is
#' \code{NULL}, which is determined from \code{maxK}.
#' @param method "fast" or " EM"; if "fast" then the missing data are filled in using k-nearest-neighbor imputation;
#' if "EM" then the missing data are taken care by the EM algorithm. Default is "fast".
#' @param n.neighbor number of neighbors to be used in the "fast" imputation method. Default is 3.
#' @param maxknot maximum number of knots to be used in
#' generating basis functions. Default is 5000.
#'
#' @return an object of class \code{FRK} is returned, which is a list of the following components:
#' \item{M}{ML estimate of \eqn{M}.}
#' \item{s}{estimate for the scale parameter of measurement errors.}
#' \item{negloglik}{ negative  log-likelihood.}
#' \item{w}{\emph{K} by \emph{T} matrix with \eqn{w[t]} as the \emph{t}-th column.}
#' \item{V}{\emph{K} by \emph{K} matrix of the prediction error covariance matrix of \eqn{w[t]}.}
#' \item{G}{user specified basis function matrix or an automatically generated \code{mrts} object.}
#' \item{LKobj}{a list from calling \code{LKrig.MLE} in \code{LatticeKrig} package if \code{useLK=TRUE};
#'  otherwise \code{NULL}. See that package for details.}
#' @details The function computes the ML estimate of M using the closed-form expression in Tzeng and Huang (2018).
#' If the user would like to specify
#' a \code{D} other than an identity matrix for a large \emph{n}, it is better to provided via \code{spam} function
#' in \code{spam} package.
#' @export
#' @seealso \code{\link{predict.FRK}}
#' @references
#' - Tzeng, S., & Huang, H.-C. (2018). Resolution Adaptive Fixed Rank Kriging, Technometrics, https://doi.org/10.1080/00401706.2017.1345701.
#' - Nychka D, Hammerling D, Sain S, Lenssen N (2016). “LatticeKrig: Multiresolution Kriging Based on Markov Random Fields.” doi:10.5065/D6HD7T1R <https://doi.org/10.5065/D6HD7T1R>, R package version 8.4, <https://github.com/NCAR/LatticeKrig>.
#' @examples
#' #### generating data from two eigenfunctions
#' originalPar <- par(no.readonly = TRUE)
#' set.seed(0)
#' n <- 150
#' s <- 5
#' grid1 <- grid2 <- seq(0, 1, l = 30)
#' grids <- expand.grid(grid1, grid2)
#' fn <- matrix(0, 900, 2)
#' fn[, 1] <- cos(sqrt((grids[, 1] - 0)^2 + (grids[, 2] - 1)^2) * pi)
#' fn[, 2] <- cos(sqrt((grids[, 1] - 0.75)^2 + (grids[, 2] - 0.25)^2) * 2 * pi)
#'
#' #### single realization simulation example
#' w <- c(rnorm(1, sd = 5), rnorm(1, sd = 3))
#' y <- fn %*% w
#' obs <- sample(900, n)
#' z <- y[obs] + rnorm(n) * sqrt(s)
#' X <- grids[obs, ]
#'
#' #### method1: automatic selection and prediction
#' one.imat <- autoFRK(data = z, loc = X, maxK = 15)
#' yhat <- predict(one.imat, newloc = grids)
#'
#' #### method2: user-specified basis functions
#' G <- mrts(X, 15)
#' Gpred <- predict(G, newx = grids)
#' one.usr <- autoFRK(data = z, loc = X, G = G)
#' yhat2 <- predict(one.usr, newloc = grids, basis = Gpred)
#'
#' require(fields)
#' par(mfrow = c(2, 2))
#' image.plot(matrix(y, 30, 30), main = "True")
#' points(X, cex = 0.5, col = "grey")
#' image.plot(matrix(yhat$pred.value, 30, 30), main = "Predicted")
#' points(X, cex = 0.5, col = "grey")
#' image.plot(matrix(yhat2$pred.value, 30, 30), main = "Predicted (method 2)")
#' points(X, cex = 0.5, col = "grey")
#' plot(yhat$pred.value, yhat2$pred.value, mgp = c(2, 0.5, 0))
#' par(originalPar)
#' #### end of single realization simulation example
#'
#' #### independent multi-realization simulation example
#' set.seed(0)
#' wt <- matrix(0, 2, 20)
#' for (tt in 1:20) wt[, tt] <- c(rnorm(1, sd = 5), rnorm(1, sd = 3))
#' yt <- fn %*% wt
#' obs <- sample(900, n)
#' zt <- yt[obs, ] + matrix(rnorm(n * 20), n, 20) * sqrt(s)
#' X <- grids[obs, ]
#' multi.imat <- autoFRK(data = zt, loc = X, maxK = 15)
#' Gpred <- predict(multi.imat$G, newx = grids)
#'
#' G <- multi.imat$G
#' Mhat <- multi.imat$M
#' dec <- eigen(G %*% Mhat %*% t(G))
#' fhat <- Gpred %*% Mhat %*% t(G) %*% dec$vector[, 1:2]
#' par(mfrow = c(2, 2))
#' image.plot(matrix(fn[, 1], 30, 30), main = "True Eigenfn 1")
#' image.plot(matrix(fn[, 2], 30, 30), main = "True Eigenfn 2")
#' image.plot(matrix(fhat[, 1], 30, 30), main = "Estimated Eigenfn 1")
#' image.plot(matrix(fhat[, 2], 30, 30), main = "Estimated Eigenfn 2")
#' par(originalPar)
#' #### end of independent multi-realization simulation example
#' @author ShengLi Tzeng, Hsin-Cheng Huang and Wen-Ting Wang.
autoFRK <- function(data,
                    loc,
                    mu = 0,
                    D = diag.spam(NROW(data)),
                    G = NULL,
                    finescale = FALSE,
                    maxit = 50,
                    tolerance = 1e-6,
                    maxK = NULL,
                    Kseq = NULL,
                    method = c("fast", "EM"),
                    n.neighbor = 3,
                    maxknot = 5000) {
  method <- match.arg(method)
  data <- data - mu
  if (!is.null(G)) {
    Fk <- G
  } else {
    Fk <- selectBasis(data,
                      loc,
                      D,
                      maxit,
                      tolerance,
                      maxK,
                      Kseq,
                      method,
                      n.neighbor,
                      maxknot,
                      NULL)
  }
  K <- NCOL(Fk)
  if (method == "fast") {
    data <- as.matrix(data)
    for (tt in 1:NCOL(data)) {
      where <- is.na(data[, tt])
      if (sum(where) == 0) {
        next
      }
      cidx <- which(!where)
      nnidx <-
        FNN::get.knnx(loc[cidx,], as.matrix(loc[where,]), k = n.neighbor)
      nnidx <- array(cidx[nnidx$nn.index], dim(nnidx$nn.index))
      nnval <- array((data[, tt])[nnidx], dim(nnidx))
      data[where, tt] <- rowMeans(nnval)
    }
  }
  if (!finescale) {
    obj <- indeMLE(
      data = data,
      Fk = Fk[, 1:K],
      D = D,
      maxit = maxit,
      avgtol = tolerance,
      wSave = TRUE,
      DfromLK = NULL
    )
  } else {
    nu <- .Options$LKinfoSetup$nu
    nlevel <- .Options$LKinfoSetup$nlevel
    a.wght <- .Options$LKinfoSetup$a.wght
    NC <- .Options$LKinfoSetup$NC
    if (is.null(nu)) {
      nu <- 1
    }
    if (is.null(nlevel)) {
      nlevel <- 3
    }
    LK_obj <- initializeLKnFRK(
      data = data,
      location = loc,
      nlevel = nlevel,
      weights = 1 / diag(D),
      n.neighbor = n.neighbor,
      nu = nu
    )
    DnLK <-
      setLKnFRKOption(LK_obj, Fk[, 1:K], nc = NC, a.wght = a.wght)
    DfromLK <- DnLK$DfromLK
    LKobj <- DnLK$LKobj
    Depsilon <- diag.spam(LK_obj$weight[LK_obj$pick],
                          length(LK_obj$weight[LK_obj$pick]))
    obj <- indeMLE(
      data = data,
      Fk = Fk[, 1:K],
      D = D,
      maxit = maxit,
      avgtol = tolerance,
      wSave = TRUE,
      DfromLK = DfromLK,
      vfixed = DnLK$s
    )
  }
  obj$G <- Fk
  if (finescale) {
    obj$LKobj <- LKobj
    attr(obj, "pinfo")$loc <- loc
    attr(obj, "pinfo")$weights <- 1 / diag(D)
  }
  else {
    obj$LKobj <- NULL
  }
  class(obj) <- "FRK"
  return(obj)
}

