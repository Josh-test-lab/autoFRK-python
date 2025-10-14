#'
#' Internal function: Remove attributes of mrts
#'
#' @keywords internal.
#' @param x A mrts object.
#' @param ... Not used directly.
#' @return A matrix object.
#' @method as.matrix mrts
#'
as.matrix.mrts <- function(x, ...) {
  attr(x, "S") <- NULL
  attr(x, "UZ") <- NULL
  attr(x, "Xu") <- NULL
  attr(x, "nconst") <- NULL
  attr(x, "BBBH") <- NULL
  attr(x, "class") <- NULL
  class(x) <- "matrix"
  return(x)
}

#'
#' @title Multi-Resolution Thin-plate Spline Basis Functions
#'
#' @description This function generates multi-resolution thin-plate spline basis functions.
#' The basis functions are (descendingly) ordered
#' in terms of their degrees of smoothness with a higher-order function corresponding
#' to larger-scale features and a lower-order one corresponding to smaller-scale details.
#' They are useful in the spatio-temporal random effects model.
#'
#' @param knot \emph{m} by \emph{d} matrix (\emph{d<=3}) for \emph{m} locations of \emph{d}-dimensional knots as in ordinary splines.
#'          Missing values are not allowed.
#' @param k the number (\emph{<=m}) of basis functions.
#' @param x  \emph{n} by \emph{d} matrix of coordinates corresponding to \emph{n} locations where the values of basis functions to be evaluated.
#' Default is \code{NULL}, which uses the \emph{m} by \emph{d} matrix in \code{knot}.
#' @param maxknot maximum number of knots to be used in generating basis functions. If  \code{maxknot} < \emph{m}, a deterministic subset selection of knots will be used.  For using all knots, set \code{maxknot}>=\emph{m}.
#' @return  An \code{mrts} object is generated. If \code{x=NULL} (default) it returns
#' an \emph{m} by \emph{k} matrix of \emph{k} basis function taken values at knots.
#' With \code{x} given, it returns \emph{n} by \emph{k} matrix for basis functions taken values at \code{x}.
#' @export
#' @seealso \code{\link{predict.mrts}}
#' @examples
#' originalPar <- par(no.readonly = TRUE)
#' knot <- seq(0, 1, l = 30)
#' b <- mrts(knot, 30)
#' x0 <- seq(0, 1, l = 200)
#' bx <- predict(b, x0)
#' par(mfrow = c(5, 6), mar = c(0, 0, 0, 0))
#' for (i in 1:30) {
#'   plot(bx[, i], type = "l", axes = FALSE)
#'   box()
#' }
#' par(originalPar)
#' @references
#' - Tzeng, S., & Huang, H. C. (2018). Resolution Adaptive Fixed Rank Kriging. Technometrics, https://doi.org/10.1080/00401706.2017.1345701.
#' - Tzeng, S., & Huang, H. C. (2015). Multi-Resolution Spatial Random-Effects Models for Irregularly Spaced Data. arXiv preprint arXiv:1504.05659.
#' - Nychka D, Hammerling D, Sain S, Lenssen N (2016). “LatticeKrig: Multiresolution Kriging Based on Markov Random Fields.” doi:10.5065/D6HD7T1R <https://doi.org/10.5065/D6HD7T1R>, R package version 8.4, <https://github.com/NCAR/LatticeKrig>.
#' @author ShengLi Tzeng, Hsin-Cheng Huang and Wen-Ting Wang.
#
mrts <- function(knot, k, x = NULL, maxknot = 5000) {
  is64bit <- length(grep("64-bit", sessionInfo()$platform)) > 0
  if ((!is64bit) & (max(NROW(x), NROW(knot)) > 20000)) {
    stop("Use 64-bit version of R for such a volume of data!")
  }
  if (NCOL(knot) == 1) {
    xobs <- as.matrix(as.double(as.matrix(knot)))
  } else {
    xobs <- apply(knot, 2, as.double)
  }
  Xu <- unique(cbind(xobs))
  if (is.null(x) & length(Xu) != length(xobs)) {
    x <- xobs
  }
  colnames(Xu) <- NULL
  n <- n.Xu <- NROW(Xu)
  ndims <- NCOL(Xu)
  if (k < (ndims + 1)) {
    stop("k-1 can not be smaller than the number of dimensions!")
  }
  if (maxknot < n) {
    bmax <- maxknot
    Xu <- subKnot(Xu, bmax)
    Xu <- as.matrix(Xu)
    if (is.null(x)) {
      x <- knot
    }
    n <- NROW(Xu)
    n.Xu <- n
  }
  xobs_diag <- diag(sqrt(n / (n - 1)) / apply(xobs, 2, sd), ndims)
  if (!is.null(x)) {
    if (NCOL(x) == 1) {
      x <- as.matrix(as.double(as.matrix(x)))
    } else {
      x <- as.matrix(array(as.double(as.matrix(x)), dim(x)))
    }
    if (k - ndims - 1 > 0) {
      result <- predictMrtsRcpp(Xu,
                                xobs_diag,
                                x,
                                k - ndims - 1)
    } else {
      X2 <- scale(Xu, scale = FALSE)
      shift <- colMeans(Xu)
      nconst <- sqrt(diag(t(X2) %*% X2))
      X2 <- cbind(1, t((t(x) - shift) / nconst) * sqrt(n))
      result <- list(X = X2[, 1:k])
      x <- NULL
    }
  }
  else {
    if (k - ndims - 1 > 0) {
      result <- computeMrtsRcpp(Xu, xobs_diag, k - ndims - 1)
    } else {
      X2 <- scale(Xu, scale = FALSE)
      shift <- colMeans(Xu)
      nconst <- sqrt(diag(t(X2) %*% X2))
      X2 <- cbind(1, t((t(Xu) - shift) / nconst) * sqrt(n))
      result <- list(X = X2[, 1:k])
    }
  }
  obj <- result$X
  if (is.null(result$nconst)) {
    X2 <- scale(Xu, scale = FALSE)
    result$nconst <- sqrt(diag(t(X2) %*% X2))
  }
  attr(obj, "UZ") <- result$UZ
  attr(obj, "Xu") <- Xu
  attr(obj, "nconst") <- result$nconst
  attr(obj, "BBBH") <- result$BBBH
  attr(obj, "class") <- c("matrix", "mrts")
  class(obj) <- "mrts"
  if (is.null(x)) {
    return(obj)
  } else {
    shift <- colMeans(attr(obj, "Xu"))
    X2 <- sweep(cbind(x), 2, shift, "-")
    X2 <- cbind(1, sweep(X2, 2, attr(obj, "nconst"), "/"))
    if (k - ndims - 1 > 0) {
      obj0 <- as.matrix(cbind(X2, result$X1))
    } else {
      obj0 <- as.matrix(X2)
    }
    dimnames(obj) <- NULL
    aname <- names(attributes(obj))
    attributes(obj0) <-
      c(attributes(obj0), attributes(obj)[setdiff(aname,
                                                  c("dim", "dimnames"))])
    return(obj0)
  }
}
