
#'
#' @title Predict method for Fixed Rank Kriging
#'
#' @description Predicted values and estimate of standard errors based on an "\code{autoFRK}" model object.
#' @param object  a model object obtained from "\code{autoFRK}".
#' @param obsData a vector with observed data used for prediction.
#' Default is \code{NULL}, which uses the \code{Data} input from \code{object}.
#' @param obsloc a matrix with rows being coordinates of observation locations for \code{obsData}.
#' Only \code{object} using \code{mrts} basis functions can have
#' \code{obsloc} different from the \code{loc} input of \code{object};
#' not applicable for user-specified basis functions.
#' Default is \code{NULL}, which uses the \code{loc} input of \code{object}.
#' @param mu.obs  a vector or scalar for the deterministic mean values at \code{obsloc}. Default is 0.
#' @param newloc a matrix with rows being coordinates of new locations for prediction.
#'  Default is \code{NULL}, which gives prediction at the locations of the observed data.
#' @param basis a matrix with each column being a basis function taken values at \code{newloc}.
#'  It can be omitted if \code{object} was fitted using  default \code{mrts} basis functions.
#' @param mu.new  a vector or scalar for the deterministic mean values at \code{newloc}. Default is 0.
#' @param se.report logical; if \code{TRUE} then the standard error of prediction is reported.
#' @param ... not used but needed for the S3 generic/method compatibility.
#' @export
#' @seealso \code{\link{autoFRK}}
#' @author ShengLi Tzeng, Hsin-Cheng Huang and Wen-Ting Wang.
#' @return A list with the components described below.
#' \item{pred.value}{a matrix with the \emph{(i,t)} element being the predicted value at \emph{i}-th location and time \emph{t}.}
#' \item{se}{a vector with the \emph{i}-th element being the standard error of the predicted value at the \emph{i}-th location.}
#
predict.FRK <-
  function(object,
           obsData = NULL,
           obsloc = NULL,
           mu.obs = 0,
           newloc = obsloc,
           basis = NULL,
           mu.new = 0,
           se.report = FALSE,
           ...) {
    if (is.null(basis)) {
      if (is.null(newloc) & is.null(obsloc)) {
        basis <- object$G
      } else {
        if (!is(object$G, "mrts")) {
          stop(
            "Basis matrix of new locations should be given (unless the model was fitted with mrts bases)!"
          )
        } else {
          if (is.null(newloc)) {
            basis <- object$G
          } else {
            basis <- predict.mrts(object$G, newx = newloc)
          }
        }
      }
    }
    if (NROW(basis) == 1) {
      basis <- as.matrix(t(basis))
    }
    if (is.null(obsloc)) {
      nobs <- NROW(object$G)
    } else {
      nobs <- NROW(obsloc)
    }
    if (!is.null(obsData)) {
      obsData <- as.matrix(obsData - mu.obs)
      if (length(obsData) != nobs) {
        stop("Dimensions of obsloc and obsData are not compatible!")
      }
    }
    if (!is.null(newloc)) {
      if (NROW(basis) != NROW(newloc)) {
        stop("Dimensions of newloc and basis are not compatible!")
      }
    }
    else {
      if (NROW(basis) != NROW(object$G)) {
        stop("Dimensions of obsloc and basis are not compatible!")
      }
    }
    
    if (is.null(object$LKobj)) {
      if (is.null(obsloc) & is.null(obsData)) {
        miss <- attr(object, "missing")
        yhat <- basis %*% object$w
        if (se.report) {
          TT <- NCOL(object$w)
          if (is.null(miss)) {
            se <- sqrt(pmax(0, rowSums((
              basis %*% object$V
            ) * basis)))
            se <- matrix(se, length(se), TT)
          }
          else {
            se <- matrix(NA, NROW(basis), TT)
            pick <- attr(object, "pinfo")$pick
            D0 <- attr(object, "pinfo")$D[pick, pick]
            miss <- (as.matrix(miss$miss) == 1)
            Fk <- object$G[pick,]
            M <- object$M
            dec <- eigen(M)
            for (tt in 1:TT) {
              G <- Fk[!miss[, tt],]
              GM <- G %*% M
              De <- D0[!miss[, tt],!miss[, tt]]
              L <-
                G %*% dec$vector %*% diag.spam(sqrt(pmax(dec$value,
                                                         0)), NROW(M))
              V <- as.matrix(M - t(GM) %*% invCz(object$s *
                                                   De, L, GM))
              se[, tt] <- sqrt(pmax(0, rowSums((
                basis %*%
                  V
              ) * basis)))
            }
          }
        }
      }
      if (!is.null(obsData)) {
        pick <- which(!is.na(obsData))
        if (is.null(obsloc)) {
          De <- attr(object, "pinfo")$D[pick, pick]
          G <- object$G[pick,]
        }
        else {
          De <- diag.spam(length(pick))
          G <-
            predict.mrts(object$G, newx = as.matrix(obsloc)[pick,])
        }
        M <- object$M
        GM <- G %*% M
        dec <- eigen(M)
        L <- G %*% dec$vector %*% diag.spam(sqrt(pmax(dec$value,
                                                      0)), NROW(M))
        yhat <- basis %*% t(GM) %*% invCz(object$s * De, L,
                                          obsData[pick])
        if (se.report) {
          V <- as.matrix(M - t(GM) %*% invCz(object$s *
                                               De, L, GM))
          se <- sqrt(pmax(0, rowSums((basis %*% V) * basis)))
        }
      }
    }
    else {
      if (is.null(obsData)) {
        if (is.null(newloc)) {
          newloc <- attr(object, "pinfo")$loc
        }
        miss <- attr(object, "missing")
        info <- object$LKobj$LKinfo.MLE
        phi0 <- LKrig.basis(newloc, info)
        pinfo <- attr(object, "pinfo")
        yhat <- basis %*% object$w + phi0 %*% pinfo$wlk
        if (se.report) {
          TT <- NCOL(object$w)
          lambda <- object$LKobj$lambda.MLE
          loc <- attr(object, "pinfo")$loc
          pick <- pinfo$pick
          G <- object$G[pick,]
          M <- object$M
          dec <- eigen(M)
          L <- G %*% dec$vector %*% diag.spam(sqrt(pmax(dec$value,
                                                        0)), NROW(M))
          L <- as.matrix(L)
          phi1 <- LKrig.basis(as.matrix(loc)[pick,], info)
          Q <- LKrig.precision(info)
          weight <- pinfo$weights[pick]
          s <- object$s
          phi0P <- phi0 %*% solve(Q)
          if (is.null(miss)) {
            se <- LKpeon(M,
                         s,
                         G,
                         basis,
                         weight,
                         phi1,
                         phi0,
                         Q,
                         lambda,
                         phi0P,
                         L,
                         only.se = TRUE)
            se <- matrix(se, length(se), TT)
          }
          else {
            se <- matrix(NA, NROW(basis), TT)
            miss <- (as.matrix(miss$miss) == 1)
            for (tt in 1:TT) {
              se[, tt] <- LKpeon(M,
                                 s,
                                 G[!miss[, tt],],
                                 basis,
                                 weight[!miss[, tt]],
                                 phi1[!miss[, tt],],
                                 phi0,
                                 Q,
                                 lambda,
                                 phi0P,
                                 L[!miss[, tt],],
                                 only.se = TRUE)
            }
          }
        }
      }
      if (!is.null(obsData)) {
        loc <- attr(object, "pinfo")$loc
        if (is.null(newloc)) {
          newloc <- loc
        }
        pick <- which(!is.na(obsData))
        if (is.null(obsloc)) {
          obsloc <- loc
          De <- attr(object, "pinfo")$D[pick, pick]
          G <- object$G[pick,]
        }
        else {
          G <- predict.mrts(object$G, newx = as.matrix(obsloc)[pick,])
        }
        M <- object$M
        dec <- eigen(M)
        L <- G %*% dec$vector %*% diag.spam(sqrt(pmax(dec$value,
                                                      0)), NROW(M))
        L <- as.matrix(L)
        info <- object$LKobj$LKinfo.MLE
        phi1 <- LKrig.basis(as.matrix(obsloc)[pick,], info)
        Q <- LKrig.precision(info)
        weight <- rep(1, length(pick))
        s <- object$s
        phi0 <- LKrig.basis(newloc, info)
        phi0P <- phi0 %*% solve(Q)
        lambda <- object$LKobj$lambda.MLE
        pred <- LKpeon(
          M,
          s,
          G,
          basis,
          weight,
          phi1,
          phi0,
          Q,
          lambda,
          phi0P,
          L,
          data = obsData[pick,],
          only.wlk = !se.report
        )
        yhat <- basis %*% pred$w + phi0 %*% pred$wlk
        if (se.report) {
          se <- pred$se
        }
      }
    }
    if (!se.report) {
      return(list(pred.value = yhat + mu.new, se = NULL))
    } else {
      return(list(pred.value = yhat + mu.new, se = as.matrix(se)))
    }
  }

#'
#' @title Multi-Resolution Thin-plate Spline Basis Functions
#'
#' @description Evaluate multi-resolution thin-plate spline basis  functions at given locations.
#' This function provides a generic prediction method for \code{mrts} objects,
#' in a similar way as \code{predict.ns} and \code{predict.bs} in \code{splines} package.
#'
#' @param object object produced from calling mrts.
#' @param newx  an \emph{n} by \emph{d} matrix of coordinates corresponding to \emph{n} locations.
#' @param ... not used but needed for the S3 generic/method compatibility.
#' @return an \emph{n} by \emph{k} matrix of the \emph{k} basis function in \code{object} taken values at \code{newx}.
#' @export
#' @seealso \code{\link{mrts}}
#' @author ShengLi Tzeng, Hsin-Cheng Huang and Wen-Ting Wang.
#
predict.mrts <- function(object, newx, ...) {
  if (missing(newx)) {
    return(object)
  }
  Xu <- attr(object, "Xu")
  n <- NROW(Xu)
  xobs_diag <- diag(sqrt(n / (n - 1)) / apply(Xu, 2, sd), ncol(Xu))
  ndims <- NCOL(Xu)
  k <- NCOL(object)
  x0 <- matrix(as.matrix(newx), ncol = ndims)
  kstar <- (k - ndims - 1)
  shift <- colMeans(attr(object, "Xu"))
  X2 <- sweep(cbind(x0), 2, shift, "-")
  X2 <- cbind(1, sweep(X2, 2, attr(object, "nconst"), "/"))
  
  if (kstar > 0) {
    X1 <- predictMrtsRcppWithBasis(
      Xu,
      xobs_diag,
      x0,
      attr(object, "BBBH"),
      attr(object, "UZ"),
      attr(object, "nconst"),
      k
    )$X1
    X1 <- X1[, 1:kstar]
    return(as.matrix(cbind(X2, X1)))
  } else {
    return(as.matrix(X2))
  }
}