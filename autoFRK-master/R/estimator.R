#'
#' Internal function: EM0miss
#'
#' @keywords internal.
#' @param Fk An \emph{n} by \emph{K} matrix of basis function values with
#'  each column being a basis function taken values at \code{loc}.
#' @param data  An \emph{n} by \emph{T} data matrix (NA allowed) with
#' \eqn{z[t]} as the \emph{t}-th column.
#' @param Depsilon An \emph{n} by \emph{n} diagonal matrix.
#' @param maxit An integer for the maximum number of iterations.
#' @param avgtol A numeric for average tolerance.
#' @param wSave A logic.
#' @param external A logic.
#' @param DfromLK An \emph{n} by \emph{n} matrix
#' @param vfixed A numeric.
#' @param verbose A logic. Print a useful information.
#' @return A list.
#'
EM0miss <- function(Fk,
                    data,
                    Depsilon,
                    maxit,
                    avgtol,
                    wSave = FALSE,
                    external = FALSE,
                    DfromLK = NULL,
                    vfixed = NULL,
                    verbose = TRUE) {
  if ("numeric" %in% class(data))
    data <- as.matrix(data)
  O <- !is.na(data)
  TT <- NCOL(data)
  ncol_Fk <- NCOL(Fk)
  tmpdir <- tempfile()
  dir.create(tmpdir)
  ftmp <- paste(tmpdir, 1:TT, sep = "/")
  oldfile <- paste(tmpdir, "old_par.Rdata", sep = "/")
  
  ziDz <- rep(NA, TT)
  ziDB <- matrix(NA, TT, ncol_Fk)
  db <- list()
  D <- toSparseMatrix(Depsilon)
  iD <- solve(D)
  diagD <- isDiagonal(D)
  
  if (!is.null(DfromLK)) {
    pick <- DfromLK$pick
    if (is.null(pick))
      pick <- 1:length(DfromLK$weights)
    weight <- DfromLK$weights[pick]
    DfromLK$wX <- DfromLK$wX[pick, ]
    wwX <- diag.spam(sqrt(weight)) %*% DfromLK$wX
    lQ <- DfromLK$lambda * DfromLK$Q
  }
  
  for (tt in 1:TT) {
    if (!is.null(DfromLK)) {
      iDt <- NULL
      if (sum(O[, tt]) == NROW(O)) {
        wXiG <- wwX %*% solve(DfromLK$G)
      } else {
        G <- t(DfromLK$wX[O[, tt], ]) %*% DfromLK$wX[O[, tt], ] + lQ
        wXiG <- wwX[O[, tt], ] %*% solve(G)
      }
      Bt <- as.matrix(Fk[O[, tt], ])
      if (NCOL(Bt) == 1)
        Bt <- t(Bt)
      iDBt <-
        as.matrix(weight[O[, tt]] * Bt - wXiG %*% (t(wwX[O[, tt], ]) %*% Bt))
      zt <- data[O[, tt], tt]
      ziDz[tt] <-
        sum(zt * as.vector(weight[O[, tt]] * zt - wXiG %*% (t(wwX[O[, tt], ]) %*% zt)))
      ziDB[tt, ] <- t(zt) %*% iDBt
      BiDBt <- t(Bt) %*% iDBt
    } else {
      if (!diagD) {
        iDt <- solve(D[O[, tt], O[, tt]])
      } else {
        iDt <- iD[O[, tt], O[, tt]]
      }
      Bt <- Fk[O[, tt], ]
      if (NCOL(Bt) == 1)
        Bt <- t(Bt)
      iDBt <- as.matrix(iDt %*% Bt)
      zt <- data[O[, tt], tt]
      ziDz[tt] <- sum(zt * as.vector(iDt %*% zt))
      ziDB[tt, ] <- t(zt) %*% iDBt
      BiDBt <- t(Bt) %*% iDBt
    }
    
    if (external) {
      db[[tt]] <- dumpObjects(iDBt,
                              zt,
                              BiDBt,
                              external,
                              oldfile,
                              dbName = ftmp[tt])
    } else {
      db[[tt]] <- list(
        iDBt = iDBt,
        zt = zt,
        BiDBt = BiDBt,
        external = external,
        oldfile = oldfile
      )
    }
  }
  # gc
  rm(iDt, Bt, iDBt, zt, BiDBt)
  gc()
  
  dif <- Inf
  cnt <- 0
  Z0 <- data
  Z0[is.na(Z0)] <- 0
  old <- cMLEimat(Fk, Z0, s = 0, wSave = TRUE)
  if (is.null(vfixed))
    old$s <- old$v
  else
    old$s <- vfixed
  old$M <- convertToPositiveDefinite(old$M)
  Ptt1 <- old$M
  if (external)
    save(old, Ptt1, file = oldfile)
  inv <- MASS::ginv
  
  while ((dif > (avgtol * (100 * ncol_Fk ^ 2))) && (cnt < maxit)) {
    etatt <- matrix(0, ncol_Fk, TT)
    sumPtt <- 0
    s1 <- rep(0, TT)
    if (external)
      load(oldfile)
    for (tt in 1:TT) {
      s1.eta.P <- with(db[[tt]], {
        ginv_Ptt1 <- MASS::ginv(convertToPositiveDefinite(Ptt1))
        iP <- convertToPositiveDefinite(ginv_Ptt1 + BiDBt / old$s)
        Ptt <- solve(iP)
        Gt <- as.matrix(Ptt %*% t(iDBt) / old$s)
        eta <- c(0 + Gt %*% zt)
        s1kk <- diag(BiDBt %*% (eta %*% t(eta) + Ptt))
        rbind(s1kk, eta, Ptt)
      })
      sumPtt <- sumPtt + s1.eta.P[-c(1:2), ]
      etatt[, tt] <- s1.eta.P[2, ]
      s1[tt] <- sum(s1.eta.P[1, ])
    }
    if (is.null(vfixed)) {
      s <-  max((sum(ziDz) - 2 * sum(ziDB * t(etatt)) + sum(s1)) / sum(O),
                1e-8)
      new <- list(M = (etatt %*% t(etatt) + sumPtt) / TT,
                  s = s)
    } else {
      new <- list(M = (etatt %*% t(etatt) + sumPtt) / TT,
                  s = vfixed)
    }
    new$M <- (new$M + t(new$M)) / 2
    dif <- sum(abs(new$M - old$M)) + abs(new$s - old$s)
    cnt <- cnt + 1
    old <- new
    Ptt1 <- old$M
    if (external)
      save(old, Ptt1, file = oldfile)
  }
  if (verbose)
    cat("Number of iteration: ", cnt, "\n")
  unlink(tmpdir, recursive = TRUE)
  n2loglik <- computeLikelihood(data, Fk, new$M, new$s, Depsilon)
  
  if (!wSave) {
    return(list(
      M = new$M,
      s = new$s,
      negloglik = n2loglik
    ))
  } else if (!is.null(DfromLK)) {
    out <- list(
      M = new$M,
      s = new$s,
      negloglik = n2loglik,
      w = etatt,
      V = new$M - etatt %*% t(etatt) / TT
    )
    dec <- eigen(new$M)
    L <- Fk %*% dec$vector %*% diag(sqrt(pmax(dec$value,
                                              0)))
    weight <- DfromLK$weights[pick]
    wlk <- matrix(NA, NROW(lQ), TT)
    for (tt in 1:TT) {
      if (sum(O[, tt]) == NROW(O)) {
        wXiG <- wwX %*% solve(DfromLK$G)
      } else {
        G <- t(DfromLK$wX[O[, tt], ]) %*% DfromLK$wX[O[, tt], ] + lQ
        wXiG <- wwX[O[, tt], ] %*% solve(G)
      }
      dat <- data[O[, tt], tt]
      Lt <- L[O[, tt], ]
      iDL <-
        weight[O[, tt]] * Lt - wXiG %*% (t(wwX[O[, tt], ]) %*% Lt)
      itmp <- solve(diag(1, NCOL(L)) + t(Lt) %*% iDL / out$s)
      iiLiD <- itmp %*% t(iDL / out$s)
      wlk[, tt] <-
        t(wXiG) %*% dat - t(wXiG) %*% Lt %*% (iiLiD %*% dat)
    }
    attr(out, "pinfo") <- list(wlk = wlk, pick = pick)
    attr(out, "missing") <- list(miss = toSparseMatrix(1 - O),
                                 maxit = maxit,
                                 avgtol = avgtol)
    return(out)
  } else {
    out <- list(
      M = as.matrix(new$M),
      s = new$s,
      negloglik = n2loglik,
      w = etatt,
      V = new$M - etatt %*% t(etatt) / TT
    )
    attr(out, "missing") <- list(miss = toSparseMatrix(1 - O),
                                 maxit = maxit,
                                 avgtol = avgtol)
    return(out)
  }
}

#'
#' Internal function: indeMLE
#'
#' @keywords internal.
#' @param data  An \emph{n} by \emph{T} data matrix (NA allowed) with
#' \eqn{z[t]} as the \emph{t}-th column.
#' @param Fk An \emph{n} by \emph{K} matrix of basis function values with
#'  each column being a basis function taken values at \code{loc}.
#' @param D An \emph{n} by \emph{n} diagonal matrix.
#' @param maxit An integer for the maximum number of iterations.
#' @param avgtol A numeric for average tolerance.
#' @param wSave A logic.
#' @param DfromLK An \emph{n} by \emph{n} matrix
#' @param vfixed A numeric.
#' @param verbose A logic. Print a useful information.
#' @return A list.
#'
indeMLE <- function(data,
                    Fk,
                    D = diag.spam(NROW(data)),
                    maxit = 50,
                    avgtol = 1e-6,
                    wSave = FALSE,
                    DfromLK = NULL,
                    vfixed = NULL,
                    verbose = TRUE) {
  withNA <- sum(is.na(data)) > 0
  if (is(data, "vector")) {
    data <- as.matrix(data)
  }
  TT <- NCOL(data)
  empty <- apply(!is.na(data), 2, sum) == 0
  notempty <- which(!empty)
  if (sum(empty) > 0) {
    data <- as.matrix(data[, notempty])
  }
  if (is(data, "vector")) {
    data <- as.matrix(data)
  }
  del <- which(rowSums(as.matrix(!is.na(data))) == 0)
  pick <- 1:NROW(data)
  if (!isDiagonal(D)) {
    D0 <- toSparseMatrix(D)
  } else {
    D0 <- diag.spam(diag(D), NROW(data))
  }
  
  if (withNA && (length(del) > 0)) {
    pick <- pick[-del]
    data <- data[-del, ]
    Fk <- Fk[-del, ]
    if (!isDiagonal(D)) {
      D <- D[-del, -del]
    } else {
      D <- diag.spam(diag(D)[-del], NROW(data))
    }
    withNA <- sum(is.na(data)) > 0
  }
  N <- NROW(data)
  K <- NCOL(Fk)
  Depsilon <- toSparseMatrix(D)
  isimat <- isDiagonal(D) * (sum(abs(rep(mean(
    diag(D)
  ), N) -
    diag(Depsilon))) < .Machine$double.eps)
  
  if (!withNA) {
    if (isimat & is.null(DfromLK)) {
      if (!is.null(.Options$sigma_FRK)) {
        sigma <- .Options$sigma_FRK
      } else {
        sigma <- 0
      }
      if (NCOL(data) == 1) {
        data <- as.matrix(data)
      }
      out <- cMLEimat(Fk, data, s = sigma, wSave)
      if (!is.null(out$v)) {
        if (sigma == 0) {
          out$s <- out$v
        } else {
          out$s <- sigma
        }
        out <- out[-which(names(out) == "v")]
      }
      if (wSave) {
        w <- matrix(0, K, TT)
        w[, notempty] <- out$w
        out$w <- w
        attr(out, "pinfo") <- list(D = D0, pick = pick)
      }
      return(out)
    }
    else {
      if (is.null(DfromLK)) {
        out <- cMLEsp(Fk, data, Depsilon, wSave)
        if (wSave) {
          w <- matrix(0, K, TT)
          w[, notempty] <- out$w
          out$w <- w
          attr(out, "pinfo") <- list(D = D0, pick = pick)
        }
        return(out)
      }
      else {
        out <- cMLElk(Fk, data, Depsilon, wSave, DfromLK,
                      vfixed)
        if (wSave) {
          w <- matrix(0, K, TT)
          w[, notempty] <- out$w
          out$w <- w
        }
        return(out)
      }
    }
  }
  else {
    out <- EM0miss(Fk,
                   data,
                   Depsilon,
                   maxit,
                   avgtol,
                   wSave,
                   external = FALSE,
                   DfromLK,
                   vfixed,
                   verbose)
    if (wSave) {
      w <- matrix(0, K, TT)
      w[, notempty] <- out$w
      out$w <- w
      if (is.null(DfromLK)) {
        attr(out, "pinfo") <- list(D = D0, pick = pick)
      }
    }
    return(out)
  }
}


#'
#' Internal function: maximum likelihood estimate with the likelihood
#'
#' @keywords internal.
#' @param Fk An \emph{n} by \emph{K} matrix of basis function values with
#'  each column being a basis function taken values at \code{loc}.
#' @param num_columns A positive numeric.
#' @param sample_covariance_trace A positive numeric.
#' @param inverse_square_root_matrix A matrix.
#' @param matrix_JSJ A multiplication matrix
#' @param s A numeric.
#' @param ldet A numeric. A log determinant.
#' @param wSave A logic.
#' @param onlylogLike A logic.
#' @param vfixed A numeric.
#' @return A numeric.
#'
cMLE <- function(Fk,
                 num_columns,
                 sample_covariance_trace,
                 inverse_square_root_matrix,
                 matrix_JSJ,
                 s = 0,
                 ldet = 0,
                 wSave = FALSE,
                 onlylogLike = !wSave,
                 vfixed = NULL) {
  nrow_Fk <- nrow(Fk)
  
  likelihood_object <- computeNegativeLikelihood(
    nrow_Fk = nrow_Fk,
    ncol_Fk = ncol(Fk),
    s = s,
    p = num_columns,
    matrix_JSJ = matrix_JSJ,
    sample_covariance_trace = sample_covariance_trace,
    vfixed = vfixed,
    ldet = ldet
  )
  negative_log_likelihood <-
    likelihood_object$negative_log_likelihood
  if (onlylogLike) {
    return(list(negloglik = negative_log_likelihood))
  }
  
  P <- likelihood_object$P
  d_hat <- likelihood_object$d_hat
  
  M <-
    inverse_square_root_matrix %*% P %*% (d_hat * t(P)) %*% inverse_square_root_matrix
  dimnames(M) <- NULL
  if (!wSave) {
    L <- NULL
  } else {
    if (d_hat[1] != 0) {
      L <- Fk %*% ((sqrt(d_hat) * t(P)) %*% inverse_square_root_matrix)
      L <- as.matrix(L[, d_hat > 0])
    } else {
      L <- matrix(0, nrow_Fk, 1)
    }
  }
  
  return(list(
    v = likelihood_object$v,
    M = M,
    s = s,
    negloglik = negative_log_likelihood,
    L = L
  ))
}


#'
#' Internal function: cMLEimat
#'
#' @keywords internal.
#' @param Fk An \emph{n} by \emph{K} matrix of basis function values with
#'  each column being a basis function taken values at \code{loc}.
#' @param data  An \emph{n} by \emph{T} data matrix (NA allowed) with
#' \eqn{z[t]} as the \emph{t}-th column.
#' @param s A positive numeric.
#' @param wSave A logic.
#' @param S An \emph{n} by \emph{n} matrix
#' @param onlylogLike A logic.
#' @return A list.
#'
cMLEimat <- function(Fk,
                     data,
                     s,
                     wSave = FALSE,
                     S = NULL,
                     onlylogLike = !wSave) {
  data <- as.matrix(data)
  Fk <- as.matrix(Fk)
  
  num_columns <- NCOL(data)
  nrow_Fk <- nrow(Fk)
  ncol_Fk <- ncol(Fk)
  
  projection <- computeProjectionMatrix(Fk, Fk, data, S)
  inverse_square_root_matrix <-
    projection$inverse_square_root_matrix
  matrix_JSJ <- projection$matrix_JSJ
  
  sample_covariance_trace <- sum(rowSums(data ^ 2)) / num_columns
  
  likelihood_object <- computeNegativeLikelihood(
    nrow_Fk = nrow_Fk,
    ncol_Fk = ncol_Fk,
    s = s,
    p = num_columns,
    matrix_JSJ = matrix_JSJ,
    sample_covariance_trace = sample_covariance_trace
  )
  negative_log_likelihood <-
    likelihood_object$negative_log_likelihood
  if (onlylogLike) {
    return(list(negloglik = negative_log_likelihood))
  }
  
  P <- likelihood_object$P
  d_hat <- likelihood_object$d_hat
  v <- likelihood_object$v
  M <-
    inverse_square_root_matrix %*% P %*% (d_hat * t(P)) %*% inverse_square_root_matrix
  dimnames(M) <- NULL
  if (!wSave) {
    return(list(
      v = v,
      M = M,
      s = s,
      negloglik = negative_log_likelihood
    ))
  } else {
    L <- Fk %*% t((sqrt(d_hat) * t(P)) %*% inverse_square_root_matrix)
    if (ncol_Fk > 2) {
      reduced_columns <- c(1, which(d_hat[2:ncol_Fk] > 0))
    } else {
      reduced_columns <- ncol_Fk
    }
    L <- L[, reduced_columns]
    invD <- rep(1, nrow_Fk) / (s + v)
    iDZ <- invD * data
    right <-
      L %*% (solve(diag(1, NCOL(L)) + t(L) %*% (invD * L)) %*% (t(L) %*% iDZ))
    INVtZ <- iDZ - invD * right
    etatt <- as.matrix(M %*% t(Fk) %*% INVtZ)
    GM <- Fk %*% M
    V <- as.matrix(M - t(GM) %*% invCz((s + v) * diag.spam(nrow_Fk),
                                       L, GM))
    return(list(
      v = v,
      M = M,
      s = s,
      negloglik = negative_log_likelihood,
      w = etatt,
      V = V
    ))
  }
}

#'
#' Internal function: cMLElk
#'
#' @keywords internal.
#' @param Fk An \emph{n} by \emph{K} matrix of basis function values with
#'  each column being a basis function taken values at \code{loc}.
#' @param data  An \emph{n} by \emph{T} data matrix (NA allowed) with
#' \eqn{z[t]} as the \emph{t}-th column.
#' @param Depsilon An \emph{n} by \emph{n} diagonal matrix.
#' @param wSave A logic.
#' @param DfromLK An \emph{n} by \emph{n} matrix
#' @param vfixed A numeric.
#' @return A list.
#'
cMLElk <- function(Fk,
                   data,
                   Depsilon,
                   wSave = FALSE,
                   DfromLK,
                   vfixed = NULL) {
  num_columns <- NCOL(data)
  N <- NROW(data)
  lambda <- DfromLK$lambda
  pick <- DfromLK$pick
  wX <- DfromLK$wX
  weight <- DfromLK$weights
  if (length(pick) < dim(wX)[1]) {
    wX <- wX[pick, ]
    weight <- weight[pick]
  }
  G <- t(wX) %*% wX + lambda * DfromLK$Q
  wwX <- diag.spam(sqrt(weight)) %*% wX
  wXiG <- (wwX) %*% solve(G)
  iDFk <- weight * Fk - wXiG %*% (t(wwX) %*% as.matrix(Fk))
  projection <- computeProjectionMatrix(Fk, iDFk, data)
  inverse_square_root_matrix <-
    projection$inverse_square_root_matrix
  matrix_JSJ <- projection$matrix_JSJ
  iDZ <- weight * data - wXiG %*% (t(wwX) %*% as.matrix(data))
  trS <- sum(rowSums(as.matrix(iDZ) * data)) / num_columns
  ldetD <- 
    -nrow(DfromLK$Q) * log(lambda) + logDeterminant(G) - logDeterminant(DfromLK$Q) -
    sum(log(weight))
  out <- cMLE(
    Fk,
    num_columns,
    trS,
    inverse_square_root_matrix,
    matrix_JSJ,
    s = 0,
    ldet = as.vector(ldetD),
    wSave = TRUE,
    onlylogLike = FALSE,
    vfixed = vfixed
  )
  L <- out$L
  out$s <- out$v
  out <- out[-which(names(out) == "v")]
  out <- out[-which(names(out) == "L")]
  if (!wSave) {
    return(out)
  } else {
    iDL <- weight * L - wXiG %*% (t(wwX) %*% L)
    itmp <- solve(diag(1, NCOL(L)) + t(L) %*% iDL / out$s)
    iiLiD <- itmp %*% t(iDL / out$s)
    MFiS11 <-
      out$M %*% t(iDFk) / out$s - ((out$M %*% t(iDFk / out$s)) %*%
                                     L) %*% iiLiD
    out$w <- MFiS11 %*% data
    out$V <- MFiS11 %*% (Fk %*% out$M)
    wlk <- t(wXiG) %*% data - t(wXiG) %*% L %*% (iiLiD %*%
                                                   data)
    ihL <- chol(itmp) %*% t(L)
    attr(out, "pinfo") <- list(wlk = wlk, pick = pick)
    return(out)
  }
}

#'
#' Internal function: cMLEsp
#'
#' @keywords internal.
#' @param Fk An \emph{n} by \emph{K} matrix of basis function values with
#'  each column being a basis function taken values at \code{loc}.
#' @param data  An \emph{n} by \emph{T} data matrix (NA allowed) with
#' \eqn{z[t]} as the \emph{t}-th column.
#' @param Depsilon An \emph{n} by \emph{n} diagonal matrix.
#' @param wSave A logic.
#' @return A list.
#'
cMLEsp <- function(Fk,
                   data,
                   Depsilon,
                   wSave = FALSE) {
  if ("numeric" %in% class(data))
    data <- as.matrix(data)
  De <- toSparseMatrix(Depsilon)
  iD <- solve(De)
  ldetD <- logDeterminant(De)
  iDFk <- iD %*% Fk
  num_columns <- NCOL(data)
  
  projection <- computeProjectionMatrix(Fk, iDFk, data)
  inverse_square_root_matrix <-
    projection$inverse_square_root_matrix
  matrix_JSJ <- projection$matrix_JSJ
  
  trS <- sum(rowSums(as.matrix(iD %*% data) * data)) / num_columns
  out <- cMLE(
    Fk,
    num_columns,
    trS,
    inverse_square_root_matrix,
    matrix_JSJ,
    s = 0,
    ldet = ldetD,
    wSave
  )
  if (wSave) {
    L <- as.matrix(out$L)
    invD <- iD / (out$s + out$v)
    iDZ <- invD %*% data
    right0 <- L %*% solve(diag(1, NCOL(L)) + t(L) %*% (invD %*%
                                                         L))
    INVtZ <- iDZ - invD %*% right0 %*% (t(L) %*% iDZ)
    etatt <- out$M %*% t(Fk) %*% INVtZ
    out$w <- as.matrix(etatt)
    GM <- Fk %*% out$M
    iDGM <- invD %*% GM
    out$V <- as.matrix(out$M - t(GM) %*% (iDGM - invD %*%
                                            right0 %*% (t(L) %*% iDGM)))
  }
  out$s <- out$v
  out <- out[-which(names(out) == "v")]
  out <- out[-which(names(out) == "L")]
  return(out)
}
