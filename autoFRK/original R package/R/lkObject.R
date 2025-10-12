#'
#' Internal function: initializeLKnFRK
#'
#' @keywords internal
#' @param data  \emph{n} by \emph{T} data matrix (NA allowed) with
#' \eqn{z[t]} as the \emph{t}-th column.
#' @param location \emph{n} by \emph{d} matrix of coordinates corresponding to \emph{n} locations.
#' @param nlelve An integer.
#' @param weights An \emph{n} by \emph{n} diagonal matrix.
#' @param n.neighbor number of neighbors to be used in the "fast" imputation method. Default is 3.
#' @param nu An integer.
#' @return list
#'
initializeLKnFRK <-
  function(data,
           location,
           nlevel = 3,
           weights = NULL,
           n.neighbor = 3,
           nu = 1) {
    if ("numeric" %in% class(data))
      data <- as.matrix(data)
    empty <- apply(!is.na(data), 2, sum) == 0
    if (sum(empty) > 0)
      data <- data[, which(!empty)]
    location <- as.matrix(location)
    N <- NROW(data)
    d <- NCOL(location)
    nas <- sum(is.na(data))
    del <- which(rowSums(as.matrix(!is.na(data))) == 0)
    pick <- 1:N
    x <- as.matrix(location)
    if (length(del) > 0) {
      data <- data[-del,]
      x <- x[-del,]
      pick <- (1:N)[-del]
    }
    nas <- sum(is.na(data))
    if (nas > 0) {
      for (tt in 1:NCOL(data)) {
        where <- is.na(data[, tt])
        if (sum(where) == 0) {
          next
        }
        cidx <- which(!where)
        nnidx <-
          FNN::get.knnx(x[cidx,], as.matrix(x[where,]), k = n.neighbor)
        nnidx <- array(cidx[nnidx$nn.index], dim(nnidx$nn.index))
        nnval <- array((data[, tt])[nnidx], dim(nnidx))
        data[where, tt] <- rowMeans(nnval)
      }
    }
    z <- as.matrix(data)
    d <- NCOL(x)
    gtype <-
      ifelse(d == 1, "LKInterval", ifelse(d == 2, "LKRectangle", "LKBox"))
    thetaL <- 2 ^ (-1 * (1:nlevel))
    alpha <- thetaL ^ (2 * nu)
    alpha <- alpha / sum(alpha)
    n <- NROW(x)
    if (is.null(weights))
      weights <- rep(1, NROW(z))
    
    return(
      list(
        x = x,
        z = z,
        n = n,
        alpha = alpha,
        gtype = gtype,
        weights = weights,
        nlevel = nlevel,
        location = location,
        pick = pick
      )
    )
  }

#'
#' Internal function: setLKnFRKOption
#'
#' @keywords internal
#' @param LK_obj  A list produced from `initializeLKnFRK`.
#' @param Fk An \emph{n} by \emph{K} matrix of basis function values with
#'  each column being a basis function taken values at \code{location}.
#' @param nc A numeric made by `setNC`.
#' @param Ks An integer.
#' @param a.wght A numeric.
#' @return list
#'
setLKnFRKOption <-
  function(LK_obj,
           Fk,
           nc = NULL,
           Ks = NCOL(Fk),
           a.wght = NULL) {
    x <- LK_obj$x
    z <- LK_obj$z
    alpha <- LK_obj$alpha
    alpha <- alpha / sum(alpha)
    gtype <- LK_obj$gtype
    weights <- LK_obj$weights
    if (length(LK_obj$pick) < length(weights))
      weights <- weights[LK_obj$pick]
    nlevel <- LK_obj$nlevel
    TT <- NCOL(z)
    Fk <- Fk[LK_obj$pick,]
    
    if (is.null(nc))
      nc <- setNC(z, x, nlevel)
    if (is.null(a.wght))
      a.wght <- 2 * NCOL(x) + 0.01
    
    info <- LKrigSetup(
      x = x,
      a.wght = a.wght,
      nlevel = nlevel,
      NC = nc,
      alpha = alpha,
      LKGeometry = gtype,
      lambda = 1
    )
    
    location <- x
    phi <- LKrig.basis(location, info)
    w <- diag.spam(sqrt(weights))
    wX <- w %*% phi
    wwX <- w %*% wX
    XwX <- t(wX) %*% wX
    
    iniLike <- function(par, data = z, full = FALSE) {
      lambda <- exp(par)
      G <- XwX + lambda * Qini
      wXiG <- (wwX) %*% solve(G)
      iDFk <- weights * Fk - wXiG %*% (t(wwX) %*% as.matrix(Fk))
      iDZ <- weights * data - wXiG %*% (t(wwX) %*% as.matrix(data))
      ldetD <- -nrow(Qini) * log(lambda) + logDeterminant(G)
      ldetD <- as.vector(ldetD)
      trS <- sum(rowSums(as.matrix(iDZ) * data)) / TT
      half <- getInverseSquareRootMatrix(Fk, iDFk)
      ihFiD <- half %*% t(iDFk)
      LSL <- tcrossprod(ihFiD %*% data) / TT
      if (!full) {
        cMLE(
          Fk,
          TT,
          trS,
          half,
          LSL,
          s = 0,
          ldet = ldetD,
          wSave = FALSE
        )$negloglik
      } else {
        llike <- ldetD - logDeterminant(Qini) - sum(log(weights))
        cMLE(
          Fk,
          TT,
          trS,
          half,
          LSL,
          s = 0,
          ldet = llike,
          wSave = TRUE,
          onlylogLike = FALSE,
          vfixed = NULL
        )
      }
    }
    
    Qini <- LKrig.precision(info)
    sol <-
      optimize(iniLike, c(-16, 16), tol = .Machine$double.eps ^ 0.025)
    lambda.MLE <- sol$minimum
    out <- iniLike(sol$minimum, z, full = TRUE)
    llike <- out$negloglik
    info.MLE <- LKrigSetup(
      x = x,
      a.wght = a.wght,
      nlevel = nlevel,
      NC = nc,
      alpha = alpha,
      LKGeometry = gtype,
      lambda = lambda.MLE
    )
    info.MLE$llike <- llike
    info.MLE$time <- NA
    Q <- LKrig.precision(info.MLE)
    G <- t(wX) %*% wX + info.MLE$lambda * Q
    
    return(list(
      DfromLK = list(
        Q = Q,
        weights = weights,
        wX = wX,
        G = G,
        lambda = info.MLE$lambda,
        pick = LK_obj$pick
      ),
      s = out$v,
      LKobj = list(
        summary = NULL,
        par.grid = NULL,
        LKinfo.MLE = info.MLE,
        lnLike.eval = NULL,
        lambda.MLE = info.MLE$lambda,
        call = NA,
        taskID = NULL
      )
    ))
  }


#'
#' Internal function: set 'nc' for LKrigInfo
#'
#' @keywords internal
#' @param z A matrix
#' @param location A location matrix
#' @param nlelve An integer
#' @return numeric
#'
setNC <- function(z, location, nlevel) {
  if(!is.matrix(z))
    z <- matrix(z)
  location_dim <- NCOL(location)
  n <- nrow(z)
  a <- sum(2 ^ (location_dim * (0:(nlevel - 1))))
  nc_estimate <- round((n / a) ^ (1 / location_dim))
  return(max(4, nc_estimate))
}


#'
#' Internal function: A wrapper for LatticeKrig::LKrigSetup
#'
#' @keywords internal
#' @param x Spatial locations that define the spatial domain for prediction.
#' @param nlevel Number of levels in multi-resolution.
#' @param alpha A vector of length nlevel with the relative variances for the different multi- resolution levels.
#' @param a.wght The correlation range in the SAR model.
#' @param NC The maximum number of lattice grid points for a spatial coordinate and at the coarsest level of resolution.
#' @param lambda A smoothing parameter.
#' @param LKGeometry A text string that gives the names of the model geometry.
#' @param ... Not used directly
#' @return list
#'
setUpKrigInfo <- function(x = NULL,
                          nlevel = NULL,
                          alpha = NA,
                          a.wght = NA,
                          NC = NULL,
                          lambda = NA,
                          LKGeometry = "LKRectangle",
                          ...) {
  setupArgs <- list(...)
  LKinfo <- list(
    x = x,
    nlevel = nlevel,
    alpha = alpha,
    alphaObject = NULL,
    a.wght = a.wght,
    a.wghtObject = NULL,
    NC = NC,
    NC.buffer = NULL,
    nu = NULL,
    normalize = TRUE,
    lambda = lambda,
    sigma = NA,
    rho = NA,
    rho.object = NULL,
    LKGeometry = LKGeometry,
    distance.type = "Euclidean",
    BasisFunction = "WendlandFunction",
    overlap = 2.5,
    V = NULL,
    BasisType = "Radial",
    fixedFunction = "LKrigDefaultFixedFunction",
    fixedFunctionArgs = list(m = 2),
    collapseFixedEffect = FALSE,
    max.points = NULL,
    mean.neighbor = 50,
    choleskyMemory = NULL,
    setupArgs = setupArgs,
    dense = FALSE
  )
  LKinfo$basisInfo <- list(
    BasisType = "Radial",
    BasisFunction = "WendlandFunction",
    overlap = 2.5,
    max.points = NULL,
    mean.neighbor = 50,
    V = NULL
  )
  
  class(LKinfo) <- c("LKinfo", LKGeometry)
  LKinfo <- setDefaultsLKinfo(LKinfo)
  
  LKinfo$latticeInfo <- do.call("LKrigSetupLattice",
                                c(list(object = LKinfo, verbose = FALSE),
                                  setupArgs))
  
  if (LKGeometry == "LKRectangle") {
    LKinfo$a.wght <- setRectangleAwght(LKinfo)
  } else {
    LKinfo$a.wght <- setDefaultAwght(LKinfo)
  }
  LKinfo$call <- NULL
  LKinfoCheck(LKinfo)
  
  return(LKinfo)
}

#'
#' Internal function: A wrapper of LatticeKrig::LKrigSetupAwght
#'
#' @keywords internal
#' @param LKinfo LKinfo object
#' @return list
#'
setDefaultAwght <- function(LKinfo) {
  a_wght <- LKinfo$a.wght
  nlevel <- LKinfo$nlevel
  isotropic <- length(a_wght) == 1
  
  if (!is.list(a_wght)) {
    if (nlevel == 1) {
      a_wght <- list(a_wght)
    }
    else {
      if (length(a_wght) == 1) {
        a_wght <- rep(a_wght, nlevel)
      }
      a_wght <- as.list(c(a_wght))
    }
  }
  if (length(a_wght) != nlevel) {
    stop("length of a_wght list differs than of nlevel")
  }
  attr(a_wght, "fastNormalize") <- FALSE
  attr(a_wght, "isotropic") <- isotropic
  return(a_wght)
}

#'
#' Internal function: A wrapper of LatticeKrig::LKrigSetupAwght.LKRectangle
#'
#' @keywords internal
#' @param LKinfo LKinfo object
#' @param ... Not used directly.
#' @return list
#'
setRectangleAwght <- function(LKinfo, ...) {
  is_awght_object <- !is.null(LKinfo$a.wghtObject)
  if (is_awght_object) {
    LKinfo$a.wght <- setDefaultAwght(LKinfo)
  }
  a_wght <- LKinfo$a.wght
  n_level <- LKinfo$nlevel
  mx <- LKinfo$latticeInfo$mx
  
  if (length(a_wght) == 1) {
    a_wght <- as.list(rep(a_wght, n_level))
  }
  
  stationary <- rep(NA, n_level)
  first_order <- rep(NA, n_level)
  isotropic <- rep(NA, n_level)
  
  for (k in 1:length(a_wght)) {
    length_awght <- length(a_wght[[k]])
    dim_a_wght <- dim(a_wght[[k]])
    
    is_a_wght_matrix <- !is.matrix(a_wght[[k]])
    stationary[k] <- is_a_wght_matrix
    isotropic[k] <- ifelse(is_a_wght_matrix,
                           length_awght == 1,
                           dim_a_wght[2] == 1)
    if (is_a_wght_matrix) {
      first_order[k] <- length_awght == 1
    }
    else {
      is_dimenstion_matched <- (length(dim_a_wght) == 2) &
        (dim_a_wght[1] == mx[k, 1] * mx[k, 2]) &
        ((dim_a_wght[2] == 1) | (dim_a_wght[2] == 9))
      if (!is_dimenstion_matched) {
        stop(
          paste0(
            "Mismatched: a.wght matrix at level ",
            k,
            " has  dimensions:",
            dim_a_wght,
            " compare to lattice: ",
            mx[k,]
          )
        )
      }
      first_order[k] <- dim_a_wght[2] == 1
    }
  }
  
  RBF <- LKinfo$basisInfo$BasisFunction
  
  is_fast_normalization <- all(stationary) &
    all(first_order) &
    all(!is.na(unlist(a_wght))) &
    (RBF == "WendlandFunction") &
    (LKinfo$basisInfo$BasisType == "Radial")
  if (!is.null(LKinfo$setupArgs$BCHook)) {
    is_fast_normalization <- FALSE
  }
  if (is_fast_normalization) {
    attr(a_wght, which = "fastNormDecomp") <-
      LKRectangleSetupNormalization(mx,
                                    a_wght)
  }
  
  attr(a_wght, which = "fastNormalize") <- is_fast_normalization
  attr(a_wght, which = "first.order") <- first_order
  attr(a_wght, which = "stationary") <- stationary
  attr(a_wght, which = "isotropic") <- isotropic
  attr(a_wght, which = "a.wghtAsObject") <- is_awght_object
  
  return(a_wght)
}

#'
#' Internal function: A modifier of LatticeKrig::LKrig.basis
#'
#' @keywords internal
#' @param x1 A matrix
#' @param LKinfo LKinfo object
#' @return A matrix
#'
calculateLatticeKrigBasis <- function(x1, LKinfo) {
  x1 <- as.matrix(x1)
  
  PHI <- NULL
  basis_delta <- LKinfo$latticeInfo$delta * LKinfo$basisInfo$overlap
  for (l in 1:LKinfo$nlevel) {
    centers <- LKinfo$latticeInfo$grid[[l]]
    if (LKinfo$LKGeometry == "LKBox")
      class(centers) <- "gridList"
    
    PHItemp <- Radial.basis(
      x1,
      centers,
      basis_delta[l],
      max.points = LKinfo$basisInfo$max.points,
      mean.neighbor = LKinfo$basisInfo$mean.neighbor,
      BasisFunction = get(LKinfo$basisInfo$BasisFunction),
      distance.type = LKinfo$distance.type
    )
    # Normalization
    wght <- normalizeBasis(LKinfo, level = l, PHI = PHItemp)
    indZero <- wght == 0
    if (any(indZero)) {
      warning("Some normalization weights are zero")
    }
    wght[indZero] <- 1.0
    if (nrow(x1) > 1) {
      PHItemp <- diag.spam(1 / sqrt(wght)) %*% PHItemp
    }
    else {
      PHItemp@entries <- PHItemp@entries / sqrt(wght)
    }
    
    if (length(LKinfo$alpha[[l]]) > 1) {
      PHItemp <- diag.spam(sqrt(LKinfo$alpha[[l]])) %*% PHItemp
    }
    else {
      PHItemp <- sqrt(LKinfo$alpha[[l]]) * PHItemp
    }
    PHI <- spam::cbind.spam(PHI, PHItemp)
  }
  
  return(PHI)
}

#'
#' Internal function: A modifier of LatticeKrig::LKrigNormalizeBasis
#'
#' @keywords internal
#' @param x1 A matrix
#' @param LKinfo LKinfo object
#' @return A matrix
#'
normalizeBasis <- function(LKinfo, level, PHI) {
  if (LKinfo$LKGeometry == "LKInterval") {
    tempB <- calculateSARForOneDimLocation(LKinfo, level)
  } else if (LKinfo$LKGeometry == "LKRectangle") {
    tempB <- calculateSARForTowDimLocation(LKinfo, level)
  } else {
    tempB <- calculateSARForThreeDimLocation(LKinfo, level)
  }
  
  # tempB is in spind format
  tempB <-
    spam(tempB[c("ind", "ra")], nrow = tempB$da[1], ncol = tempB$da[2])
  # quadratic form involves applying inverse precision matrix to basis function evaluted at
  # locations for evaluation
  wght <- LKrig.quadraticform(t(tempB) %*% tempB,
                              PHI = PHI,
                              choleskyMemory = LKinfo$choleskyMemory)
  return(wght)
}

#'
#' Internal function: A modifier of LatticeKrig::LKrigSAR.LKInterval
#'
#' @keywords internal
#' @param LKinfo LKinfo object
#' @param level an integer
#' @return A list
#'
calculateSARForOneDimLocation <- function(LKinfo, level) {
  m <- LKinfo$latticeInfo$mLevel[level]
  a.wght <- (LKinfo$a.wght)[[level]]
  if (length(a.wght) == 1) {
    a.wght <- rep(a.wght, m)
  }
  if (length(a.wght) != m) {
    cat("Level, m, length( a.wght): ", level, m, length(a.wght), fill = TRUE)
    stop("a.wght wrong length")
  }
  da <- c(m, m)
  ra <- c(a.wght, rep(-1, (m - 1)), rep(-1, (m - 1)))
  Bi <- c(1:m, 2:m, 1:(m - 1))
  Bj <- c(1:m, 1:(m - 1), 2:m)
  return(list(
    ind = cbind(Bi, Bj),
    ra = ra,
    da = da
  ))
}

#'
#' Internal function: A modifier of LatticeKrig::LKrigSAR.LKBox
#'
#' @keywords internal
#' @param LKinfo LKinfo object
#' @param level an integer
#' @return A list
#'
calculateSARForTowDimLocation <- function(LKinfo, level) {
  m <- LKinfo$latticeInfo$mLevel[level]
  a.wght <- (LKinfo$a.wght)[[level]]
  if (length(a.wght) > 1) {
    stop("a.wght must be constant")
  }
  da <- c(m, m)
  
  ra <- c(rep(a.wght, m), rep(-1, m * 6))
  Bi <- c(rep(1:m, 7))
  Bindex <- array(1:m, LKinfo$latticeInfo$mx[level,])
  Bj <- c(
    1:m,
    c(shiftArray(Bindex, c(-1, 0, 0))),
    c(shiftArray(Bindex, c(1, 0, 0))),
    c(shiftArray(Bindex, c(0,-1, 0))),
    c(shiftArray(Bindex, c(0, 1, 0))),
    c(shiftArray(Bindex, c(0, 0,-1))),
    c(shiftArray(Bindex, c(0, 0, 1)))
  )
  in_range <- !is.na(Bj)
  Bi <- Bi[in_range]
  Bj <- Bj[in_range]
  ra <- ra[in_range]
  return(list(
    ind = cbind(Bi, Bj),
    ra = ra,
    da = da
  ))
}

#'
#' Internal function: A modifier of LatticeKrig::LKrigSAR.LKRectangle
#'
#' @keywords internal
#' @param LKinfo LKinfo object
#' @param level an integer
#' @return A list
#'
calculateSARForThreeDimLocation <- function(LKinfo, level) {
  mx1 <- LKinfo$latticeInfo$mx[level, 1]
  mx2 <- LKinfo$latticeInfo$mx[level, 2]
  m <- mx1 * mx2
  a.wght <- (LKinfo$a.wght)[[level]]
  
  stationary <- (attr(LKinfo$a.wght, "stationary"))[level]
  first.order <- attr(LKinfo$a.wght, "first.order")[level]
  isotropic <- attr(LKinfo$a.wght, "isotropic")[level]
  distance.type <- LKinfo$distance.type
  if (all(stationary & isotropic)) {
    if (any(unlist(a.wght) < 4)) {
      stop("a.wght less than 4")
    }
  }
  
  dim.a.wght <- dim(a.wght)
  index <- c(5, 4, 6, 2, 8, 3, 9, 1, 7)
  da <- as.integer(c(m, m))
  if (first.order) {
    ra <- array(NA, c(mx1 * mx2, 5))
    ra[, 1] <- a.wght
    ra[, 2:5] <- -1
  }
  else {
    ra <- array(NA, c(mx1 * mx2, 9))
    for (kk in 1:9) {
      if (stationary) {
        ra[, kk] <- a.wght[index[kk]]
      }
      else {
        ra[, kk] <- a.wght[, index[kk]]
      }
    }
  }
  
  Bi <- rep(1:m, 5)
  i.c <- matrix(1:m, nrow = mx1, ncol = mx2)
  Bj <- c(
    i.c,
    LKrig.shift.matrix(i.c, 0,-1),
    LKrig.shift.matrix(i.c, 0, 1),
    LKrig.shift.matrix(i.c, 1, 0),
    LKrig.shift.matrix(i.c,-1, 0)
  )
  if (!first.order) {
    Bi <- c(Bi, rep(1:m, 4))
    Bj <- c(
      Bj,
      LKrig.shift.matrix(i.c, 1, 1),
      LKrig.shift.matrix(i.c,-1, 1),
      LKrig.shift.matrix(i.c, 1,-1),
      LKrig.shift.matrix(i.c,-1,-1)
    )
  }
  
  good <- !is.na(Bj)
  Bi <- as.integer(Bi[good])
  Bj <- as.integer(Bj[good])
  ra <- c(ra)[good]
  
  if (!is.null(LKinfo$setupArgs$BCHook)) {
    M <- da[1]
    for (i in 1:M) {
      rowI <- which(Bi == i)
      rowNN <- rowI[-1]
      ra[rowNN] <- 4 * ra[rowNN] / length(rowNN)
    }
  }
  return(list(
    ind = cbind(Bi, Bj),
    ra = ra,
    da = da
  ))
}

#'
#' Internal function: A modifier of LatticeKrig::LKArrayShift
#'
#' @keywords internal
#' @param array_object array
#' @param shift_index one-dim array
#'
shiftArray <- function(array_object, shift_index) {
  shape <- dim(array_object)
  reshaped_array <- array(NA, shape)
  if (any(abs(shift_index) > shape)) {
    stop("shift exceeds array dimensions")
  }
  
  index_list_source <- index_list_target <- NULL
  
  for (k in 1:length(shape)) {
    index_list_source <- c(index_list_source, list(c(1:shape[k])))
    temp_index <-
      (0:(shape[k] - 1) + shift_index[k]) %% shape[k] + 1
    temp_index[(temp_index < 1) | (temp_index > shape[k])] <- NA
    index_list_target <- c(index_list_target, list(temp_index))
  }
  index_source <- as.matrix(expand.grid(index_list_source))
  index_target <- as.matrix(expand.grid(index_list_target))
  in_range <- rowSums(is.na(index_target)) == 0
  reshaped_array[index_target[in_range,]] <-
    array_object[index_source[in_range,]]
  return(reshaped_array)
}
