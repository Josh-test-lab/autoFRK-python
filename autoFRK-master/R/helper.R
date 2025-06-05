#'
#' Internal function: eigen-decomposition in decreasing order
#'
#' @keywords internal.
#' @param mat A matrix.
#' @return A list.
#'
eigenDecomposeInDecreasingOrder <- function(mat) {
  obj <- eigenDecompose(mat)
  obj$value <- rev(obj$value)
  obj$vector <- obj$vector[, ncol(mat):1]
  return(obj)
}

#'
#' Internal function: calculate the log determinant for the likelihood use.
#'
#' @keywords internal.
#' @param R A p x p positive-definite matrix.
#' @param L A p x K matrix.
#' @param K A numeric.
#' @return A numeric.
#'
calculateLogDeterminant <- function(R, L, K) {
  first_part_determinant <-
    logDeterminant(diag(1, K) + t(L) %*% solve(R) %*% L)
  second_part_determinant <- logDeterminant(R)
  return(first_part_determinant + second_part_determinant)
}

#'
#' Internal function: compute a negative log-likelihood (-2*log(likelihood))
#'
#' @keywords internal.
#' @param data  An \emph{n} by \emph{T} data matrix (NA allowed) with
#' \eqn{z[t]} as the \emph{t}-th column.
#' @param Fk An \emph{n} by \emph{K} matrix of basis function values with
#'  each column being a basis function taken values at \code{loc}.
#' @param M A \emph{K} by \emph{K} symmetric matrix.
#' @param s A scalar.
#' @param Depsilon An \emph{n} by \emph{n} diagonal matrix.
#' @return A numeric.
#'
computeLikelihood <- function(data, Fk, M, s, Depsilon) {
  data <- as.matrix(data)
  non_missing_points_matrix <- as.matrix(!is.na(data))
  num_columns <- NCOL(data)
  
  n2loglik <- sum(non_missing_points_matrix) * log(2 * pi)
  R <- toSparseMatrix(s * Depsilon)
  eg <- eigenDecompose(M)
  
  K <- NCOL(Fk)
  L <-
    Fk %*% eg$vector %*% diag(sqrt(pmax(eg$value, 0)), K) %*% t(eg$vector)
  
  for (t in 1:num_columns) {
    zt <- data[non_missing_points_matrix[, t], t]
    Rt <-
      R[non_missing_points_matrix[, t], non_missing_points_matrix[, t]]
    Lt <- L[non_missing_points_matrix[, t],]
    n2loglik <-
      n2loglik + calculateLogDeterminant(Rt, Lt, K) + sum(zt * invCz(Rt, Lt, zt))
  }
  return(n2loglik)
}

#'
#' Internal function: select basis functions
#'
#' @keywords internal.
#' @param data  An \emph{n} by \emph{T} data matrix (NA allowed) with
#' \eqn{z[t]} as the \emph{t}-th column.
#' @param loc \emph{n} by \emph{d} matrix of coordinates corresponding to \emph{n} locations.
#' @param D A diagonal matrix.
#' @param maxit An integer for the maximum number of iterations used in indeMLE.
#' @param avgtol A numeric for average tolerance used in indeMLE.
#' @param max_rank An integer of the maximum of K values.
#' @param sequence_rank An array of K values
#' @param method A character of a list of characters.
#' @param num_neighbors An integer.
#' @param max_knot An integer for the maximum number of knots
#' @param DfromLK An \emph{n} by \emph{n} diagonal matrix.
#' @param Fk An \emph{n} by \emph{K} matrix of basis function values with
#'  each column being a basis function taken values at \code{loc}.
#' @return An mrts object with 6 attributes
#'
selectBasis <- function(data,
                        loc,
                        D = NULL,
                        maxit = 50,
                        avgtol = 1e-6,
                        max_rank = NULL,
                        sequence_rank = NULL,
                        method = c("fast", "EM"),
                        num_neighbors = 3,
                        max_knot = 5000,
                        DfromLK = NULL,
                        Fk = NULL) {
  data <- as.matrix(data)
  are_all_missing_in_columns <- apply(!is.na(data), 2, sum) == 0
  if (any(are_all_missing_in_columns)) {
    data <- as.matrix(data[,!are_all_missing_in_columns])
  }
  if (is.null(D))
    D <- diag.spam(NROW(data))
  
  loc <- as.matrix(loc)
  d <- NCOL(loc)
  is_data_with_missing_values <- any(is.na(data))
  na_rows <- which(rowSums(as.matrix(!is.na(data))) == 0)
  pick <- 1:NROW(data)
  if (length(na_rows) > 0) {
    data <- as.matrix(data[-na_rows,])
    D <- D[-na_rows,-na_rows]
    pick <- pick[-na_rows]
    is_data_with_missing_values <- any(is.na(data))
  }
  
  N <- length(pick)
  klim <- min(N, round(10 * sqrt(N)))
  if (N < max_knot) {
    knot <- loc[pick,]
  } else {
    knot <- subKnot(loc[pick,], min(max_knot, klim))
  }
  
  if (!is.null(max_rank)) {
    max_rank <- round(max_rank)
  } else {
    max_rank <-
      ifelse(!is.null(sequence_rank), round(max(sequence_rank)), klim)
  }
  
  if (!is.null(sequence_rank)) {
    K <- unique(round(sequence_rank))
    if (max(K) > max_rank) {
      stop("maximum of sequence_rank is larger than max_rank!")
    }
    if (all(K <= d)) {
      stop("Not valid sequence_rank!")
    } else if (any(K < (d + 1))) {
      warning(
        "The minimum of sequence_rank can not less than ",
        d + 1,
        ". Too small values will be ignored."
      )
    }
    K <- K[K > d]
  } else {
    K <-
      unique(round(seq(d + 1, max_rank, by = max_rank ^ (1 / 3) * d)))
    if (length(K) > 30) {
      K <- unique(round(seq(d + 1, max_rank, l = 30)))
    }
  }
  
  if (is.null(Fk)) {
    Fk <- mrts(knot, max(K), loc, max_knot)
  }
  AIC_list <- rep(Inf, length(K))
  method <- match.arg(method)
  num_data_columns <- NCOL(data)
  
  if ((method == "EM") & (is.null(DfromLK))) {
    for (k in 1:length(K)) {
      AIC_list[k] <- indeMLE(data,
                             Fk[pick, 1:K[k]],
                             D,
                             maxit,
                             avgtol,
                             wSave = FALSE,
                             verbose = FALSE)$negloglik
    }
  } else {
    if (is_data_with_missing_values) {
      for (tt in 1:num_data_columns) {
        is_cell_missing_in_a_column <- is.na(data[, tt])
        if (!any(is_cell_missing_in_a_column)) {
          next
        }
        cidx <- which(!is_cell_missing_in_a_column)
        nnidx <- FNN::get.knnx(loc[cidx,],
                               matrix(loc[is_cell_missing_in_a_column,], ncol = dim(loc)[2]),
                               k = num_neighbors)
        nnidx <- array(cidx[nnidx$nn.index], dim(nnidx$nn.index))
        nnval <- array((data[, tt])[nnidx], dim(nnidx))
        data[is_cell_missing_in_a_column, tt] <- rowMeans(nnval)
      }
    }
    if (is.null(DfromLK)) {
      iD <- solve(D)
      iDFk <- iD %*% Fk[pick,]
      iDZ <- iD %*% data
    }
    else {
      wX <- DfromLK$wX[pick,]
      G <- t(DfromLK$wX) %*% DfromLK$wX + DfromLK$lambda *
        DfromLK$Q
      weight <- DfromLK$weights[pick]
      wwX <- diag.spam(sqrt(weight)) %*% wX
      wXiG <- (wwX) %*% solve(G)
      iDFk <- weight * Fk[pick,] - wXiG %*% (t(wwX) %*%
                                               as.matrix(Fk[pick,]))
      iDZ <- weight * data - wXiG %*% (t(wwX) %*% as.matrix(data))
    }
    sample_covariance_trace <-
      sum(rowSums(as.matrix(iDZ) * data)) / num_data_columns
    for (k in 1:length(K)) {
      inverse_square_root_matrix <-
        getInverseSquareRootMatrix(Fk[pick, 1:K[k]], iDFk[, 1:K[k]])
      ihFiD <- inverse_square_root_matrix %*% t(iDFk[, 1:K[k]])
      matrix_JSJ <- tcrossprod(ihFiD %*% data) / num_data_columns
      matrix_JSJ <- (matrix_JSJ + t(matrix_JSJ)) / 2
      AIC_list[k] <- cMLE(
        Fk = Fk[pick, 1:K[k]],
        num_columns = num_data_columns,
        sample_covariance_trace = sample_covariance_trace,
        inverse_square_root_matrix = inverse_square_root_matrix,
        matrix_JSJ = matrix_JSJ
      )$negloglik
    }
  }
  
  df <-
    (K * (K + 1) / 2 + 1) * (K <= num_data_columns) + (K * num_data_columns + 1 - num_data_columns * (num_data_columns - 1) / 2) * (K > num_data_columns)
  AIC_list <- AIC_list + 2 * df
  Kopt <- K[which.min(AIC_list)]
  out <- Fk[, 1:Kopt]
  
  dimnames(Fk) <- NULL
  aname <- names(attributes(Fk))
  attributes(out) <-
    c(attributes(out), attributes(Fk)[setdiff(aname, "dim")])
  
  return(out)
}

#'
#' Internal function: solve v parameter
#'
#' @keywords internal.
#' @param d An array of nonnegative values.
#' @param s A positive numeric.
#' @param sample_covariance_trace A positive numeric.
#' @param n An integer. Sample size.
#' @return A numeric.
#'
estimateV <- function(d, s, sample_covariance_trace, n) {
  if (max(d) < max(sample_covariance_trace / n, s)) {
    return(max(sample_covariance_trace / n - s, 0))
  }
  
  k <- length(d)
  cumulative_d_values <- cumsum(d)
  ks <- 1:k
  if (k == n)
    ks[n] <- n - 1
  
  eligible_indexs <-
    which(d > ((
      sample_covariance_trace - cumulative_d_values
    ) / (n - ks)))
  L <- max(eligible_indexs)
  if (L >= n)
    L <- n - 1
  return(max((
    sample_covariance_trace - cumulative_d_values[L]
  ) / (n - L) - s, 0))
}

#'
#' Internal function: estimate eta parameter
#'
#' @keywords internal.
#' @param d An array of nonnegative values.
#' @param s A positive numeric.
#' @param v A positive numeric.
#' @return A numeric.
#'
estimateEta <- function(d, s, v) {
  return(pmax(d - s - v, 0))
}

#'
#' Internal function: estimate negative log-likelihood
#'
#' @keywords internal.
#' @param d An array of nonnegative values.
#' @param s A positive numeric.
#' @param v A positive numeric.
#' @param sample_covariance_trace A positive numeric.
#' @param sample_size An integer. Sample size.
#' @return A numeric.
#'
neg2llik <-
  function(d,
           s,
           v,
           sample_covariance_trace,
           sample_size) {
    k <- length(d)
    eta <- estimateEta(d, s, v)
    if (max(eta / (s + v)) > 1e20) {
      return(Inf)
    } else {
      return(
        sample_size * log(2 * pi) + sum(log(eta + s + v)) + log(s + v) * (sample_size - k) + 1 / (s + v) * sample_covariance_trace - 1 / (s + v) * sum(d * eta / (eta + s + v))
      )
    }
  }

#'
#' Internal function: compute a negative likelihood
#'
#' @keywords internal.
#' @param nrow_Fk An integer. The number of rows of Fk.
#' @param ncol_Fk An integer. The number of columns of Fk.
#' @param s An integer.
#' @param p A positive integers. The number of columns of data.
#' @param matrix_JSJ A multiplication matrix
#' @param sample_covariance_trace A positive numeric.
#' @param vfixed A numeric
#' @param ldet A numeric. A log determinant.
#' @return A list.
#'
computeNegativeLikelihood <- function(nrow_Fk,
                                      ncol_Fk,
                                      s,
                                      p,
                                      matrix_JSJ,
                                      sample_covariance_trace,
                                      vfixed = NULL,
                                      ldet = 0) {
  if (!isSymmetric.matrix(matrix_JSJ)) {
    stop("Please input a symmetric matrix")
  }
  if (ncol(matrix_JSJ) < ncol_Fk) {
    stop(paste0(
      "Please input the rank of a matrix larger than ncol_Fk = ",
      ncol_Fk
    ))
  }
  decomposed_JSJ <- eigenDecomposeInDecreasingOrder(matrix_JSJ)
  eigenvalues_JSJ <- decomposed_JSJ$value[1:ncol_Fk]
  eigenvectors_JSJ <- as.matrix(decomposed_JSJ$vector)[, 1:ncol_Fk]
  v <- ifelse(
    is.null(vfixed),
    estimateV(eigenvalues_JSJ,
              s,
              sample_covariance_trace,
              nrow_Fk),
    vfixed
  )
  d <- pmax(eigenvalues_JSJ, 0)
  d_hat <- estimateEta(d, s, v)
  negative_log_likelihood <-
    neg2llik(d, s, v, sample_covariance_trace, nrow_Fk) * p + ldet * p
  return(
    list(
      negative_log_likelihood = negative_log_likelihood,
      P = eigenvectors_JSJ,
      v = v,
      d_hat = d_hat
    )
  )
}

#'
#' Internal function: maximum likelihood estimate with the likelihood
#'
#' @keywords internal.
#' @param Fk1 An \emph{n} by \emph{K} matrix
#' @param Fk2 An \emph{n} by \emph{K} matrix
#' @param data  An \emph{n} by \emph{T} data matrix
#' @param S An \emph{n} by \emph{n} matrix
#' @return A list.
#'
computeProjectionMatrix <- function(Fk1,
                                    Fk2,
                                    data,
                                    S = NULL) {
  num_columns <- NCOL(data)
  inverse_square_root_matrix <- getInverseSquareRootMatrix(Fk1, Fk2)
  inverse_square_root_on_Fk2 <-
    inverse_square_root_matrix %*% t(Fk2)
  if (is.null(S)) {
    matrix_JSJ <-
      tcrossprod(inverse_square_root_on_Fk2 %*% data) / num_columns
  }
  else {
    matrix_JSJ <-
      (inverse_square_root_on_Fk2 %*% S) %*% t(inverse_square_root_on_Fk2)
  }
  matrix_JSJ <- (matrix_JSJ + t(matrix_JSJ)) / 2
  return(
    list(
      inverse_square_root_matrix = inverse_square_root_matrix,
      matrix_JSJ = matrix_JSJ
    )
  )
}

#'
#' Internal function: check if a numeric-like object is diagonal
#'
#' @keywords internal
#' @param object An R object
#' @return logical
#'
isDiagonal <- function(object) {
  if (is.numeric(object) & (length(object) == 1)) {
    return(TRUE)
  }
  if (is.matrix(object)) {
    return(sum(abs(diag(diag(
      object
    )) - object)) < .Machine$double.eps)
  }
  else if (is.spam(object)) {
    x <- diag.spam(diag.of.spam(object), NROW(object))
    return(sum(abs(x - object)) < .Machine$double.eps)
  } else {
    return(FALSE)
  }
}

#'
#' Internal function: sampling knots
#'
#' @keywords internal
#' @param x A matrix or an array
#' @param nknot A location matrix
#' @param xrng An array including two integers
#' @return sampling knots
#'
subKnot <- function(x,
                    nknot,
                    xrng = NULL,
                    nsamp = 1) {
  x <- apply(as.matrix(x), 2, sort)
  xdim <- dim(x)
  
  if (is.null(xrng)) {
    xrng <- apply(x, 2, range)
  }
  # To-do: move out the function based on SRP
  mysamp <- function(z_and_id) {
    z <- as.double(names(z_and_id))
    if (length(z) == 1L) {
      return(z)
    }
    else {
      set.seed(mean(z_and_id))
      return(sample(z, size = min(nsamp, length(z))))
    }
  }
  
  rng <- sqrt(xrng[2,] - xrng[1,])
  rng[rng == 0] <- min(rng[rng > 0]) / 5
  rng <- rng * 10 / min(rng)
  rng_max_index <- which.max(rng)
  
  nmbin <- round(exp(log(rng) * log(nknot) / sum(log(rng))))
  nmbin <- pmax(2, nmbin)
  while (prod(nmbin) < nknot) {
    nmbin[rng_max_index] <- nmbin[rng_max_index] + 1
  }
  
  gvec <- matrix(1, nrow = xdim[1])
  cnt <- 0
  while (length(unique(gvec)) < nknot) {
    nmbin <- nmbin + cnt
    kconst <- 1
    gvec <- matrix(1, nrow = xdim[1])
    for (kk in 1:xdim[2]) {
      grp <-
        pmin(round((nmbin[kk] - 1) * ((x[, kk] - xrng[1, kk]) / (xrng[2, kk] - xrng[1, kk]))),
             nmbin[kk] - 1L)
      if (length(unique(grp)) < nmbin[kk]) {
        brk <- quantile(x[, kk], seq(0, 1, l = nmbin[kk] + 1))
        brk[1] <- brk[1] - 0.1 ^ 8
        grp <- as.double(cut(x[, kk], brk))
      }
      gvec <- gvec + kconst * grp
      kconst <- kconst * nmbin[kk]
    }
    cnt <- cnt + 1
  }
  # To-do: refactor the following four lines
  gvec <- as.factor(gvec)
  gid <- as.double(as.character(gvec))
  names(gid) <- 1:xdim[1]
  index <- unlist(tapply(gid, gvec, mysamp))
  return(x[index,])
}

#'
#' Internal function: convert to a sparse matrix
#'
#' @keywords internal
#' @param mat A matrix
#' @return An array of indeces
#'
fetchNonZeroIndexs <- function(mat) {
  if (!is.matrix(mat)) {
    stop(paste0(c(
      "Wrong matrix format, but got ", class(mat)
    )))
  }
  db <- tempfile()
  NR <- NROW(mat)
  NC <- NCOL(mat)
  f <- fm.create(db, NR, NC)
  f[, 1:NCOL(mat)] <- mat
  j <- sapply(1:NC, function(j)
    which(f[, j] != 0))
  ridx <- unlist(j)
  k <- sapply(1:NR, function(k)
    rbind(k, which(f[k,] != 0)))
  kk <- matrix(unlist(k), ncol = 2, byrow = T)
  cidx <- sort(kk[, 2])
  where <- (cidx - 1) * NR + ridx
  closeAndDeleteFiles(f)
  return(where)
}

#'
#' Internal function: convert to a sparse matrix
#'
#' @keywords internal
#' @param mat A matrix or a dataframe
#' @param verbose A boolean
#' @return sparse matrix
#'
toSparseMatrix <- function(mat, verbose = FALSE) {
  if (is.spam(mat)) {
    if (verbose)
      message("The input is already a sparse matrix")
    return(mat)
  }
  if (!(is.data.frame(mat) || is.matrix(mat))) {
    stop(paste0(c(
      "Wrong format for toSparseMatrix, but got ", class(mat)
    )))
  }
  if (is.data.frame(mat)) {
    mat <- as.matrix(mat)
  }
  
  if (length(mat) > 1e8) {
    warnings("Use sparse matrix as input instead; otherwise it could take a very long time!")
    non_zero_indexs <- fetchNonZeroIndexs(mat)
  } else {
    non_zero_indexs <- which(mat != 0)
  }
  matrix_dim <- dim(mat)
  sparse_matrix <- spam(0, nrow = NROW(mat), ncol = NCOL(mat))
  for (j in 1:matrix_dim[2]) {
    flat_indexs <-
      non_zero_indexs[non_zero_indexs >= ((j - 1) * matrix_dim[1] + 1) &
                        non_zero_indexs <= (j * matrix_dim[1])]
    if (length(flat_indexs) > 0) {
      row_indexs <- flat_indexs - ((j - 1) * matrix_dim[1])
      sparse_matrix[row_indexs, j] <- mat[row_indexs, j]
    }
  }
  return(sparse_matrix)
}

#'
#' Internal function: internal matrix calcuation (will be deprecated)
#'
#' @keywords internal
#' @param R A p x p matrix
#' @param L A p x K matrix
#' @param z An array with length p or 1 x p matrix
#' @return A 1 x p matrix
#'
ZinvC <- function(R, L, z) {
  K <- NCOL(L)
  iR <- solve(R)
  ZiR <- z %*% iR
  left <- ZiR %*% L %*% solve(diag(1, K) + t(L) %*% iR %*% L) %*%
    t(L)
  return(ZiR - left %*% iR)
}

#'
#' Internal function: internal matrix calcuation
#'
#' @keywords internal
#' @param R A p x p matrix
#' @param L A p x K matrix
#' @param z An array with length p or 1 x p matrix
#' @return A 1 x p matrix
#'
invCz <- function(R, L, z) {
  K <- NCOL(L)
  iR <- solve(R)
  iRZ <- iR %*% z
  right <-
    L %*% solve(diag(1, K) + as.matrix(t(L) %*% iR %*% L)) %*% (t(L) %*% iRZ)
  
  return(iRZ - iR %*% right)
}

#'
#' Internal function: convert a matrix to positive definite
#'
#' @keywords internal
#' @param mat A matrix or a dataframe
#' @return A positive-definite matrix
#'
convertToPositiveDefinite <- function(mat) {
  v <- try(min(eigen(mat, only.values = T)$value), silent = TRUE)
  if (is(v, "try-error") || !isSymmetric.matrix(mat)) {
    mat <- (mat + t(mat)) / 2
    v <- min(eigen(mat, only.values = T)$value)
  }
  if (v <= 0) {
    mat <- mat + diag(max(0,-v) + 0.1 ^ 7.5, NROW(mat))
  }
  return(mat)
}

#'
#' Internal function: A wendland function with k = 2 and l = 4
#'
#' @keywords internal
#' @param r A matrix of 2 or 3 d locations
#' @return A matrix
#'
wendland <- function(r) {
  # Ref: https://arxiv.org/pdf/1203.5696.pdf
  if (any(r < 0)) {
    stop(paste(c("Invalid values: ", r[which(r < 0)]), " "))
  }
  return(r < 1) * (1 - r) ^ 6 * (35 * r ^ 2 + 18 * r + 3)
}

#'
#' Internal function: remove white spaces for a given string
#'
#' @keywords internal
#' @param x A character
#' @return A character
#'
removeWhitespace <-
  function(x)
    gsub("(^[[:space:]]+|[[:space:]]+$)", "", x)

#'
#' Internal function: log-determinant of a sqaure matrix
#'
#' @keywords internal
#' @param mat A sqaure matrix.
#' @return A numeric.
#'
logDeterminant <-
  function(mat)
    spam::determinant(mat, logarithm = TRUE)$modulus[1]



#'
#' Internal function: LKpeon
#'
#' @keywords internal
#' @param M A \emph{K} by \emph{K} symmetric matrix.
#' @param s A scalar.
#' @param Fk An \emph{n} by \emph{K} matrix of basis function values with
#'  each column being a basis function taken values at \code{loc}.
#' @param basis An \emph{n} by \emph{K} matrix
#' @param weight A vector
#' @param phi1 A matrix produced by LatticeKrig basis fucntions
#' @param phi0 A matrix produced by LatticeKrig basis fucntions with new locations
#' @param Q A precision matrix
#' @param lambda A scalar.
#' @param phi0P A projection matrix of phi1
#' @param L An \emph{n} by \emph{K} matrix
#' @param Data An \emph{n} by \emph{T} data matrix
#' @param only.wlk A logic.
#' @param only.se A logic.
#' @return A list.
#'
LKpeon <- function(M,
                   s,
                   Fk,
                   basis,
                   weight,
                   phi1,
                   phi0,
                   Q,
                   lambda,
                   phi0P,
                   L = NULL,
                   data = NULL,
                   only.wlk = FALSE,
                   only.se = FALSE) {
  wwX <- diag.spam(weight) %*% phi1
  wXiG <- (wwX) %*% solve(t(wwX) %*% phi1 + lambda *
                            Q)
  fM <- Fk %*% M
  if (is.null(L)) {
    dec <- eigen(M)
    L <- Fk %*% dec$vector %*% diag.spam(sqrt(pmax(dec$value,
                                                   0)), NROW(M))
    L <- as.matrix(L)
  }
  iDL <- weight * L - wXiG %*% (t(wwX) %*% L)
  iDFk <- weight * Fk - wXiG %*% (t(wwX) %*% as.matrix(Fk))
  itmp <- solve(diag(1, NCOL(L)) + t(L) %*% iDL / s)
  ihL <- chol(itmp) %*% t(L)
  iiLiD <- itmp %*% t(iDL / s)
  if (only.wlk) {
    if (!is.null(data)) {
      LiiLiDZ <- L %*% (iiLiD %*% data)
      w <- M %*% t(iDFk) %*% data / s - (M %*% t(iDFk / s)) %*%
        (LiiLiDZ)
      wlk <- t(wXiG) %*% data - t(wXiG) %*% (LiiLiDZ)
    } else {
      w <- NULL
      wlk <- NULL
    }
    return(list(w = w, wlk = wlk))
  }
  MFiS11 <- M %*% t(iDFk) / s - ((M %*% t(iDFk / s)) %*%
                                   L) %*% iiLiD
  FMfi <- basis %*% MFiS11
  p0Pp1 <- as.matrix(phi0P %*% t(phi1))
  se0 <- rowSums((basis %*% M) * basis) + rowSums(as.matrix(phi0P *
                                                              phi0)) / lambda * s
  se11 <- rowSums((FMfi %*% fM) * basis)
  se12 <- rowSums(p0Pp1 * (FMfi)) * s / lambda
  se13 <- se12
  se14 <- rowSums(as.matrix(phi0 %*% t(wXiG)) * p0Pp1) *
    s / lambda - colSums((ihL %*% wXiG %*% t(phi0)) ^ 2)
  se <- sqrt(pmax(se0 - (se11 + se12 + se13 + se14),
                  0))
  if (only.se) {
    return(se)
  } else {
    if (!is.null(data)) {
      w <- MFiS11 %*% data
      wlk <- t(wXiG) %*% data - t(wXiG) %*% L %*% (iiLiD %*% data)
    } else {
      w <- NULL
      wlk <- NULL
    }
    return(list(se = se,
                w = w,
                wlk = wlk))
  }
}
